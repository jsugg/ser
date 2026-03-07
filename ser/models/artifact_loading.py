"""Model artifact candidate discovery and load-resolution helpers."""

from __future__ import annotations

import logging
import pickle
import warnings
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast

type ArtifactFormat = Literal["pickle", "skops"]


class ModelCandidateLike(Protocol):
    """Structural contract for model artifact candidates."""

    @property
    def path(self) -> Path: ...

    @property
    def artifact_format(self) -> ArtifactFormat: ...


class LoadedModelLike(Protocol):
    """Structural contract for loaded model metadata access."""

    @property
    def artifact_metadata(self) -> dict[str, object] | None: ...


ModelCandidateT = TypeVar("ModelCandidateT", bound=ModelCandidateLike)
LoadedModelT = TypeVar("LoadedModelT", bound=LoadedModelLike)
SettingsT = TypeVar("SettingsT")
ResolveModelForLoading = Callable[..., tuple[ModelCandidateT, LoadedModelT]]
EnvelopeLoadedModelT = TypeVar("EnvelopeLoadedModelT")
SecureLoadedModelT = TypeVar("SecureLoadedModelT")
PickleLoadedModelT = TypeVar("PickleLoadedModelT")


_ARTIFACT_DISCOVERY_PATTERNS: tuple[tuple[str, ArtifactFormat], ...] = (
    ("ser_model*.skops", "skops"),
    ("ser_model*.pkl", "pickle"),
)


class _SkopsGetUntrustedTypes(Protocol):
    def __call__(self, *, file: str) -> object: ...


class _SkopsLoad(Protocol):
    def __call__(self, file: str, *, trusted: list[object]) -> object: ...


def build_model_artifact_envelope(
    *,
    artifact_version: int,
    model: object,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Builds one versioned model artifact envelope payload."""
    return {
        "artifact_version": artifact_version,
        "model": model,
        "metadata": metadata,
    }


def deserialize_model_artifact_envelope(
    payload: object,
    *,
    artifact_version: int,
    model_instance_check: Callable[[object], bool],
    normalize_metadata: Callable[[dict[str, object]], dict[str, object]],
    read_positive_int: Callable[[dict[str, object], str], int],
    loaded_model_factory: Callable[
        [object, int | None, dict[str, object]],
        EnvelopeLoadedModelT,
    ],
) -> EnvelopeLoadedModelT:
    """Validates and unwraps one versioned model artifact envelope payload."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Model artifact payload must be a versioned dictionary envelope "
            f"(received {type(payload).__name__})."
        )

    resolved_artifact_version = payload.get("artifact_version")
    if resolved_artifact_version != artifact_version:
        raise ValueError(
            "Unsupported model artifact version "
            f"{resolved_artifact_version!r}; expected {artifact_version}. "
            "Regenerate artifacts with current training code."
        )

    model = payload.get("model")
    if not model_instance_check(model):
        raise ValueError(
            "Unexpected model object type in artifact envelope: "
            f"{type(model).__name__}."
        )

    metadata_obj = payload.get("metadata")
    if not isinstance(metadata_obj, dict):
        raise ValueError("Model artifact metadata is missing or invalid.")
    normalized_metadata = normalize_metadata(metadata_obj)
    expected_feature_size = read_positive_int(
        normalized_metadata,
        "feature_vector_size",
    )
    return loaded_model_factory(
        model,
        expected_feature_size,
        normalized_metadata,
    )


def load_secure_model_artifact(
    *,
    candidate_path: Path,
    model_instance_check: Callable[[object], bool],
    training_report_file: Path,
    read_training_report_feature_size: Callable[[Path], int | None],
    loaded_model_factory: Callable[[object, int | None], SecureLoadedModelT],
    import_module_fn: Callable[[str], object] = import_module,
) -> SecureLoadedModelT:
    """Loads one secure model artifact with strict trust and payload checks."""
    try:
        skops_module = import_module_fn("skops.io")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Secure model artifact found but `skops` is not installed."
        ) from err

    get_untrusted_types_raw = getattr(skops_module, "get_untrusted_types", None)
    load_raw = getattr(skops_module, "load", None)
    if not callable(get_untrusted_types_raw) or not callable(load_raw):
        raise RuntimeError(
            "Secure model artifact found but `skops` API is unavailable."
        )
    get_untrusted_types = cast(_SkopsGetUntrustedTypes, get_untrusted_types_raw)
    load_secure = cast(_SkopsLoad, load_raw)
    untrusted_types_raw = get_untrusted_types(file=str(candidate_path))
    if not isinstance(untrusted_types_raw, list | tuple | set | frozenset):
        raise ValueError("Secure model artifact trust metadata is invalid.")
    untrusted_types = set(untrusted_types_raw)
    if untrusted_types:
        raise ValueError(
            "Secure model artifact contains untrusted types; refusing automatic "
            f"trust for {candidate_path}."
        )

    payload = load_secure(str(candidate_path), trusted=[])
    if not model_instance_check(payload):
        raise ValueError(
            "Unexpected secure model payload type: "
            f"{type(payload).__name__}. Expected sklearn classifier/pipeline."
        )
    feature_size = read_training_report_feature_size(training_report_file)
    return loaded_model_factory(payload, feature_size)


def load_pickle_model_artifact(
    *,
    candidate_path: Path,
    deserialize_payload: Callable[[object], PickleLoadedModelT],
) -> PickleLoadedModelT:
    """Loads and deserializes one compatibility pickle artifact."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with candidate_path.open("rb") as model_fh:
            payload = pickle.load(model_fh)
    return deserialize_payload(payload)


def discover_model_candidates(
    folder: Path,
    *,
    candidate_factory: Callable[[Path, ArtifactFormat], ModelCandidateT],
) -> list[ModelCandidateT]:
    """Discovers model artifacts in a folder using SER naming conventions."""
    if not folder.exists():
        return []

    discovered: list[ModelCandidateT] = []
    for pattern, artifact_format in _ARTIFACT_DISCOVERY_PATTERNS:
        for path in sorted(folder.glob(pattern)):
            if path.is_file():
                discovered.append(candidate_factory(path, artifact_format))
    return discovered


def model_load_candidates(
    *,
    folder: Path,
    secure_model_file: Path,
    model_file: Path,
    candidate_factory: Callable[[Path, ArtifactFormat], ModelCandidateT],
) -> tuple[ModelCandidateT, ...]:
    """Returns model artifacts in preferred load order from primary storage."""
    discovered_primary = discover_model_candidates(
        folder,
        candidate_factory=candidate_factory,
    )
    ordered: tuple[ModelCandidateT, ...] = (
        candidate_factory(secure_model_file, "skops"),
        candidate_factory(model_file, "pickle"),
        *discovered_primary,
    )

    deduped: list[ModelCandidateT] = []
    seen: set[tuple[str, ArtifactFormat]] = set()
    for candidate in ordered:
        key: tuple[str, ArtifactFormat] = (
            str(candidate.path),
            candidate.artifact_format,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def resolve_model_for_loading_from_settings(
    settings: SettingsT,
    *,
    folder: Path,
    secure_model_file: Path,
    model_file: Path,
    candidate_factory: Callable[[Path, ArtifactFormat], ModelCandidateT],
    load_secure_model_for_settings: Callable[
        [ModelCandidateT, SettingsT], LoadedModelT
    ],
    load_pickle_model: Callable[[ModelCandidateT], LoadedModelT],
    logger: logging.Logger,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> tuple[ModelCandidateT, LoadedModelT]:
    """Resolves one compatible model artifact using settings-derived paths."""
    candidates = model_load_candidates(
        folder=folder,
        secure_model_file=secure_model_file,
        model_file=model_file,
        candidate_factory=candidate_factory,
    )

    def load_secure_model(candidate: ModelCandidateT) -> LoadedModelT:
        return load_secure_model_for_settings(candidate, settings)

    return resolve_model_for_loading(
        candidates=candidates,
        load_secure_model=load_secure_model,
        load_pickle_model=load_pickle_model,
        logger=logger,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
    )


def artifact_matches_expected_profile(
    artifact_metadata: object,
    *,
    expected_backend_id: str | None,
    expected_profile: str | None,
    expected_backend_model_id: str | None,
) -> bool:
    """Checks whether artifact metadata matches expected backend/profile values."""
    if (
        expected_backend_id is None
        and expected_profile is None
        and expected_backend_model_id is None
    ):
        return True
    if not isinstance(artifact_metadata, dict):
        return False
    if (
        expected_backend_id is not None
        and artifact_metadata.get("backend_id") != expected_backend_id
    ):
        return False
    if (
        expected_profile is not None
        and artifact_metadata.get("profile") != expected_profile
    ):
        return False
    if expected_backend_model_id is not None:
        backend_model_id = artifact_metadata.get("backend_model_id")
        if (
            not isinstance(backend_model_id, str)
            or backend_model_id.strip() != expected_backend_model_id
        ):
            return False
    return True


def resolve_model_for_loading(
    *,
    candidates: tuple[ModelCandidateT, ...],
    load_secure_model: Callable[[ModelCandidateT], LoadedModelT],
    load_pickle_model: Callable[[ModelCandidateT], LoadedModelT],
    logger: logging.Logger,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
    compatibility_check: Callable[[LoadedModelT], bool] | None = None,
) -> tuple[ModelCandidateT, LoadedModelT]:
    """Finds and loads the first valid model artifact candidate."""
    existing_candidates: list[ModelCandidateT] = [
        candidate for candidate in candidates if candidate.path.exists()
    ]
    if not existing_candidates:
        candidate_list = ", ".join(str(candidate.path) for candidate in candidates)
        raise FileNotFoundError(
            "Model not found. Checked: "
            f"{candidate_list}. Train it first with `ser --train`."
        )

    last_error: Exception | None = None
    rejected_candidates: list[str] = []
    for candidate in existing_candidates:
        try:
            loaded_model: LoadedModelT = (
                load_secure_model(candidate)
                if candidate.artifact_format == "skops"
                else load_pickle_model(candidate)
            )
            if compatibility_check is not None:
                if compatibility_check(loaded_model):
                    return candidate, loaded_model
            elif artifact_matches_expected_profile(
                loaded_model.artifact_metadata,
                expected_backend_id=expected_backend_id,
                expected_profile=expected_profile,
                expected_backend_model_id=expected_backend_model_id,
            ):
                return candidate, loaded_model
            rejected_candidates.append(str(candidate.path))
        except Exception as err:
            last_error = err
            logger.debug(
                "Failed to load %s model artifact at %s: %s",
                candidate.artifact_format,
                candidate.path,
                err,
            )

    if rejected_candidates:
        expected_constraints: list[str] = []
        if expected_backend_id is not None:
            expected_constraints.append(f"backend_id={expected_backend_id!r}")
        if expected_profile is not None:
            expected_constraints.append(f"profile={expected_profile!r}")
        if expected_backend_model_id is not None:
            expected_constraints.append(
                f"backend_model_id={expected_backend_model_id!r}"
            )
        constraint_text = ", ".join(expected_constraints)
        checked = ", ".join(rejected_candidates)
        raise FileNotFoundError(
            "No compatible model artifact is available for "
            f"{constraint_text}. Checked: {checked}. "
            "Train/select a matching artifact and retry."
        )

    candidate_list = ", ".join(str(candidate.path) for candidate in existing_candidates)
    raise ValueError(
        f"Failed to deserialize model from any candidate path: {candidate_list}."
    ) from last_error


def load_model_with_resolution(
    *,
    settings: SettingsT | None,
    settings_resolver: Callable[[], SettingsT],
    resolve_model: ResolveModelForLoading[ModelCandidateT, LoadedModelT],
    logger: logging.Logger,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> LoadedModelT:
    """Loads one model with compatibility filters and stable error semantics."""
    try:
        active_settings = settings if settings is not None else settings_resolver()
        candidate, loaded_model = resolve_model(
            active_settings,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=expected_backend_model_id,
        )
        logger.info(
            "Model loaded from %s (%s).",
            candidate.path,
            candidate.artifact_format,
        )
        return loaded_model
    except FileNotFoundError:
        raise
    except Exception as err:
        logger.error("Failed to load model: %s", err)
        raise ValueError("Failed to load model from configured locations.") from err
