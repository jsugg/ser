"""Training and inference helpers for the SER emotion classification model."""

from __future__ import annotations

import glob
import importlib
import json
import logging
import os
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, NamedTuple, cast

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ser.config import AppConfig, ProfileRuntimeConfig, get_settings
from ser.data import (
    EmbeddingCache,
    LabeledAudioSample,
    Utterance,
    load_data,
    load_utterances,
)
from ser.data.data_loader import extract_ravdess_speaker_id_from_path
from ser.data.embedding_cache import EmbeddingCacheEntry
from ser.domain import EmotionSegment
from ser.features import extract_feature_frames
from ser.license_check import (
    build_provenance_metadata,
    ensure_backend_access,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
)
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import (
    Emotion2VecBackend,
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    WhisperBackend,
    XLSRBackend,
)
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.runtime.schema import (
    ARTIFACT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
    to_legacy_emotion_segments,
)
from ser.train.eval import grouped_train_test_split
from ser.train.metrics import compute_grouped_ser_metrics_by_sample, compute_ser_metrics
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

MODEL_ARTIFACT_VERSION = 2
DEFAULT_BACKEND_ID = "handcrafted"
DEFAULT_PROFILE_ID = "fast"
DEFAULT_FRAME_SIZE_SECONDS = 3.0
DEFAULT_FRAME_STRIDE_SECONDS = 1.0
DEFAULT_POOLING_STRATEGY = "mean"
MEDIUM_BACKEND_ID = "hf_xlsr"
MEDIUM_PROFILE_ID = "medium"
MEDIUM_MODEL_ID = "facebook/wav2vec2-xls-r-300m"
MEDIUM_FRAME_SIZE_SECONDS = 1.0
MEDIUM_FRAME_STRIDE_SECONDS = 1.0
MEDIUM_POOLING_STRATEGY = "mean_std"
ACCURATE_BACKEND_ID = "hf_whisper"
ACCURATE_PROFILE_ID = "accurate"
ACCURATE_MODEL_ID = "openai/whisper-large-v3"
ACCURATE_POOLING_STRATEGY = "mean_std"
ACCURATE_RESEARCH_BACKEND_ID = "emotion2vec"
ACCURATE_RESEARCH_PROFILE_ID = "accurate-research"
ACCURATE_RESEARCH_MODEL_ID = "iic/emotion2vec_plus_large"
type EmotionClassifier = MLPClassifier | Pipeline
type ArtifactFormat = Literal["pickle", "skops"]


def resolve_medium_model_id(settings: AppConfig | None = None) -> str:
    """Resolves the medium XLS-R model id from settings with safe fallback."""
    active_settings: AppConfig = settings if settings is not None else get_settings()
    configured_model_id = getattr(
        getattr(active_settings, "models", None),
        "medium_model_id",
        MEDIUM_MODEL_ID,
    )
    if not isinstance(configured_model_id, str) or not configured_model_id.strip():
        return MEDIUM_MODEL_ID
    return configured_model_id.strip()


def resolve_accurate_model_id(settings: AppConfig | None = None) -> str:
    """Resolves the accurate Whisper model id from settings with safe fallback."""
    active_settings: AppConfig = settings if settings is not None else get_settings()
    configured_model_id = getattr(
        getattr(active_settings, "models", None),
        "accurate_model_id",
        ACCURATE_MODEL_ID,
    )
    if not isinstance(configured_model_id, str) or not configured_model_id.strip():
        return ACCURATE_MODEL_ID
    return configured_model_id.strip()


def resolve_accurate_research_model_id(settings: AppConfig | None = None) -> str:
    """Resolves the accurate-research model id from settings with safe fallback."""
    active_settings: AppConfig = settings if settings is not None else get_settings()
    configured_model_id = getattr(
        getattr(active_settings, "models", None),
        "accurate_research_model_id",
        ACCURATE_RESEARCH_MODEL_ID,
    )
    if not isinstance(configured_model_id, str) or not configured_model_id.strip():
        return ACCURATE_RESEARCH_MODEL_ID
    return configured_model_id.strip()


def _resolve_feature_runtime_selectors(
    *,
    backend_id: str,
    settings: AppConfig,
) -> tuple[str, str]:
    """Resolves backend-aware runtime selectors for feature extraction."""
    backend_override_device: str | None = None
    backend_override_dtype: str | None = None
    feature_runtime_policy = getattr(settings, "feature_runtime_policy", None)
    resolve_backend_override = (
        getattr(feature_runtime_policy, "for_backend", None)
        if feature_runtime_policy is not None
        else None
    )
    if callable(resolve_backend_override):
        backend_override = resolve_backend_override(backend_id)
        if backend_override is not None:
            backend_override_device = getattr(backend_override, "device", None)
            backend_override_dtype = getattr(backend_override, "dtype", None)
    runtime_policy = resolve_feature_runtime_policy(
        backend_id=backend_id,
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=backend_override_device,
        backend_override_dtype=backend_override_dtype,
    )
    if (
        runtime_policy.device != settings.torch_runtime.device
        or runtime_policy.dtype != settings.torch_runtime.dtype
    ):
        logger.info(
            "Feature runtime policy adjusted selectors for backend=%s "
            "(device=%s, dtype=%s, reason=%s).",
            backend_id,
            runtime_policy.device,
            runtime_policy.dtype,
            runtime_policy.reason,
        )
    return runtime_policy.device, runtime_policy.dtype


class LoadedModel(NamedTuple):
    """Loaded model object and optional expected feature-vector length."""

    model: EmotionClassifier
    expected_feature_size: int | None
    artifact_metadata: dict[str, object] | None = None


class ModelCandidate(NamedTuple):
    """A candidate model artifact path and serialization format."""

    path: Path
    artifact_format: ArtifactFormat


@dataclass(frozen=True)
class PersistedArtifacts:
    """Paths to persisted model artifacts from training."""

    pickle_path: Path
    secure_path: Path | None


@dataclass(frozen=True)
class MediumNoiseControlStats:
    """Window-level filtering statistics for medium training traceability."""

    total_windows: int = 0
    kept_windows: int = 0
    dropped_low_std_windows: int = 0
    dropped_cap_windows: int = 0
    forced_keep_windows: int = 0


@dataclass(frozen=True)
class MediumSplitMetadata:
    """Split diagnostics persisted for grouped-evaluation traceability."""

    split_strategy: str
    speaker_grouped: bool
    speaker_id_coverage: float
    train_unique_speakers: int
    test_unique_speakers: int
    speaker_overlap_count: int


@dataclass(frozen=True)
class WindowMeta:
    """Window-level metadata for evaluation breakdowns."""

    sample_id: str
    corpus: str
    language: str


def _read_positive_int(metadata: dict[str, object], field_name: str) -> int:
    """Returns a required positive integer field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, int) or raw_value <= 0:
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return raw_value


def _read_positive_float(metadata: dict[str, object], field_name: str) -> float:
    """Returns a required positive float field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, int | float) or float(raw_value) <= 0.0:
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return float(raw_value)


def _read_non_empty_text(metadata: dict[str, object], field_name: str) -> str:
    """Returns a required non-empty text field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return raw_value


def _read_labels(metadata: dict[str, object]) -> list[str]:
    """Returns validated class labels from metadata."""
    raw_labels = metadata.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("Model artifact metadata contains invalid 'labels' value.")
    labels: list[str] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("Model artifact metadata contains invalid 'labels' value.")
        labels.append(raw_label)
    if not labels:
        raise ValueError("Model artifact metadata contains invalid 'labels' value.")
    return sorted(set(labels))


def _normalize_provenance_metadata(raw_value: object) -> dict[str, object]:
    """Validates optional provenance metadata, defaulting to an empty payload."""
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("Model artifact metadata contains invalid 'provenance' value.")

    allowed_text_fields = (
        "code_revision",
        "dependency_manifest_fingerprint",
        "backend_id",
        "backend_license_id",
        "profile",
        "dataset_glob_pattern",
        "backend_access_source",
        "restricted_backend_consent_source",
        "restricted_backend_consent_accepted_at_utc",
        "restricted_backend_policy_fingerprint",
        "license_source_url",
    )
    normalized: dict[str, object] = {}
    for field_name in allowed_text_fields:
        field_value = raw_value.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                "Model artifact metadata contains invalid provenance field "
                f"{field_name!r}."
            )
        normalized[field_name] = field_value

    allowed_bool_fields = (
        "runtime_restricted_backends_enabled",
        "backend_is_restricted",
        "backend_access_allowed",
    )
    for field_name in allowed_bool_fields:
        field_value = raw_value.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, bool):
            raise ValueError(
                "Model artifact metadata contains invalid provenance field "
                f"{field_name!r}."
            )
        normalized[field_name] = field_value
    return normalized


def _normalize_v2_artifact_metadata(metadata: dict[str, object]) -> dict[str, object]:
    """Validates and normalizes artifact metadata to v2 shape."""
    artifact_version = _read_positive_int(metadata, "artifact_version")
    if artifact_version != MODEL_ARTIFACT_VERSION:
        raise ValueError(
            "Model artifact metadata contains unsupported 'artifact_version' value."
        )
    feature_vector_size = _read_positive_int(metadata, "feature_vector_size")
    feature_dim = _read_positive_int(metadata, "feature_dim")
    if feature_dim != feature_vector_size:
        raise ValueError(
            "Model artifact metadata 'feature_dim' must match 'feature_vector_size'."
        )
    normalized: dict[str, object] = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": _read_non_empty_text(
            metadata, "artifact_schema_version"
        ),
        "created_at_utc": _read_non_empty_text(metadata, "created_at_utc"),
        "feature_vector_size": feature_vector_size,
        "training_samples": _read_positive_int(metadata, "training_samples"),
        "labels": _read_labels(metadata),
        "backend_id": _read_non_empty_text(metadata, "backend_id"),
        "profile": _read_non_empty_text(metadata, "profile"),
        "feature_dim": feature_dim,
        "frame_size_seconds": _read_positive_float(metadata, "frame_size_seconds"),
        "frame_stride_seconds": _read_positive_float(metadata, "frame_stride_seconds"),
        "pooling_strategy": _read_non_empty_text(metadata, "pooling_strategy"),
        "provenance": _normalize_provenance_metadata(metadata.get("provenance")),
    }
    backend_model_id = metadata.get("backend_model_id")
    if backend_model_id is not None:
        if not isinstance(backend_model_id, str) or not backend_model_id.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'backend_model_id' value."
            )
        normalized["backend_model_id"] = backend_model_id.strip()
    torch_device = metadata.get("torch_device")
    if torch_device is not None:
        if not isinstance(torch_device, str) or not torch_device.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_device' value."
            )
        normalized["torch_device"] = torch_device.strip().lower()
    torch_dtype = metadata.get("torch_dtype")
    if torch_dtype is not None:
        if not isinstance(torch_dtype, str) or not torch_dtype.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_dtype' value."
            )
        normalized["torch_dtype"] = torch_dtype.strip().lower()
    return normalized


def _build_v2_artifact_metadata(
    *,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
    backend_id: str = DEFAULT_BACKEND_ID,
    profile: str = DEFAULT_PROFILE_ID,
    feature_dim: int | None = None,
    frame_size_seconds: float = DEFAULT_FRAME_SIZE_SECONDS,
    frame_stride_seconds: float = DEFAULT_FRAME_STRIDE_SECONDS,
    pooling_strategy: str = DEFAULT_POOLING_STRATEGY,
    backend_model_id: str | None = None,
    torch_device: str | None = None,
    torch_dtype: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Builds normalized v2 artifact metadata for persisted model envelopes."""
    resolved_feature_dim = feature_vector_size if feature_dim is None else feature_dim
    payload: dict[str, object] = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "feature_vector_size": feature_vector_size,
        "training_samples": training_samples,
        "labels": labels,
        "backend_id": backend_id,
        "profile": profile,
        "feature_dim": resolved_feature_dim,
        "frame_size_seconds": frame_size_seconds,
        "frame_stride_seconds": frame_stride_seconds,
        "pooling_strategy": pooling_strategy,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    if backend_model_id is not None:
        resolved_backend_model_id = backend_model_id.strip()
        if not resolved_backend_model_id:
            raise ValueError(
                "Model artifact metadata contains invalid 'backend_model_id' value."
            )
        payload["backend_model_id"] = resolved_backend_model_id
    if torch_device is not None:
        resolved_torch_device = torch_device.strip().lower()
        if not resolved_torch_device:
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_device' value."
            )
        payload["torch_device"] = resolved_torch_device
    if torch_dtype is not None:
        resolved_torch_dtype = torch_dtype.strip().lower()
        if not resolved_torch_dtype:
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_dtype' value."
            )
        payload["torch_dtype"] = resolved_torch_dtype
    return _normalize_v2_artifact_metadata(payload)


def _create_classifier() -> EmotionClassifier:
    """Builds a reproducible scaler+MLP training pipeline."""
    settings = get_settings()
    classifier: MLPClassifier = MLPClassifier(
        alpha=settings.nn.alpha,
        batch_size=settings.nn.batch_size,  # pyright: ignore[reportArgumentType]
        epsilon=settings.nn.epsilon,
        hidden_layer_sizes=settings.nn.hidden_layer_sizes,
        learning_rate=settings.nn.learning_rate,
        max_iter=settings.nn.max_iter,
        random_state=settings.nn.random_state,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def _build_model_artifact(
    model: EmotionClassifier,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
    backend_id: str = DEFAULT_BACKEND_ID,
    profile: str = DEFAULT_PROFILE_ID,
    feature_dim: int | None = None,
    frame_size_seconds: float = DEFAULT_FRAME_SIZE_SECONDS,
    frame_stride_seconds: float = DEFAULT_FRAME_STRIDE_SECONDS,
    pooling_strategy: str = DEFAULT_POOLING_STRATEGY,
    backend_model_id: str | None = None,
    torch_device: str | None = None,
    torch_dtype: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Constructs a versioned model artifact envelope for safer loading."""
    metadata = _build_v2_artifact_metadata(
        feature_vector_size=feature_vector_size,
        training_samples=training_samples,
        labels=labels,
        backend_id=backend_id,
        profile=profile,
        feature_dim=feature_dim,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        pooling_strategy=pooling_strategy,
        backend_model_id=backend_model_id,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        provenance=provenance,
    )
    return {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model": model,
        "metadata": metadata,
    }


def _deserialize_model_artifact(payload: object) -> LoadedModel:
    """Validates and unwraps persisted model payloads."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Model artifact payload must be a versioned dictionary envelope "
            f"(received {type(payload).__name__})."
        )

    artifact_version = payload.get("artifact_version")
    if artifact_version != MODEL_ARTIFACT_VERSION:
        raise ValueError(
            "Unsupported model artifact version "
            f"{artifact_version!r}; expected {MODEL_ARTIFACT_VERSION}. "
            "Regenerate artifacts with current training code."
        )

    model = payload.get("model")
    if not isinstance(model, MLPClassifier | Pipeline):
        raise ValueError(
            "Unexpected model object type in artifact envelope: "
            f"{type(model).__name__}."
        )

    metadata_obj = payload.get("metadata")
    if not isinstance(metadata_obj, dict):
        raise ValueError("Model artifact metadata is missing or invalid.")
    normalized_metadata = _normalize_v2_artifact_metadata(metadata_obj)
    expected_feature_size = _read_positive_int(
        normalized_metadata,
        "feature_vector_size",
    )

    return LoadedModel(
        model=model,
        expected_feature_size=expected_feature_size,
        artifact_metadata=normalized_metadata,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Writes binary payload atomically to avoid partial artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, payload: str) -> None:
    """Writes UTF-8 text atomically to avoid truncated JSON output."""
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _persist_pickle_artifact(path: Path, artifact: dict[str, object]) -> None:
    """Serializes and stores the pickle envelope for broad compatibility."""
    serialized = pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(path, serialized)


def _persist_secure_artifact(path: Path, model: EmotionClassifier) -> bool:
    """Attempts to persist a secure model format via `skops`, if available."""
    try:
        skops_io: Any = importlib.import_module("skops.io")
    except ModuleNotFoundError:
        logger.info(
            "Optional dependency `skops` is not installed; skipping secure model "
            "artifact generation."
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        skops_io.dump(model, str(tmp_path))
        os.replace(tmp_path, path)
        return True
    except Exception as err:
        logger.warning(
            "Failed to persist secure model artifact at %s: %s. Continuing with "
            "pickle artifact.",
            path,
            err,
        )
        return False
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _persist_training_report(report: dict[str, object], path: Path) -> None:
    """Persists a deterministic JSON training report to disk."""
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(path, serialized)


def _read_training_report_feature_size(path: Path) -> int | None:
    """Reads expected feature size from the training report when available."""
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as report_fh:
            payload = json.load(report_fh)
    except Exception as err:
        logger.warning("Failed to parse training report at %s: %s", path, err)
        return None

    if not isinstance(payload, dict):
        return None

    feature_size = payload.get("feature_vector_size")
    if isinstance(feature_size, int) and feature_size > 0:
        return feature_size

    artifact_metadata = payload.get("artifact_metadata")
    if isinstance(artifact_metadata, dict):
        nested_feature_size = artifact_metadata.get("feature_vector_size")
        if isinstance(nested_feature_size, int) and nested_feature_size > 0:
            return nested_feature_size
    return None


def _build_training_report(
    *,
    accuracy: float,
    macro_f1: float,
    ser_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    feature_vector_size: int,
    labels: list[str],
    artifacts: PersistedArtifacts,
    artifact_metadata: dict[str, object],
    data_controls: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Builds a structured report for training quality and artifact traceability."""
    settings = get_settings()
    corpus_samples = len(glob.glob(settings.dataset.glob_pattern))
    effective_samples = train_samples + test_samples
    label_distribution = dict(Counter(labels))
    model_artifacts: dict[str, str] = {"pickle": str(artifacts.pickle_path)}
    if artifacts.secure_path is not None:
        model_artifacts["secure"] = str(artifacts.secure_path)

    report: dict[str, object] = {
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_glob_pattern": settings.dataset.glob_pattern,
        "dataset_corpus_samples": corpus_samples,
        "dataset_effective_samples": effective_samples,
        "dataset_skipped_samples": max(0, corpus_samples - effective_samples),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "feature_vector_size": feature_vector_size,
        "labels": sorted(set(labels)),
        "label_distribution": label_distribution,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metrics": ser_metrics,
        "artifact_metadata": artifact_metadata,
        "model_artifacts": model_artifacts,
    }
    if data_controls is not None:
        report["data_controls"] = data_controls
    if provenance is not None:
        report["provenance"] = provenance
    return report


def _persist_model_artifacts(
    model: EmotionClassifier, artifact: dict[str, object]
) -> PersistedArtifacts:
    """Persists model artifacts in compatibility-first and secure formats."""
    settings = get_settings()
    pickle_path = settings.models.model_file
    secure_path = settings.models.secure_model_file

    _persist_pickle_artifact(pickle_path, artifact)
    secure_saved = _persist_secure_artifact(secure_path, model)
    return PersistedArtifacts(
        pickle_path=pickle_path,
        secure_path=secure_path if secure_saved else None,
    )


def _discover_model_candidates(
    folder: Path,
) -> list[ModelCandidate]:
    """Discovers model artifacts in a folder using SER naming conventions."""
    if not folder.exists():
        return []

    discovered: list[ModelCandidate] = []
    for pattern, artifact_format in (
        ("ser_model*.skops", "skops"),
        ("ser_model*.pkl", "pickle"),
    ):
        for path in sorted(folder.glob(pattern)):
            if path.is_file():
                discovered.append(
                    ModelCandidate(
                        path=path,
                        artifact_format=cast(ArtifactFormat, artifact_format),
                    )
                )
    return discovered


def _model_load_candidates(
    settings: AppConfig | None = None,
) -> tuple[ModelCandidate, ...]:
    """Returns model artifacts in preferred load order from primary storage."""
    active_settings = settings if settings is not None else get_settings()
    primary_secure = active_settings.models.secure_model_file
    primary_pickle = active_settings.models.model_file
    discovered_primary = _discover_model_candidates(active_settings.models.folder)

    ordered = (
        ModelCandidate(primary_secure, "skops"),
        ModelCandidate(primary_pickle, "pickle"),
        *discovered_primary,
    )

    deduped: list[ModelCandidate] = []
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


def _load_secure_model(candidate: ModelCandidate, settings: AppConfig) -> LoadedModel:
    """Loads a secure artifact when `skops` is available and trusted."""
    assert candidate.artifact_format == "skops"
    try:
        skops_io: Any = importlib.import_module("skops.io")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Secure model artifact found but `skops` is not installed."
        ) from err

    untrusted_types = set(skops_io.get_untrusted_types(file=str(candidate.path)))
    if untrusted_types:
        raise ValueError(
            "Secure model artifact contains untrusted types; refusing automatic "
            f"trust for {candidate.path}."
        )

    payload = skops_io.load(str(candidate.path), trusted=[])
    if not isinstance(payload, MLPClassifier | Pipeline):
        raise ValueError(
            "Unexpected secure model payload type: "
            f"{type(payload).__name__}. Expected sklearn classifier/pipeline."
        )

    feature_size: int | None = _read_training_report_feature_size(
        settings.models.training_report_file
    )
    return LoadedModel(model=payload, expected_feature_size=feature_size)


def _load_pickle_model(candidate: ModelCandidate) -> LoadedModel:
    """Loads and validates the compatibility pickle model artifact."""
    assert candidate.artifact_format == "pickle"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with candidate.path.open("rb") as model_fh:
            payload = pickle.load(model_fh)
    return _deserialize_model_artifact(payload)


def _artifact_matches_expected_profile(
    loaded_model: LoadedModel,
    *,
    expected_backend_id: str | None,
    expected_profile: str | None,
    expected_backend_model_id: str | None,
) -> bool:
    """Checks whether loaded artifact metadata matches expected backend/profile."""
    if (
        expected_backend_id is None
        and expected_profile is None
        and expected_backend_model_id is None
    ):
        return True

    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        return False
    if (
        expected_backend_id is not None
        and metadata.get("backend_id") != expected_backend_id
    ):
        return False
    if expected_profile is not None and metadata.get("profile") != expected_profile:
        return False
    if expected_backend_model_id is not None:
        backend_model_id = metadata.get("backend_model_id")
        if (
            not isinstance(backend_model_id, str)
            or backend_model_id.strip() != expected_backend_model_id
        ):
            return False
    return True


def _resolve_model_for_loading(
    settings: AppConfig | None = None,
    *,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> tuple[ModelCandidate, LoadedModel]:
    """Finds and loads the first valid model artifact candidate."""
    active_settings = settings if settings is not None else get_settings()
    candidates: tuple[ModelCandidate, ...] = _model_load_candidates(active_settings)
    existing_candidates: list[ModelCandidate] = [
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
            loaded_model: LoadedModel = (
                _load_secure_model(candidate, active_settings)
                if candidate.artifact_format == "skops"
                else _load_pickle_model(candidate)
            )
            if _artifact_matches_expected_profile(
                loaded_model,
                expected_backend_id=expected_backend_id,
                expected_profile=expected_profile,
                expected_backend_model_id=expected_backend_model_id,
            ):
                return candidate, loaded_model
            rejected_candidates.append(str(candidate.path))
            # logger.info(
            #     "Skipping model artifact at %s: incompatible metadata for "
            #     "expected backend/profile/model-id (%s, %s, %s).",
            #     candidate.path,
            #     expected_backend_id,
            #     expected_profile,
            #     expected_backend_model_id,
            # )
        except Exception as err:
            last_error = err
            logger.warning(
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


def _split_labeled_audio_samples(
    samples: list[LabeledAudioSample],
) -> tuple[list[LabeledAudioSample], list[LabeledAudioSample], MediumSplitMetadata]:
    """Splits labeled files with grouped-speaker preference and traceable metadata."""
    settings: AppConfig = get_settings()
    if len(samples) < 2:
        raise RuntimeError("Medium training requires at least two labeled audio files.")

    indices: np.ndarray = np.arange(len(samples), dtype=np.int64)
    labels: list[str] = [label for _, label in samples]
    raw_speaker_ids: list[str | None] = [
        extract_ravdess_speaker_id_from_path(audio_path) for audio_path, _ in samples
    ]
    resolved_speaker_ids = [item for item in raw_speaker_ids if item is not None]
    speaker_coverage = float(len(resolved_speaker_ids)) / float(len(samples))

    split_strategy = "stratified_shuffle_split"
    train_idx = np.empty(0, dtype=np.int64)
    test_idx = np.empty(0, dtype=np.int64)
    can_group_by_speaker = (
        len(resolved_speaker_ids) == len(samples)
        and len(set(resolved_speaker_ids)) >= 2
    )
    if can_group_by_speaker:
        grouped_features = np.zeros((len(samples), 1), dtype=np.float64)
        try:
            grouped_split = grouped_train_test_split(
                grouped_features,
                labels,
                [str(item) for item in resolved_speaker_ids],
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
            )
            train_idx = grouped_split.train_indices
            test_idx = grouped_split.test_indices
            split_strategy = "group_shuffle_split"
        except ValueError as err:
            logger.warning(
                "Medium grouped split failed (%s); falling back to stratified split.",
                err,
            )
            can_group_by_speaker = False

    if not can_group_by_speaker:
        split_strategy = "stratified_shuffle_split_fallback"
        stratify_labels: list[str] | None = (
            labels if settings.training.stratify_split else None
        )
        try:
            train_idx_raw, test_idx_raw = train_test_split(
                indices,
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
                stratify=stratify_labels,
            )
            train_idx = np.asarray(train_idx_raw, dtype=np.int64)
            test_idx = np.asarray(test_idx_raw, dtype=np.int64)
        except ValueError as err:
            logger.warning(
                "Medium stratified split failed (%s); falling back to non-stratified split.",
                err,
            )
            train_idx_raw, test_idx_raw = train_test_split(
                indices,
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
                stratify=None,
            )
            train_idx = np.asarray(train_idx_raw, dtype=np.int64)
            test_idx = np.asarray(test_idx_raw, dtype=np.int64)

    train_samples: list[LabeledAudioSample] = [
        samples[int(index)] for index in train_idx
    ]
    test_samples: list[LabeledAudioSample] = [samples[int(index)] for index in test_idx]
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            "Medium training split failed to produce deterministic index partitions."
        )
    if not train_samples or not test_samples:
        raise RuntimeError(
            "Medium training split produced an empty partition; adjust test_size."
        )

    train_speakers = {
        raw_speaker_ids[int(index)]
        for index in train_idx.tolist()
        if raw_speaker_ids[int(index)] is not None
    }
    test_speakers = {
        raw_speaker_ids[int(index)]
        for index in test_idx.tolist()
        if raw_speaker_ids[int(index)] is not None
    }
    speaker_overlap_count = len(train_speakers.intersection(test_speakers))
    if split_strategy == "group_shuffle_split" and speaker_overlap_count > 0:
        raise RuntimeError(
            "Grouped medium split produced overlapping speakers in train/test."
        )

    return (
        train_samples,
        test_samples,
        MediumSplitMetadata(
            split_strategy=split_strategy,
            speaker_grouped=split_strategy == "group_shuffle_split",
            speaker_id_coverage=speaker_coverage,
            train_unique_speakers=len(train_speakers),
            test_unique_speakers=len(test_speakers),
            speaker_overlap_count=speaker_overlap_count,
        ),
    )


def _resolve_corpus_scoped_speaker_id(utterance: Utterance) -> str | None:
    """Returns speaker id with fallback extraction for known RAVDESS layouts."""
    if utterance.speaker_id is not None:
        return utterance.speaker_id
    if utterance.corpus != "ravdess":
        return None
    speaker_raw = extract_ravdess_speaker_id_from_path(str(utterance.audio_path))
    if speaker_raw is None:
        return None
    return f"{utterance.corpus}:{speaker_raw}"


def _hash_for_split(sample_id: str, *, salt: str) -> int:
    digest = sha1(f"{salt}|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _hash_stratified_split(
    samples: list[Utterance],
    *,
    test_size: float,
    salt: str,
) -> tuple[list[Utterance], list[Utterance]]:
    by_label: dict[str, list[Utterance]] = {}
    for utterance in samples:
        by_label.setdefault(utterance.label, []).append(utterance)

    train: list[Utterance] = []
    test: list[Utterance] = []
    for _, group in sorted(by_label.items(), key=lambda item: item[0]):
        group_sorted = sorted(
            group,
            key=lambda utterance: _hash_for_split(utterance.sample_id, salt=salt),
        )
        if len(group_sorted) < 2:
            train.extend(group_sorted)
            continue
        n_test = int(round(test_size * len(group_sorted)))
        if n_test <= 0:
            n_test = 1
        if n_test >= len(group_sorted):
            n_test = len(group_sorted) - 1
        test.extend(group_sorted[:n_test])
        train.extend(group_sorted[n_test:])

    if not test and train:
        train_sorted = sorted(
            train,
            key=lambda utterance: _hash_for_split(utterance.sample_id, salt=salt),
        )
        test.append(train_sorted.pop(0))
        train = train_sorted
    if not train and test:
        test_sorted = sorted(
            test,
            key=lambda utterance: _hash_for_split(utterance.sample_id, salt=salt),
        )
        train.append(test_sorted.pop(0))
        test = test_sorted
    return train, test


def _split_utterances(
    samples: list[Utterance],
) -> tuple[list[Utterance], list[Utterance], MediumSplitMetadata]:
    """Splits utterances deterministically with manifest/speaker/hash policy."""
    settings: AppConfig = get_settings()
    if len(samples) < 2:
        raise RuntimeError("Training requires at least two labeled audio files.")

    labels: list[str] = [utterance.label for utterance in samples]
    speaker_ids: list[str | None] = [
        _resolve_corpus_scoped_speaker_id(utterance) for utterance in samples
    ]
    resolved_speaker_ids = [item for item in speaker_ids if item is not None]
    speaker_coverage = float(len(resolved_speaker_ids)) / float(len(samples))

    has_manifest_split = all(utterance.split is not None for utterance in samples)
    if has_manifest_split:
        train_split = [
            utterance for utterance in samples if utterance.split in {"train", "dev"}
        ]
        test_split = [utterance for utterance in samples if utterance.split == "test"]
        if train_split and test_split:
            train_speakers = {
                speaker
                for utterance, speaker in zip(samples, speaker_ids, strict=False)
                if utterance in train_split and speaker is not None
            }
            test_speakers = {
                speaker
                for utterance, speaker in zip(samples, speaker_ids, strict=False)
                if utterance in test_split and speaker is not None
            }
            return (
                train_split,
                test_split,
                MediumSplitMetadata(
                    split_strategy="manifest_split",
                    speaker_grouped=False,
                    speaker_id_coverage=speaker_coverage,
                    train_unique_speakers=len(train_speakers),
                    test_unique_speakers=len(test_speakers),
                    speaker_overlap_count=len(
                        train_speakers.intersection(test_speakers)
                    ),
                ),
            )

    can_group_by_speaker = (
        len(resolved_speaker_ids) == len(samples)
        and len(set(resolved_speaker_ids)) >= 2
    )
    if can_group_by_speaker:
        grouped_features = np.zeros((len(samples), 1), dtype=np.float64)
        try:
            grouped_split = grouped_train_test_split(
                grouped_features,
                labels,
                [str(item) for item in resolved_speaker_ids],
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
            )
            train_idx = grouped_split.train_indices
            test_idx = grouped_split.test_indices
            train_split = [samples[int(index)] for index in train_idx]
            test_split = [samples[int(index)] for index in test_idx]
            train_speakers = {
                speaker
                for index in train_idx.tolist()
                if (speaker := speaker_ids[int(index)]) is not None
            }
            test_speakers = {
                speaker
                for index in test_idx.tolist()
                if (speaker := speaker_ids[int(index)]) is not None
            }
            overlap = len(train_speakers.intersection(test_speakers))
            if overlap > 0:
                raise RuntimeError(
                    "Grouped split produced overlapping speakers in train/test."
                )
            return (
                train_split,
                test_split,
                MediumSplitMetadata(
                    split_strategy="group_shuffle_split",
                    speaker_grouped=True,
                    speaker_id_coverage=speaker_coverage,
                    train_unique_speakers=len(train_speakers),
                    test_unique_speakers=len(test_speakers),
                    speaker_overlap_count=overlap,
                ),
            )
        except ValueError as err:
            logger.warning(
                "Grouped split failed (%s); falling back to deterministic hash split.",
                err,
            )

    salt = os.getenv("SER_SPLIT_SALT", f"ser:{settings.training.random_state}").strip()
    train_split, test_split = _hash_stratified_split(
        samples,
        test_size=settings.training.test_size,
        salt=salt,
    )
    if not train_split or not test_split:
        raise RuntimeError(
            "Deterministic split produced an empty partition; adjust test_size."
        )
    train_speakers = {
        speaker
        for utterance, speaker in zip(samples, speaker_ids, strict=False)
        if utterance in train_split and speaker is not None
    }
    test_speakers = {
        speaker
        for utterance, speaker in zip(samples, speaker_ids, strict=False)
        if utterance in test_split and speaker is not None
    }
    return (
        train_split,
        test_split,
        MediumSplitMetadata(
            split_strategy="hash_stratified_split",
            speaker_grouped=False,
            speaker_id_coverage=speaker_coverage,
            train_unique_speakers=len(train_speakers),
            test_unique_speakers=len(test_speakers),
            speaker_overlap_count=len(train_speakers.intersection(test_speakers)),
        ),
    )


def _utterances_to_labeled_samples(
    utterances: list[Utterance],
) -> list[LabeledAudioSample]:
    return [(str(utterance.audio_path), utterance.label) for utterance in utterances]


def _build_dataset_controls(utterances: list[Utterance]) -> dict[str, object]:
    settings = get_settings()
    corpus_counts = dict(Counter(utterance.corpus for utterance in utterances))
    language_counts = dict(
        Counter((utterance.language or "unknown") for utterance in utterances)
    )

    manifest_paths: list[str] = [str(path) for path in settings.dataset.manifest_paths]
    mode = "manifest" if manifest_paths else "glob"
    if not manifest_paths:
        try:
            from ser.data.dataset_registry import (
                load_dataset_registry,
                registered_manifest_paths,
            )

            registry = load_dataset_registry(settings=settings)
            if registry:
                mode = "registry"
                manifest_paths = [
                    str(path)
                    for path in sorted(
                        set(registered_manifest_paths(settings=settings))
                    )
                ]
        except Exception:
            # Optional feature; ignore and keep glob mode.
            pass
    controls: dict[str, object] = {
        "mode": mode,
        "manifest_paths": manifest_paths,
        "utterance_count": len(utterances),
        "corpus_counts": corpus_counts,
        "language_counts": language_counts,
    }
    return controls


def _ensure_dataset_consents_for_training(*, utterances: list[Utterance]) -> None:
    """Enforces dataset policy/license acknowledgements before training."""

    from ser.config import get_settings
    from ser.data.dataset_consents import (
        DatasetConsentError,
        compute_missing_dataset_consents,
        ensure_dataset_consents,
        persist_dataset_consents,
    )

    settings = get_settings()
    try:
        ensure_dataset_consents(settings=settings, utterances=utterances)
        return
    except DatasetConsentError as err:
        message = str(err)
        interactive = os.isatty(0) and os.isatty(2)
        if not interactive:
            raise

    logger.warning("%s", message)
    print("To acknowledge and continue, type 'accept': ", end="", flush=True)
    try:
        response = input().strip().lower()
    except EOFError:
        response = ""
    if response != "accept":
        raise DatasetConsentError(message)
    missing_policies, missing_licenses = compute_missing_dataset_consents(
        settings=settings,
        utterances=utterances,
    )
    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=sorted(missing_policies),
        accept_license_ids=sorted(missing_licenses),
        source="training",
    )


def _group_metrics_min_support() -> int:
    """Minimum sample support required to report per-group metrics."""

    raw = os.getenv("SER_GROUP_METRICS_MIN_SUPPORT", "").strip()
    if not raw:
        return 20
    try:
        value = int(raw)
    except ValueError:
        return 20
    return max(1, value)


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    window_size_seconds: float,
    window_stride_seconds: float,
) -> list[PoolingWindow]:
    """Creates temporal pooling windows from explicit window policy."""
    return temporal_pooling_windows(
        encoded,
        window_size_seconds=window_size_seconds,
        window_stride_seconds=window_stride_seconds,
    )


def _encode_medium_sequence(
    *,
    audio_path: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    backend: XLSRBackend,
    cache: EmbeddingCache,
    model_id: str | None = None,
) -> EncodedSequence:
    """Encodes one file with cache reuse for medium training."""
    settings: AppConfig = get_settings()
    resolved_model_id = (
        model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    )
    if resolved_model_id is None:
        resolved_model_id = resolve_medium_model_id(settings)

    def _compute_sequence() -> EncodedSequence:
        audio, sample_rate = read_audio_file(
            audio_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        audio_array = np.asarray(audio, dtype=np.float32)
        return backend.encode_sequence(audio_array, sample_rate)

    cache_entry: EmbeddingCacheEntry = cache.get_or_compute(
        audio_path=audio_path,
        backend_id=MEDIUM_BACKEND_ID,
        model_id=resolved_model_id,
        frame_size_seconds=settings.medium_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.medium_runtime.pool_window_stride_seconds,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        compute=_compute_sequence,
    )
    logger.debug(
        "Medium embedding cache %s for %s (%s).",
        "hit" if cache_entry.cache_hit else "miss",
        audio_path,
        cache_entry.cache_key[:12],
    )
    return cache_entry.encoded


def _build_medium_feature_dataset(
    *,
    utterances: list[Utterance],
    backend: XLSRBackend,
    cache: EmbeddingCache,
    model_id: str | None = None,
) -> tuple[np.ndarray, list[str], list[WindowMeta], MediumNoiseControlStats]:
    """Builds frame-pooled medium feature matrix from labeled audio files."""
    settings: AppConfig = get_settings()
    runtime_config = settings.medium_runtime
    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[WindowMeta] = []
    aggregate_stats = MediumNoiseControlStats()
    for utterance in utterances:
        encoded = _encode_medium_sequence(
            audio_path=str(utterance.audio_path),
            start_seconds=utterance.start_seconds,
            duration_seconds=utterance.duration_seconds,
            backend=backend,
            cache=cache,
            model_id=model_id,
        )
        windows = _pooling_windows_from_encoded_frames(
            encoded,
            window_size_seconds=runtime_config.pool_window_size_seconds,
            window_stride_seconds=runtime_config.pool_window_stride_seconds,
        )
        pooled = mean_std_pool(encoded, windows)
        filtered, stats = _apply_medium_noise_controls(
            np.asarray(pooled, dtype=np.float64)
        )
        feature_blocks.append(filtered)
        labels.extend([utterance.label] * int(filtered.shape[0]))
        language = utterance.language or "unknown"
        meta.extend(
            [
                WindowMeta(
                    sample_id=utterance.sample_id,
                    corpus=utterance.corpus,
                    language=language,
                )
            ]
            * int(filtered.shape[0])
        )
        aggregate_stats = _merge_medium_noise_stats(aggregate_stats, stats)

    if not feature_blocks:
        raise RuntimeError("Medium training produced no feature vectors.")
    feature_matrix = np.vstack(feature_blocks).astype(np.float64, copy=False)
    if int(feature_matrix.shape[0]) != len(labels) or int(
        feature_matrix.shape[0]
    ) != len(meta):
        raise RuntimeError("Medium feature/label row mismatch during dataset build.")
    return feature_matrix, labels, meta, aggregate_stats


def _encode_accurate_sequence(
    *,
    audio_path: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    backend: FeatureBackend,
    cache: EmbeddingCache,
    model_id: str | None = None,
    backend_id: str = ACCURATE_BACKEND_ID,
) -> EncodedSequence:
    """Encodes one file with cache reuse for accurate training."""
    settings: AppConfig = get_settings()
    runtime_config: ProfileRuntimeConfig
    if backend_id == ACCURATE_BACKEND_ID:
        runtime_config = settings.accurate_runtime
    elif backend_id == ACCURATE_RESEARCH_BACKEND_ID:
        runtime_config = settings.accurate_research_runtime
    else:
        raise ValueError(f"Unknown accurate backend id: {backend_id!r}.")
    resolved_model_id = (
        model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    )
    if resolved_model_id is None:
        resolved_model_id = (
            resolve_accurate_model_id(settings)
            if backend_id == ACCURATE_BACKEND_ID
            else resolve_accurate_research_model_id(settings)
        )

    def _compute_sequence() -> EncodedSequence:
        audio, sample_rate = read_audio_file(
            audio_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        audio_array = np.asarray(audio, dtype=np.float32)
        return backend.encode_sequence(audio_array, sample_rate)

    cache_entry: EmbeddingCacheEntry = cache.get_or_compute(
        audio_path=audio_path,
        backend_id=backend_id,
        model_id=resolved_model_id,
        frame_size_seconds=runtime_config.pool_window_size_seconds,
        frame_stride_seconds=runtime_config.pool_window_stride_seconds,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        compute=_compute_sequence,
    )
    logger.debug(
        "Accurate backend %s embedding cache %s for %s (%s).",
        backend_id,
        "hit" if cache_entry.cache_hit else "miss",
        audio_path,
        cache_entry.cache_key[:12],
    )
    return cache_entry.encoded


def _build_accurate_feature_dataset(
    *,
    utterances: list[Utterance],
    backend: FeatureBackend,
    cache: EmbeddingCache,
    model_id: str | None = None,
    backend_id: str = ACCURATE_BACKEND_ID,
) -> tuple[np.ndarray, list[str], list[WindowMeta]]:
    """Builds frame-pooled accurate feature matrix from labeled audio files."""
    settings: AppConfig = get_settings()
    runtime_config: ProfileRuntimeConfig
    if backend_id == ACCURATE_BACKEND_ID:
        runtime_config = settings.accurate_runtime
    elif backend_id == ACCURATE_RESEARCH_BACKEND_ID:
        runtime_config = settings.accurate_research_runtime
    else:
        raise ValueError(f"Unknown accurate backend id: {backend_id!r}.")
    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[WindowMeta] = []
    for utterance in utterances:
        encoded = _encode_accurate_sequence(
            audio_path=str(utterance.audio_path),
            start_seconds=utterance.start_seconds,
            duration_seconds=utterance.duration_seconds,
            backend=backend,
            cache=cache,
            model_id=model_id,
            backend_id=backend_id,
        )
        windows = _pooling_windows_from_encoded_frames(
            encoded,
            window_size_seconds=runtime_config.pool_window_size_seconds,
            window_stride_seconds=runtime_config.pool_window_stride_seconds,
        )
        pooled = mean_std_pool(encoded, windows)
        feature_blocks.append(np.asarray(pooled, dtype=np.float64))
        labels.extend([utterance.label] * int(pooled.shape[0]))
        language = utterance.language or "unknown"
        meta.extend(
            [
                WindowMeta(
                    sample_id=utterance.sample_id,
                    corpus=utterance.corpus,
                    language=language,
                )
            ]
            * int(pooled.shape[0])
        )

    if not feature_blocks:
        raise RuntimeError("Accurate training produced no feature vectors.")
    feature_matrix = np.vstack(feature_blocks).astype(np.float64, copy=False)
    if int(feature_matrix.shape[0]) != len(labels) or int(
        feature_matrix.shape[0]
    ) != len(meta):
        raise RuntimeError("Accurate feature/label row mismatch during dataset build.")
    return feature_matrix, labels, meta


def _merge_medium_noise_stats(
    base: MediumNoiseControlStats,
    incoming: MediumNoiseControlStats,
) -> MediumNoiseControlStats:
    """Aggregates per-clip medium noise-control counters."""
    return MediumNoiseControlStats(
        total_windows=base.total_windows + incoming.total_windows,
        kept_windows=base.kept_windows + incoming.kept_windows,
        dropped_low_std_windows=(
            base.dropped_low_std_windows + incoming.dropped_low_std_windows
        ),
        dropped_cap_windows=base.dropped_cap_windows + incoming.dropped_cap_windows,
        forced_keep_windows=base.forced_keep_windows + incoming.forced_keep_windows,
    )


def _apply_medium_noise_controls(
    pooled_features: np.ndarray,
) -> tuple[np.ndarray, MediumNoiseControlStats]:
    """Applies deterministic label-noise controls to pooled medium features."""
    if pooled_features.ndim != 2 or int(pooled_features.shape[1]) <= 0:
        raise RuntimeError("Medium pooled features must be a non-empty 2D matrix.")
    total_windows = int(pooled_features.shape[0])
    if total_windows == 0:
        raise RuntimeError("Medium pooled feature matrix contains zero rows.")
    feature_width = int(pooled_features.shape[1])
    if feature_width % 2 != 0:
        raise RuntimeError(
            "Medium pooled feature width must be even (mean+std concatenation)."
        )

    settings = get_settings()
    min_window_std = settings.medium_training.min_window_std
    max_windows_per_clip = settings.medium_training.max_windows_per_clip

    std_components = pooled_features[:, feature_width // 2 :]
    std_scores = np.linalg.norm(std_components, axis=1) / np.sqrt(feature_width / 2.0)

    keep_mask = np.ones(total_windows, dtype=np.bool_)
    dropped_low_std_windows = 0
    forced_keep_windows = 0
    if min_window_std > 0.0:
        keep_mask = std_scores >= min_window_std
        if not np.any(keep_mask):
            keep_mask[int(np.argmax(std_scores))] = True
            forced_keep_windows = 1
        dropped_low_std_windows = total_windows - int(np.sum(keep_mask))

    filtered = np.asarray(pooled_features[keep_mask], dtype=np.float64)
    dropped_cap_windows = 0
    if max_windows_per_clip > 0 and int(filtered.shape[0]) > max_windows_per_clip:
        selected_indices = np.linspace(
            0,
            int(filtered.shape[0]) - 1,
            num=max_windows_per_clip,
            dtype=np.int64,
        )
        dropped_cap_windows = int(filtered.shape[0]) - max_windows_per_clip
        filtered = np.asarray(filtered[selected_indices], dtype=np.float64)

    return filtered, MediumNoiseControlStats(
        total_windows=total_windows,
        kept_windows=int(filtered.shape[0]),
        dropped_low_std_windows=dropped_low_std_windows,
        dropped_cap_windows=dropped_cap_windows,
        forced_keep_windows=forced_keep_windows,
    )


def train_medium_model() -> None:
    """Trains and persists medium-profile model artifacts with XLS-R metadata."""
    settings = get_settings()
    utterances = load_utterances()
    if utterances is None:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    _ensure_dataset_consents_for_training(utterances=utterances)
    train_utterances, test_utterances, split_metadata = _split_utterances(utterances)
    medium_model_id = resolve_medium_model_id(settings)
    medium_runtime_device, medium_runtime_dtype = _resolve_feature_runtime_selectors(
        backend_id=MEDIUM_BACKEND_ID,
        settings=settings,
    )
    backend = XLSRBackend(
        model_id=medium_model_id,
        cache_dir=settings.models.huggingface_cache_root,
        device=medium_runtime_device,
        dtype=medium_runtime_dtype,
    )
    cache = EmbeddingCache(settings.tmp_folder / "medium_embeddings")
    x_train, y_train, _train_meta, train_noise_stats = _build_medium_feature_dataset(
        utterances=train_utterances,
        backend=backend,
        cache=cache,
        model_id=medium_model_id,
    )
    x_test, y_test, test_meta, test_noise_stats = _build_medium_feature_dataset(
        utterances=test_utterances,
        backend=backend,
        cache=cache,
        model_id=medium_model_id,
    )
    model: EmotionClassifier = _create_classifier()
    logger.info(
        "Medium dataset loaded successfully (train_files=%s, test_files=%s, split=%s).",
        len(train_utterances),
        len(test_utterances),
        split_metadata.split_strategy,
    )

    model.fit(x_train, y_train)
    logger.info("Medium model trained with %s pooled samples", len(x_train))

    y_pred = [str(item) for item in model.predict(x_test)]
    accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
    macro_f1: float = float(f1_score(y_test, y_pred, average="macro"))
    ser_metrics = compute_ser_metrics(y_true=y_test, y_pred=y_pred)
    min_support = _group_metrics_min_support()
    ser_metrics["group_metrics"] = {
        "by_corpus": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.corpus for m in test_meta],
            min_support=min_support,
        ),
        "by_language": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.language for m in test_meta],
            min_support=min_support,
        ),
    }
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    logger.info(msg=f"Medium accuracy: {accuracy * 100:.2f}%")
    logger.info(msg=f"Medium macro F1 score: {macro_f1:.4f}")
    logger.info(msg=f"Medium UAR: {uar:.4f}")

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=MEDIUM_BACKEND_ID,
        profile=MEDIUM_PROFILE_ID,
    )
    artifact = _build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        backend_id=MEDIUM_BACKEND_ID,
        profile=MEDIUM_PROFILE_ID,
        feature_dim=int(x_train.shape[1]),
        frame_size_seconds=settings.medium_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.medium_runtime.pool_window_stride_seconds,
        pooling_strategy=MEDIUM_POOLING_STRATEGY,
        backend_model_id=medium_model_id,
        torch_device=medium_runtime_device,
        torch_dtype=medium_runtime_dtype,
        provenance=provenance,
    )
    artifact_metadata_obj = artifact.get("metadata")
    if not isinstance(artifact_metadata_obj, dict):
        raise RuntimeError("Model artifact metadata is missing before persistence.")
    artifact_metadata = _normalize_v2_artifact_metadata(artifact_metadata_obj)
    persisted_artifacts = _persist_model_artifacts(model=model, artifact=artifact)
    logger.info("Medium model saved to %s", persisted_artifacts.pickle_path)
    if persisted_artifacts.secure_path is not None:
        logger.info(
            "Medium secure model saved to %s",
            persisted_artifacts.secure_path,
        )

    report = _build_training_report(
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
        data_controls={
            "dataset": _build_dataset_controls(utterances),
            "medium_noise_controls": {
                "min_window_std": settings.medium_training.min_window_std,
                "max_windows_per_clip": settings.medium_training.max_windows_per_clip,
                "train": {
                    "total_windows": train_noise_stats.total_windows,
                    "kept_windows": train_noise_stats.kept_windows,
                    "dropped_low_std_windows": train_noise_stats.dropped_low_std_windows,
                    "dropped_cap_windows": train_noise_stats.dropped_cap_windows,
                    "forced_keep_windows": train_noise_stats.forced_keep_windows,
                },
                "test": {
                    "total_windows": test_noise_stats.total_windows,
                    "kept_windows": test_noise_stats.kept_windows,
                    "dropped_low_std_windows": test_noise_stats.dropped_low_std_windows,
                    "dropped_cap_windows": test_noise_stats.dropped_cap_windows,
                    "forced_keep_windows": test_noise_stats.forced_keep_windows,
                },
            },
            "medium_grouped_evaluation": {
                "split_strategy": split_metadata.split_strategy,
                "speaker_grouped": split_metadata.speaker_grouped,
                "speaker_id_coverage": split_metadata.speaker_id_coverage,
                "train_unique_speakers": split_metadata.train_unique_speakers,
                "test_unique_speakers": split_metadata.test_unique_speakers,
                "speaker_overlap_count": split_metadata.speaker_overlap_count,
            },
        },
    )
    _persist_training_report(report, settings.models.training_report_file)
    logger.info(
        "Medium training report saved to %s", settings.models.training_report_file
    )


def train_accurate_model() -> None:
    """Trains and persists accurate-profile model artifacts with Whisper metadata."""
    settings = get_settings()
    utterances = load_utterances()
    if utterances is None:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    _ensure_dataset_consents_for_training(utterances=utterances)
    train_utterances, test_utterances, split_metadata = _split_utterances(utterances)
    accurate_model_id = resolve_accurate_model_id(settings)
    accurate_runtime_device, accurate_runtime_dtype = (
        _resolve_feature_runtime_selectors(
            backend_id=ACCURATE_BACKEND_ID,
            settings=settings,
        )
    )
    backend = WhisperBackend(
        model_id=accurate_model_id,
        cache_dir=settings.models.huggingface_cache_root,
        device=accurate_runtime_device,
        dtype=accurate_runtime_dtype,
    )
    cache = EmbeddingCache(settings.tmp_folder / "accurate_embeddings")
    x_train, y_train, _train_meta = _build_accurate_feature_dataset(
        utterances=train_utterances,
        backend=backend,
        cache=cache,
        model_id=accurate_model_id,
        backend_id=ACCURATE_BACKEND_ID,
    )
    x_test, y_test, test_meta = _build_accurate_feature_dataset(
        utterances=test_utterances,
        backend=backend,
        cache=cache,
        model_id=accurate_model_id,
        backend_id=ACCURATE_BACKEND_ID,
    )
    model: EmotionClassifier = _create_classifier()
    logger.info(
        "Accurate dataset loaded successfully (train_files=%s, test_files=%s, split=%s).",
        len(train_utterances),
        len(test_utterances),
        split_metadata.split_strategy,
    )

    model.fit(x_train, y_train)
    logger.info("Accurate model trained with %s pooled samples", len(x_train))

    y_pred = [str(item) for item in model.predict(x_test)]
    accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
    macro_f1: float = float(f1_score(y_test, y_pred, average="macro"))
    ser_metrics = compute_ser_metrics(y_true=y_test, y_pred=y_pred)
    min_support = _group_metrics_min_support()
    ser_metrics["group_metrics"] = {
        "by_corpus": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.corpus for m in test_meta],
            min_support=min_support,
        ),
        "by_language": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.language for m in test_meta],
            min_support=min_support,
        ),
    }
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    logger.info(msg=f"Accurate accuracy: {accuracy * 100:.2f}%")
    logger.info(msg=f"Accurate macro F1 score: {macro_f1:.4f}")
    logger.info(msg=f"Accurate UAR: {uar:.4f}")

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=ACCURATE_BACKEND_ID,
        profile=ACCURATE_PROFILE_ID,
    )
    artifact = _build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        backend_id=ACCURATE_BACKEND_ID,
        profile=ACCURATE_PROFILE_ID,
        feature_dim=int(x_train.shape[1]),
        frame_size_seconds=settings.accurate_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.accurate_runtime.pool_window_stride_seconds,
        pooling_strategy=ACCURATE_POOLING_STRATEGY,
        backend_model_id=accurate_model_id,
        torch_device=accurate_runtime_device,
        torch_dtype=accurate_runtime_dtype,
        provenance=provenance,
    )
    artifact_metadata_obj = artifact.get("metadata")
    if not isinstance(artifact_metadata_obj, dict):
        raise RuntimeError("Model artifact metadata is missing before persistence.")
    artifact_metadata = _normalize_v2_artifact_metadata(artifact_metadata_obj)
    persisted_artifacts = _persist_model_artifacts(model=model, artifact=artifact)
    logger.info("Accurate model saved to %s", persisted_artifacts.pickle_path)
    if persisted_artifacts.secure_path is not None:
        logger.info(
            "Accurate secure model saved to %s",
            persisted_artifacts.secure_path,
        )

    report = _build_training_report(
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
        data_controls={
            "dataset": _build_dataset_controls(utterances),
            "accurate_grouped_evaluation": {
                "split_strategy": split_metadata.split_strategy,
                "speaker_grouped": split_metadata.speaker_grouped,
                "speaker_id_coverage": split_metadata.speaker_id_coverage,
                "train_unique_speakers": split_metadata.train_unique_speakers,
                "test_unique_speakers": split_metadata.test_unique_speakers,
                "speaker_overlap_count": split_metadata.speaker_overlap_count,
            },
        },
    )
    _persist_training_report(report, settings.models.training_report_file)
    logger.info(
        "Accurate training report saved to %s", settings.models.training_report_file
    )


def train_accurate_research_model() -> None:
    """Trains and persists accurate-research model artifacts with emotion2vec metadata."""
    settings = get_settings()
    allowed_restricted_backends = parse_allowed_restricted_backends_env()
    persisted_consents = load_persisted_backend_consents(settings=settings)
    ensure_backend_access(
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    utterances = load_utterances()
    if utterances is None:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    _ensure_dataset_consents_for_training(utterances=utterances)
    train_utterances, test_utterances, split_metadata = _split_utterances(utterances)
    accurate_research_model_id = resolve_accurate_research_model_id(settings)
    accurate_research_runtime_device, accurate_research_runtime_dtype = (
        _resolve_feature_runtime_selectors(
            backend_id=ACCURATE_RESEARCH_BACKEND_ID,
            settings=settings,
        )
    )
    backend = Emotion2VecBackend(
        model_id=accurate_research_model_id,
        device=accurate_research_runtime_device,
        modelscope_cache_root=settings.models.modelscope_cache_root,
        huggingface_cache_root=settings.models.huggingface_cache_root,
    )
    cache = EmbeddingCache(settings.tmp_folder / "accurate_research_embeddings")
    x_train, y_train, _train_meta = _build_accurate_feature_dataset(
        utterances=train_utterances,
        backend=backend,
        cache=cache,
        model_id=accurate_research_model_id,
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
    )
    x_test, y_test, test_meta = _build_accurate_feature_dataset(
        utterances=test_utterances,
        backend=backend,
        cache=cache,
        model_id=accurate_research_model_id,
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
    )
    model: EmotionClassifier = _create_classifier()
    logger.info(
        "Accurate-research dataset loaded successfully "
        "(train_files=%s, test_files=%s, split=%s).",
        len(train_utterances),
        len(test_utterances),
        split_metadata.split_strategy,
    )

    model.fit(x_train, y_train)
    logger.info("Accurate-research model trained with %s pooled samples", len(x_train))

    y_pred = [str(item) for item in model.predict(x_test)]
    accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
    macro_f1: float = float(f1_score(y_test, y_pred, average="macro"))
    ser_metrics = compute_ser_metrics(y_true=y_test, y_pred=y_pred)
    min_support = _group_metrics_min_support()
    ser_metrics["group_metrics"] = {
        "by_corpus": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.corpus for m in test_meta],
            min_support=min_support,
        ),
        "by_language": compute_grouped_ser_metrics_by_sample(
            y_true=y_test,
            y_pred=y_pred,
            sample_ids=[m.sample_id for m in test_meta],
            group_ids=[m.language for m in test_meta],
            min_support=min_support,
        ),
    }
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    logger.info(msg=f"Accurate-research accuracy: {accuracy * 100:.2f}%")
    logger.info(msg=f"Accurate-research macro F1 score: {macro_f1:.4f}")
    logger.info(msg=f"Accurate-research UAR: {uar:.4f}")

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
        profile=ACCURATE_RESEARCH_PROFILE_ID,
    )
    artifact = _build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
        profile=ACCURATE_RESEARCH_PROFILE_ID,
        feature_dim=int(x_train.shape[1]),
        frame_size_seconds=(
            settings.accurate_research_runtime.pool_window_size_seconds
        ),
        frame_stride_seconds=(
            settings.accurate_research_runtime.pool_window_stride_seconds
        ),
        pooling_strategy=ACCURATE_POOLING_STRATEGY,
        backend_model_id=accurate_research_model_id,
        torch_device=accurate_research_runtime_device,
        torch_dtype=accurate_research_runtime_dtype,
        provenance=provenance,
    )
    artifact_metadata_obj = artifact.get("metadata")
    if not isinstance(artifact_metadata_obj, dict):
        raise RuntimeError("Model artifact metadata is missing before persistence.")
    artifact_metadata = _normalize_v2_artifact_metadata(artifact_metadata_obj)
    persisted_artifacts = _persist_model_artifacts(model=model, artifact=artifact)
    logger.info("Accurate-research model saved to %s", persisted_artifacts.pickle_path)
    if persisted_artifacts.secure_path is not None:
        logger.info(
            "Accurate-research secure model saved to %s",
            persisted_artifacts.secure_path,
        )

    report = _build_training_report(
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
        data_controls={
            "dataset": _build_dataset_controls(utterances),
            "accurate_grouped_evaluation": {
                "split_strategy": split_metadata.split_strategy,
                "speaker_grouped": split_metadata.speaker_grouped,
                "speaker_id_coverage": split_metadata.speaker_id_coverage,
                "train_unique_speakers": split_metadata.train_unique_speakers,
                "test_unique_speakers": split_metadata.test_unique_speakers,
                "speaker_overlap_count": split_metadata.speaker_overlap_count,
            },
        },
    )
    _persist_training_report(report, settings.models.training_report_file)
    logger.info(
        "Accurate-research training report saved to %s",
        settings.models.training_report_file,
    )


def train_model() -> None:
    """Trains the MLP classifier and persists model + training report artifacts.

    Raises:
        RuntimeError: If no training data could be loaded from the dataset path.
    """
    settings = get_settings()
    utterances_for_consent = load_utterances()
    if utterances_for_consent:
        _ensure_dataset_consents_for_training(utterances=list(utterances_for_consent))

    if data := load_data(test_size=settings.training.test_size):
        x_train, x_test, y_train, y_test = data
        model: EmotionClassifier = _create_classifier()
        logger.info(msg="Dataset loaded successfully.")
    else:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")

    model.fit(x_train, y_train)
    logger.info(msg=f"Model trained with {len(x_train)} samples")

    y_pred = [str(item) for item in model.predict(x_test)]
    accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
    macro_f1: float = float(f1_score(y_test, y_pred, average="macro"))
    ser_metrics = compute_ser_metrics(y_true=y_test, y_pred=y_pred)
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")
    logger.info(msg=f"Macro F1 score: {macro_f1:.4f}")
    logger.info(msg=f"UAR: {uar:.4f}")

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=DEFAULT_BACKEND_ID,
        profile=DEFAULT_PROFILE_ID,
    )
    artifact = _build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        provenance=provenance,
    )
    artifact_metadata_obj = artifact.get("metadata")
    if not isinstance(artifact_metadata_obj, dict):
        raise RuntimeError("Model artifact metadata is missing before persistence.")
    artifact_metadata = _normalize_v2_artifact_metadata(artifact_metadata_obj)
    persisted_artifacts = _persist_model_artifacts(model=model, artifact=artifact)
    logger.info(msg=f"Model saved to {persisted_artifacts.pickle_path}")
    if persisted_artifacts.secure_path is not None:
        logger.info(msg=f"Secure model saved to {persisted_artifacts.secure_path}")

    report = _build_training_report(
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
    )
    _persist_training_report(report, settings.models.training_report_file)
    logger.info(msg=f"Training report saved to {settings.models.training_report_file}")


def load_model(
    settings: AppConfig | None = None,
    *,
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> LoadedModel:
    """Loads the serialized SER model from disk.

    Args:
        settings: Optional settings snapshot used to resolve model/report paths.
        expected_backend_id: Optional backend-id compatibility filter.
        expected_profile: Optional profile compatibility filter.
        expected_backend_model_id: Optional backend-model-id compatibility filter.

    Returns:
        The loaded model plus expected feature-vector size.

    Raises:
        FileNotFoundError: If no trained model artifact could be found.
        ValueError: If model artifacts exist but none can be deserialized.
    """
    try:
        active_settings = settings if settings is not None else get_settings()
        candidate, loaded_model = _resolve_model_for_loading(
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


def _frame_confidence_and_probabilities(
    model: EmotionClassifier,
    feature_matrix: np.ndarray,
    frame_count: int,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability maps."""
    fallback_confidence = [1.0] * frame_count
    fallback_probabilities: list[dict[str, float] | None] = [None] * frame_count

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        logger.warning(
            "Loaded model does not expose predict_proba; using confidence=1.0 fallback."
        )
        return fallback_confidence, fallback_probabilities

    classes_attr = getattr(model, "classes_", None)
    class_values: list[object] | None = None
    if isinstance(classes_attr, np.ndarray):
        class_values = list(classes_attr.tolist())
    elif isinstance(classes_attr, list | tuple):
        class_values = list(classes_attr)
    if class_values is None:
        logger.warning(
            "Loaded model predict_proba path missing classes_; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    class_labels = [str(item) for item in class_values]
    raw_probabilities = np.asarray(predict_proba(feature_matrix), dtype=np.float64)
    if raw_probabilities.ndim != 2:
        logger.warning(
            "Unexpected predict_proba output shape %s; using confidence fallback.",
            raw_probabilities.shape,
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[0] != frame_count:
        logger.warning(
            "predict_proba frame count mismatch (expected=%s, got=%s); using fallback.",
            frame_count,
            raw_probabilities.shape[0],
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[1] != len(class_labels):
        logger.warning(
            "predict_proba class count mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            raw_probabilities.shape[1],
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {class_labels[idx]: float(row[idx]) for idx in range(len(class_labels))}
        for row in raw_probabilities
    ]
    return confidences, probabilities


def _aggregate_probabilities(
    probabilities: list[dict[str, float] | None],
) -> dict[str, float] | None:
    """Averages per-frame probabilities when all frames provide full maps."""
    if not probabilities or any(item is None for item in probabilities):
        return None

    first = probabilities[0]
    if first is None:
        return None
    labels = list(first.keys())
    if any(
        item is None or set(item.keys()) != set(labels) for item in probabilities[1:]
    ):
        return None

    aggregates: dict[str, float] = {}
    for label in labels:
        values = [item[label] for item in probabilities if item is not None]
        aggregates[label] = float(fmean(values))
    return aggregates


def _segment_predictions(
    frame_predictions: list[FramePrediction],
) -> list[SegmentPrediction]:
    """Merges adjacent equal frame labels into segment-level predictions."""
    if not frame_predictions:
        return []

    segments: list[SegmentPrediction] = []
    active_emotion = frame_predictions[0].emotion
    active_start = frame_predictions[0].start_seconds
    active_end = frame_predictions[0].end_seconds
    active_confidences = [frame_predictions[0].confidence]
    active_probabilities = [frame_predictions[0].probabilities]

    for frame in frame_predictions[1:]:
        if frame.emotion == active_emotion:
            active_end = frame.end_seconds
            active_confidences.append(frame.confidence)
            active_probabilities.append(frame.probabilities)
            continue

        segments.append(
            SegmentPrediction(
                emotion=active_emotion,
                start_seconds=active_start,
                end_seconds=active_end,
                confidence=float(fmean(active_confidences)),
                probabilities=_aggregate_probabilities(active_probabilities),
            )
        )
        active_emotion = frame.emotion
        active_start = frame.start_seconds
        active_end = frame.end_seconds
        active_confidences = [frame.confidence]
        active_probabilities = [frame.probabilities]

    segments.append(
        SegmentPrediction(
            emotion=active_emotion,
            start_seconds=active_start,
            end_seconds=active_end,
            confidence=float(fmean(active_confidences)),
            probabilities=_aggregate_probabilities(active_probabilities),
        )
    )
    return segments


def _predict_emotions_detailed_with_model(
    file: str,
    *,
    loaded_model: LoadedModel,
) -> InferenceResult:
    """Runs inference with a preloaded model and returns detailed predictions."""
    model = loaded_model.model

    feature_frames = extract_feature_frames(file)
    if not feature_frames:
        logger.warning("No features extracted for file %s.", file)
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    feature_vectors: list[np.ndarray] = [frame.features for frame in feature_frames]
    if loaded_model.expected_feature_size is not None:
        invalid_sizes = {
            vector.shape[0]
            for vector in feature_vectors
            if vector.shape[0] != loaded_model.expected_feature_size
        }
        if invalid_sizes:
            raise ValueError(
                "Feature vector size mismatch for loaded model. "
                f"Expected {loaded_model.expected_feature_size}, "
                f"got {sorted(invalid_sizes)}."
            )

    feature_matrix = np.asarray(feature_vectors, dtype=np.float64)
    predicted_emotions: list[str] = [
        str(item) for item in model.predict(feature_matrix)
    ]
    if len(predicted_emotions) != len(feature_frames):
        raise RuntimeError(
            "Frame/prediction length mismatch. "
            f"Got {len(feature_frames)} frames and {len(predicted_emotions)} predictions."
        )
    confidences, probabilities = _frame_confidence_and_probabilities(
        model=model,
        feature_matrix=feature_matrix,
        frame_count=len(feature_frames),
    )

    logger.debug(
        "Emotion model prediction completed for %d frames.",
        len(predicted_emotions),
    )
    if not predicted_emotions:
        logger.warning("No emotions predicted for file %s.", file)
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    frame_predictions = [
        FramePrediction(
            start_seconds=feature_frames[idx].start_seconds,
            end_seconds=feature_frames[idx].end_seconds,
            emotion=predicted_emotions[idx],
            confidence=confidences[idx],
            probabilities=probabilities[idx],
        )
        for idx in range(len(feature_frames))
    ]
    logger.debug("Timestamp extraction started.")
    segment_predictions = _segment_predictions(frame_predictions)
    logger.debug(
        "Timestamp extraction completed for %d segments.",
        len(segment_predictions),
    )
    return InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=segment_predictions,
        frames=frame_predictions,
    )


def predict_emotions_detailed(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
) -> InferenceResult:
    """Runs inference and returns detailed frame + segment predictions."""
    active_loaded_model = loaded_model if loaded_model is not None else load_model()
    return _predict_emotions_detailed_with_model(
        file,
        loaded_model=active_loaded_model,
    )


def predict_emotions(
    file: str,
    *,
    loaded_model: LoadedModel | None = None,
) -> list[EmotionSegment]:
    """Compatibility wrapper returning legacy emotion segments."""
    inference = (
        predict_emotions_detailed(file)
        if loaded_model is None
        else predict_emotions_detailed(
            file,
            loaded_model=loaded_model,
        )
    )
    return to_legacy_emotion_segments(inference)
