"""Runtime profile resolution for staged pipeline rollout."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from ser.config import AppConfig

type ProfileName = Literal["fast", "medium", "accurate", "accurate-research"]
type ProfileEnableFlag = Literal[
    "SER_ENABLE_MEDIUM_PROFILE",
    "SER_ENABLE_ACCURATE_PROFILE",
    "SER_ENABLE_ACCURATE_RESEARCH_PROFILE",
]
type TranscriptionBackendId = Literal["stable_whisper", "faster_whisper"]


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved runtime profile configuration.

    Attributes:
        name: Canonical profile identifier.
        description: Short profile intent summary.
    """

    name: ProfileName
    description: str


@dataclass(frozen=True)
class ProfileModelDefinition:
    """Model-id configuration keys for one runtime profile."""

    env_var: str | None
    default_model_id: str | None


@dataclass(frozen=True)
class ProfileTranscriptionDefaults:
    """Default Whisper transcription controls for one runtime profile."""

    backend_id: TranscriptionBackendId
    model_name: str
    use_demucs: bool
    use_vad: bool


@dataclass(frozen=True)
class ProfileRuntimeDefaults:
    """Default runtime controls for one runtime profile."""

    timeout_seconds: float
    max_timeout_retries: int
    max_transient_retries: int
    retry_backoff_seconds: float
    pool_window_size_seconds: float
    pool_window_stride_seconds: float
    post_smoothing_window_frames: int
    post_hysteresis_enter_confidence: float
    post_hysteresis_exit_confidence: float
    post_min_segment_duration_seconds: float
    process_isolation: bool


@dataclass(frozen=True)
class ProfileRuntimeEnvDefinition:
    """Environment-variable keys for one profile runtime definition."""

    timeout_seconds: str | None
    max_timeout_retries: str | None
    max_transient_retries: str | None
    retry_backoff_seconds: str | None
    pool_window_size_seconds: str | None
    pool_window_stride_seconds: str | None
    post_smoothing_window_frames: str | None
    post_hysteresis_enter_confidence: str | None
    post_hysteresis_exit_confidence: str | None
    post_min_segment_duration_seconds: str | None
    process_isolation: str | None


@dataclass(frozen=True)
class ProfileFeatureRuntimeDefaults:
    """Optional feature-runtime selector defaults for one profile backend."""

    torch_device: str | None
    torch_dtype: str | None


@dataclass(frozen=True)
class ProfileCatalogEntry:
    """Declarative profile definition entry loaded from YAML."""

    name: ProfileName
    description: str
    backend_id: str
    required_modules: tuple[str, ...]
    enable_flag: ProfileEnableFlag | None
    enabled_by_default: bool
    model: ProfileModelDefinition
    transcription_defaults: ProfileTranscriptionDefaults
    runtime_defaults: ProfileRuntimeDefaults
    runtime_env: ProfileRuntimeEnvDefinition
    feature_runtime_defaults: ProfileFeatureRuntimeDefaults | None


_PROFILE_ORDER: tuple[ProfileName, ...] = (
    "fast",
    "medium",
    "accurate",
    "accurate-research",
)
_PROFILE_DEFS_FILE_NAME = "profile_defs.yaml"
_ALLOWED_ENABLE_FLAGS: frozenset[str] = frozenset(
    {
        "SER_ENABLE_MEDIUM_PROFILE",
        "SER_ENABLE_ACCURATE_PROFILE",
        "SER_ENABLE_ACCURATE_RESEARCH_PROFILE",
    }
)
_ALLOWED_TRANSCRIPTION_BACKENDS: frozenset[str] = frozenset(
    {"stable_whisper", "faster_whisper"}
)
_ALLOWED_TORCH_DTYPE_SELECTORS: frozenset[str] = frozenset(
    {"auto", "float16", "float32", "bfloat16"}
)


_PROFILE_DEFS_SCHEMA_VERSION = 1


def _catalog_file_path() -> Path:
    """Returns the on-disk path for the profile definitions YAML file."""
    return Path(__file__).with_name(_PROFILE_DEFS_FILE_NAME)


def _read_catalog_payload() -> dict[str, object]:
    """Reads raw YAML payload for profile definitions with strict shape checks."""
    yaml_module: Any
    try:
        yaml_module = importlib.import_module("yaml")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Missing dependency 'PyYAML' required for profile catalog loading."
        ) from err
    path = _catalog_file_path()
    try:
        with path.open("r", encoding="utf-8") as catalog_file:
            payload = yaml_module.safe_load(catalog_file)
    except FileNotFoundError as err:
        raise RuntimeError(f"Profile definitions file not found at {path}.") from err
    except Exception as err:
        raise RuntimeError(
            f"Invalid YAML in profile definitions file at {path}: {err}"
        ) from err
    if not isinstance(payload, dict):
        raise RuntimeError(
            "Profile definitions payload must be a mapping with top-level keys "
            "'schema_version' and 'profiles'."
        )
    return cast(dict[str, object], payload)


def _read_optional_text(
    raw_mapping: dict[str, object],
    *,
    key: str,
    entry_name: str,
) -> str | None:
    """Reads a nullable non-empty string key from one profile mapping."""
    raw_value = raw_mapping.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    return raw_value.strip()


def _read_required_float(
    raw_mapping: dict[str, object],
    *,
    key: str,
    entry_name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Reads and validates one required numeric float field."""
    raw_value = raw_mapping.get(key)
    if not isinstance(raw_value, int | float):
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    parsed = float(raw_value)
    if minimum is not None and parsed < minimum:
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    if maximum is not None and parsed > maximum:
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    return parsed


def _read_required_int(
    raw_mapping: dict[str, object],
    *,
    key: str,
    entry_name: str,
    minimum: int | None = None,
) -> int:
    """Reads and validates one required integer field."""
    raw_value = raw_mapping.get(key)
    if not isinstance(raw_value, int):
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    if minimum is not None and raw_value < minimum:
        raise RuntimeError(
            f"Profile definition entry {entry_name!r} has invalid {key!r}."
        )
    return raw_value


def _normalize_torch_device_selector(value: str) -> str | None:
    """Normalizes one torch device selector or returns None when invalid."""
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "mps", "cuda"}:
        return normalized
    if re.fullmatch(r"cuda:\d+", normalized):
        return normalized
    return None


def _normalize_torch_dtype_selector(value: str) -> str | None:
    """Normalizes one torch dtype selector or returns None when invalid."""
    normalized = value.strip().lower()
    if normalized in _ALLOWED_TORCH_DTYPE_SELECTORS:
        return normalized
    return None


def _validate_model_definition(
    *,
    name: ProfileName,
    raw: object,
) -> ProfileModelDefinition:
    """Validates model-id definition mapping for one profile."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"Profile definition entry {name!r} has invalid 'model'.")
    default_model_id = _read_optional_text(
        cast(dict[str, object], raw),
        key="default_model_id",
        entry_name=name,
    )
    env_var = _read_optional_text(
        cast(dict[str, object], raw),
        key="env_var",
        entry_name=name,
    )
    return ProfileModelDefinition(env_var=env_var, default_model_id=default_model_id)


def _validate_runtime_defaults(
    *,
    name: ProfileName,
    raw: object,
) -> ProfileRuntimeDefaults:
    """Validates runtime defaults for one profile."""
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'runtime_defaults'."
        )
    runtime_defaults = cast(dict[str, object], raw)
    enter_confidence = _read_required_float(
        runtime_defaults,
        key="post_hysteresis_enter_confidence",
        entry_name=name,
        minimum=0.0,
        maximum=1.0,
    )
    exit_confidence = _read_required_float(
        runtime_defaults,
        key="post_hysteresis_exit_confidence",
        entry_name=name,
        minimum=0.0,
        maximum=1.0,
    )
    if enter_confidence < exit_confidence:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid hysteresis thresholds."
        )
    process_isolation = runtime_defaults.get("process_isolation")
    if not isinstance(process_isolation, bool):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'process_isolation'."
        )
    return ProfileRuntimeDefaults(
        timeout_seconds=_read_required_float(
            runtime_defaults,
            key="timeout_seconds",
            entry_name=name,
            minimum=0.0,
        ),
        max_timeout_retries=_read_required_int(
            runtime_defaults,
            key="max_timeout_retries",
            entry_name=name,
            minimum=0,
        ),
        max_transient_retries=_read_required_int(
            runtime_defaults,
            key="max_transient_retries",
            entry_name=name,
            minimum=0,
        ),
        retry_backoff_seconds=_read_required_float(
            runtime_defaults,
            key="retry_backoff_seconds",
            entry_name=name,
            minimum=0.0,
        ),
        pool_window_size_seconds=_read_required_float(
            runtime_defaults,
            key="pool_window_size_seconds",
            entry_name=name,
            minimum=0.05,
        ),
        pool_window_stride_seconds=_read_required_float(
            runtime_defaults,
            key="pool_window_stride_seconds",
            entry_name=name,
            minimum=0.05,
        ),
        post_smoothing_window_frames=_read_required_int(
            runtime_defaults,
            key="post_smoothing_window_frames",
            entry_name=name,
            minimum=1,
        ),
        post_hysteresis_enter_confidence=enter_confidence,
        post_hysteresis_exit_confidence=exit_confidence,
        post_min_segment_duration_seconds=_read_required_float(
            runtime_defaults,
            key="post_min_segment_duration_seconds",
            entry_name=name,
            minimum=0.0,
        ),
        process_isolation=process_isolation,
    )


def _validate_transcription_defaults(
    *,
    name: ProfileName,
    raw: object,
) -> ProfileTranscriptionDefaults:
    """Validates transcription defaults for one profile."""
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'transcription_defaults'."
        )
    transcription_defaults = cast(dict[str, object], raw)
    backend_id_raw = _read_optional_text(
        transcription_defaults,
        key="backend_id",
        entry_name=name,
    )
    if backend_id_raw is None or backend_id_raw not in _ALLOWED_TRANSCRIPTION_BACKENDS:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'backend_id'."
        )
    model_name = _read_optional_text(
        transcription_defaults,
        key="model_name",
        entry_name=name,
    )
    if model_name is None:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'model_name'."
        )
    use_demucs = transcription_defaults.get("use_demucs")
    if not isinstance(use_demucs, bool):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'use_demucs'."
        )
    use_vad = transcription_defaults.get("use_vad")
    if not isinstance(use_vad, bool):
        raise RuntimeError(f"Profile definition entry {name!r} has invalid 'use_vad'.")
    return ProfileTranscriptionDefaults(
        backend_id=cast(TranscriptionBackendId, backend_id_raw),
        model_name=model_name,
        use_demucs=use_demucs,
        use_vad=use_vad,
    )


def _validate_runtime_env(
    *,
    name: ProfileName,
    raw: object,
) -> ProfileRuntimeEnvDefinition:
    """Validates runtime environment variable mapping for one profile."""
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'runtime_env'."
        )
    runtime_env = cast(dict[str, object], raw)
    return ProfileRuntimeEnvDefinition(
        timeout_seconds=_read_optional_text(
            runtime_env,
            key="timeout_seconds",
            entry_name=name,
        ),
        max_timeout_retries=_read_optional_text(
            runtime_env,
            key="max_timeout_retries",
            entry_name=name,
        ),
        max_transient_retries=_read_optional_text(
            runtime_env,
            key="max_transient_retries",
            entry_name=name,
        ),
        retry_backoff_seconds=_read_optional_text(
            runtime_env,
            key="retry_backoff_seconds",
            entry_name=name,
        ),
        pool_window_size_seconds=_read_optional_text(
            runtime_env,
            key="pool_window_size_seconds",
            entry_name=name,
        ),
        pool_window_stride_seconds=_read_optional_text(
            runtime_env,
            key="pool_window_stride_seconds",
            entry_name=name,
        ),
        post_smoothing_window_frames=_read_optional_text(
            runtime_env,
            key="post_smoothing_window_frames",
            entry_name=name,
        ),
        post_hysteresis_enter_confidence=_read_optional_text(
            runtime_env,
            key="post_hysteresis_enter_confidence",
            entry_name=name,
        ),
        post_hysteresis_exit_confidence=_read_optional_text(
            runtime_env,
            key="post_hysteresis_exit_confidence",
            entry_name=name,
        ),
        post_min_segment_duration_seconds=_read_optional_text(
            runtime_env,
            key="post_min_segment_duration_seconds",
            entry_name=name,
        ),
        process_isolation=_read_optional_text(
            runtime_env,
            key="process_isolation",
            entry_name=name,
        ),
    )


def _validate_feature_runtime_defaults(
    *,
    name: ProfileName,
    raw: object,
) -> ProfileFeatureRuntimeDefaults | None:
    """Validates optional feature-runtime defaults for one profile."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'feature_runtime_defaults'."
        )
    runtime_defaults = cast(dict[str, object], raw)
    torch_device_raw = _read_optional_text(
        runtime_defaults,
        key="torch_device",
        entry_name=name,
    )
    torch_dtype_raw = _read_optional_text(
        runtime_defaults,
        key="torch_dtype",
        entry_name=name,
    )
    torch_device = (
        _normalize_torch_device_selector(torch_device_raw)
        if torch_device_raw is not None
        else None
    )
    if torch_device_raw is not None and torch_device is None:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid "
            "'feature_runtime_defaults.torch_device'."
        )
    torch_dtype = (
        _normalize_torch_dtype_selector(torch_dtype_raw)
        if torch_dtype_raw is not None
        else None
    )
    if torch_dtype_raw is not None and torch_dtype is None:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid "
            "'feature_runtime_defaults.torch_dtype'."
        )
    if torch_device is None and torch_dtype is None:
        raise RuntimeError(
            f"Profile definition entry {name!r} must define at least one of "
            "'feature_runtime_defaults.torch_device' or "
            "'feature_runtime_defaults.torch_dtype'."
        )
    return ProfileFeatureRuntimeDefaults(
        torch_device=torch_device,
        torch_dtype=torch_dtype,
    )


def _validate_catalog_entry(name: ProfileName, raw: object) -> ProfileCatalogEntry:
    """Validates one profile definition entry loaded from YAML."""
    if not isinstance(raw, dict):
        raise RuntimeError(f"Profile definition entry {name!r} must be a mapping.")
    raw_mapping = cast(dict[str, object], raw)
    description = raw_mapping.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'description'."
        )
    backend_id = raw_mapping.get("backend_id")
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'backend_id'."
        )
    required_modules_raw = raw_mapping.get("required_modules")
    if not isinstance(required_modules_raw, list):
        raise RuntimeError(
            f"Profile definition entry {name!r} must define list 'required_modules'."
        )
    required_modules: list[str] = []
    for item in required_modules_raw:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(
                f"Profile definition entry {name!r} has invalid module in "
                "'required_modules'."
            )
        required_modules.append(item.strip())
    enable_flag_raw = raw_mapping.get("enable_flag")
    if enable_flag_raw is None:
        enable_flag: ProfileEnableFlag | None = None
    elif isinstance(enable_flag_raw, str) and enable_flag_raw in _ALLOWED_ENABLE_FLAGS:
        enable_flag = cast(ProfileEnableFlag, enable_flag_raw)
    else:
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'enable_flag'."
        )
    enabled_by_default = raw_mapping.get("enabled_by_default")
    if not isinstance(enabled_by_default, bool):
        raise RuntimeError(
            f"Profile definition entry {name!r} has invalid 'enabled_by_default'."
        )
    model_definition = _validate_model_definition(
        name=name,
        raw=raw_mapping.get("model"),
    )
    transcription_defaults = _validate_transcription_defaults(
        name=name,
        raw=raw_mapping.get("transcription_defaults"),
    )
    runtime_defaults = _validate_runtime_defaults(
        name=name,
        raw=raw_mapping.get("runtime_defaults"),
    )
    runtime_env = _validate_runtime_env(name=name, raw=raw_mapping.get("runtime_env"))
    feature_runtime_defaults = _validate_feature_runtime_defaults(
        name=name,
        raw=raw_mapping.get("feature_runtime_defaults"),
    )
    return ProfileCatalogEntry(
        name=name,
        description=description.strip(),
        backend_id=backend_id.strip(),
        required_modules=tuple(required_modules),
        enable_flag=enable_flag,
        enabled_by_default=enabled_by_default,
        model=model_definition,
        transcription_defaults=transcription_defaults,
        runtime_defaults=runtime_defaults,
        runtime_env=runtime_env,
        feature_runtime_defaults=feature_runtime_defaults,
    )


def _load_profile_catalog() -> MappingProxyType[ProfileName, ProfileCatalogEntry]:
    """Loads and validates profile definition entries from YAML."""
    payload = _read_catalog_payload()
    schema_version = payload.get("schema_version")
    if schema_version != _PROFILE_DEFS_SCHEMA_VERSION:
        raise RuntimeError(
            "Profile definitions payload has unsupported schema_version. "
            f"Expected {_PROFILE_DEFS_SCHEMA_VERSION}, got {schema_version!r}."
        )
    profiles_raw = payload.get("profiles")
    if not isinstance(profiles_raw, dict):
        raise RuntimeError(
            "Profile definitions payload must contain mapping key 'profiles'."
        )
    profile_names: set[str] = set(profiles_raw.keys())
    expected_names: set[str] = set(_PROFILE_ORDER)
    missing = expected_names - profile_names
    unexpected = profile_names - expected_names
    if missing or unexpected:
        raise RuntimeError(
            "Profile definitions names mismatch. "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
        )

    parsed: dict[ProfileName, ProfileCatalogEntry] = {}
    for profile_name in _PROFILE_ORDER:
        parsed[profile_name] = _validate_catalog_entry(
            profile_name,
            profiles_raw[profile_name],
        )
    return MappingProxyType(parsed)


_PROFILE_CATALOG: MappingProxyType[ProfileName, ProfileCatalogEntry] = (
    _load_profile_catalog()
)
_PROFILE_MAP: MappingProxyType[str, RuntimeProfile] = MappingProxyType(
    {
        profile_name: RuntimeProfile(
            name=profile_name,
            description=entry.description,
        )
        for profile_name, entry in _PROFILE_CATALOG.items()
    }
)


def get_profile_catalog() -> Mapping[ProfileName, ProfileCatalogEntry]:
    """Returns immutable profile definition entries loaded from YAML."""
    return _PROFILE_CATALOG


def available_profiles() -> Mapping[str, RuntimeProfile]:
    """Returns immutable runtime profile definitions."""
    return _PROFILE_MAP


def resolve_profile_name(settings: AppConfig) -> ProfileName:
    """Resolves the active profile name from runtime flags."""
    runtime_flags = getattr(settings, "runtime_flags", None)
    if bool(getattr(runtime_flags, "accurate_research_profile", False)):
        return "accurate-research"
    if bool(getattr(runtime_flags, "accurate_profile", False)):
        return "accurate"
    return "medium" if bool(getattr(runtime_flags, "medium_profile", False)) else "fast"


def resolve_profile(settings: AppConfig) -> RuntimeProfile:
    """Resolves the full profile definition from runtime flags."""
    return _PROFILE_MAP[resolve_profile_name(settings)]
