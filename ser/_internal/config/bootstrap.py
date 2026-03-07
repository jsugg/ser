"""Configuration bootstrap, environment resolution, and scoped settings state."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Literal, cast

from ser._internal.config import runtime_environment as runtime_environment_helpers
from ser._internal.config.schema import (
    AppConfig,
    ArtifactProfileName,
    _artifact_profile_from_runtime_flags,
    _default_cache_root,
    _default_data_root,
    default_profile_model_id,
    profile_artifact_file_names,
)
from ser._internal.config.settings_inputs import (
    ResolvedSettingsInputs,
    SettingsInputDeps,
)
from ser._internal.config.settings_inputs import (
    resolve_settings_inputs as _resolve_settings_inputs_from_internal,
)
from ser.profiles import (
    ProfileCatalogEntry,
    TranscriptionBackendId,
    get_profile_catalog,
)
from ser.utils.torch_inference import normalize_torch_device_selector

_SER_TORCH_ENABLE_MPS_FALLBACK_ENV = "SER_TORCH_ENABLE_MPS_FALLBACK"
_PYTORCH_ENABLE_MPS_FALLBACK_ENV = "PYTORCH_ENABLE_MPS_FALLBACK"


def _read_path_env(name: str, default: Path) -> Path:
    """Reads a path env var and returns an expanded path."""
    raw_value: str | None = os.getenv(name)
    selected: Path = Path(raw_value) if raw_value else default
    return selected.expanduser()


def _parse_manifest_paths(raw_value: str) -> tuple[Path, ...]:
    """Parses comma-separated dataset manifest paths."""
    if not raw_value:
        return ()
    return tuple(
        Path(item.strip()).expanduser() for item in raw_value.split(",") if item.strip()
    )


def _resolve_optional_path(raw_value: str) -> Path | None:
    """Returns an expanded path when set, otherwise None."""
    normalized = raw_value.strip()
    if not normalized:
        return None
    return Path(normalized).expanduser()


def _read_bool_env(name: str, default: bool) -> bool:
    """Reads a boolean env var and returns a validated value."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    normalized: str = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return False if normalized in {"0", "false", "no", "off"} else default


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    """Reads an integer env var and enforces an optional lower bound."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        return default

    return default if minimum is not None and parsed < minimum else parsed


def _read_float_env(
    name: str,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Reads a float env var and enforces optional bounds."""
    raw_value: str | None = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed: float = float(raw_value)
    except ValueError:
        return default

    if minimum is not None and parsed < minimum:
        return default
    return default if maximum is not None and parsed > maximum else parsed


def _read_bool_env_optional(name: str | None, default: bool) -> bool:
    """Reads a nullable boolean env var and returns a validated value."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_bool_env(name, default)


def _read_int_env_optional(
    name: str | None,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """Reads a nullable integer env var and enforces an optional lower bound."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_int_env(name, default, minimum=minimum)


def _read_float_env_optional(
    name: str | None,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Reads a nullable float env var and enforces optional bounds."""
    if not isinstance(name, str) or not name.strip():
        return default
    return _read_float_env(name, default, minimum=minimum, maximum=maximum)


def _resolve_profile_enabled(entry: ProfileCatalogEntry) -> bool:
    """Returns whether one profile is enabled after env/default resolution."""
    return _read_bool_env_optional(entry.enable_flag, entry.enabled_by_default)


def _resolve_profile_model_id(entry: ProfileCatalogEntry) -> str:
    """Resolves one profile model id from definition defaults and environment."""
    default_model_id = default_profile_model_id(entry.name)
    env_var = entry.model.env_var
    if not isinstance(env_var, str) or not env_var.strip():
        return default_model_id
    resolved = os.getenv(env_var, default_model_id).strip()
    return resolved if resolved else default_model_id


def _read_torch_device_env(name: str, default: str = "auto") -> str:
    """Reads and validates torch device selector from environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = normalize_torch_device_selector(raw_value)
    if normalized is not None:
        return normalized
    return default


def _read_torch_dtype_env(name: str, default: str = "auto") -> str:
    """Reads and validates torch dtype selector from environment."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    return (
        normalized
        if normalized in {"auto", "float32", "float16", "bfloat16"}
        else default
    )


def _read_confidence_level_env(
    name: str,
    default: Literal["high", "medium", "low"],
) -> Literal["high", "medium", "low"]:
    """Reads one confidence level env var with strict validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"high", "medium", "low"}:
        return cast(Literal["high", "medium", "low"], normalized)
    return default


def resolve_profile_transcription_config(
    profile: ArtifactProfileName,
) -> tuple[TranscriptionBackendId, str, bool, bool]:
    """Resolves transcription defaults for one profile with env overrides."""
    entry = get_profile_catalog()[profile]
    defaults = entry.transcription_defaults
    env_keys = entry.transcription_env
    backend_override = (
        os.getenv(env_keys.backend_id)
        if isinstance(env_keys.backend_id, str) and env_keys.backend_id.strip()
        else None
    )
    backend_id_raw = (
        defaults.backend_id
        if backend_override is None
        else backend_override.strip().lower()
    )
    if backend_id_raw in {"stable_whisper", "faster_whisper"}:
        backend_id = cast(TranscriptionBackendId, backend_id_raw)
    else:
        backend_id = defaults.backend_id
    model_name = (
        os.getenv(env_keys.model_name, defaults.model_name)
        if isinstance(env_keys.model_name, str) and env_keys.model_name.strip()
        else defaults.model_name
    ).strip()
    if not model_name:
        model_name = defaults.model_name
    use_demucs = _read_bool_env_optional(env_keys.use_demucs, defaults.use_demucs)
    use_vad = _read_bool_env_optional(env_keys.use_vad, defaults.use_vad)
    return backend_id, model_name, use_demucs, use_vad


def _resolve_settings_inputs() -> ResolvedSettingsInputs:
    """Resolves environment/profile values into immutable builder inputs."""
    deps = SettingsInputDeps(
        getenv=os.getenv,
        cpu_count=os.cpu_count,
        parse_manifest_paths=_parse_manifest_paths,
        resolve_optional_path=_resolve_optional_path,
        read_path_env=_read_path_env,
        read_bool_env=_read_bool_env,
        read_int_env=_read_int_env,
        read_float_env=_read_float_env,
        read_bool_env_optional=_read_bool_env_optional,
        read_int_env_optional=_read_int_env_optional,
        read_float_env_optional=_read_float_env_optional,
        read_torch_device_env=_read_torch_device_env,
        read_torch_dtype_env=_read_torch_dtype_env,
        read_confidence_level_env=_read_confidence_level_env,
        get_profile_catalog=get_profile_catalog,
        resolve_profile_enabled=_resolve_profile_enabled,
        resolve_profile_model_id=_resolve_profile_model_id,
        artifact_profile_from_runtime_flags=_artifact_profile_from_runtime_flags,
        resolve_profile_transcription_config=resolve_profile_transcription_config,
        profile_artifact_file_names=profile_artifact_file_names,
        default_cache_root=_default_cache_root,
        default_data_root=_default_data_root,
        ser_torch_enable_mps_fallback_env=_SER_TORCH_ENABLE_MPS_FALLBACK_ENV,
        pytorch_enable_mps_fallback_env=_PYTORCH_ENABLE_MPS_FALLBACK_ENV,
    )
    return _resolve_settings_inputs_from_internal(deps)


def _build_settings() -> AppConfig:
    """Builds settings from process environment."""
    from ser._internal.config.settings_builder import build_settings_from_inputs

    return build_settings_from_inputs(_resolve_settings_inputs())


_SETTINGS: AppConfig = _build_settings()
_SETTINGS_OVERRIDE: ContextVar[AppConfig | None] = ContextVar(
    "ser_settings_override",
    default=None,
)


def get_settings() -> AppConfig:
    """Returns current immutable settings."""
    override_settings = _SETTINGS_OVERRIDE.get()
    return _SETTINGS if override_settings is None else override_settings


@contextmanager
def settings_override(settings: AppConfig) -> Iterator[AppConfig]:
    """Temporarily overrides active settings in the current execution context."""
    token = _SETTINGS_OVERRIDE.set(settings)
    runtime_environment_helpers.sync_torch_runtime_environment(
        enable_mps_fallback=settings.torch_runtime.enable_mps_fallback,
        environ=os.environ,
        pytorch_enable_mps_fallback_env=_PYTORCH_ENABLE_MPS_FALLBACK_ENV,
    )
    try:
        yield settings
    finally:
        _SETTINGS_OVERRIDE.reset(token)
        restored_settings = get_settings()
        runtime_environment_helpers.sync_torch_runtime_environment(
            enable_mps_fallback=restored_settings.torch_runtime.enable_mps_fallback,
            environ=os.environ,
            pytorch_enable_mps_fallback_env=_PYTORCH_ENABLE_MPS_FALLBACK_ENV,
        )


def apply_settings(settings: AppConfig) -> AppConfig:
    """Applies an explicit settings snapshot as the active process settings."""
    _SETTINGS_OVERRIDE.set(settings)
    runtime_environment_helpers.sync_torch_runtime_environment(
        enable_mps_fallback=settings.torch_runtime.enable_mps_fallback,
        environ=os.environ,
        pytorch_enable_mps_fallback_env=_PYTORCH_ENABLE_MPS_FALLBACK_ENV,
    )
    return settings


def reload_settings() -> AppConfig:
    """Reloads settings from environment and returns the new snapshot."""
    global _SETTINGS
    _SETTINGS = _build_settings()
    _SETTINGS_OVERRIDE.set(None)
    runtime_environment_helpers.sync_torch_runtime_environment(
        enable_mps_fallback=_SETTINGS.torch_runtime.enable_mps_fallback,
        environ=os.environ,
        pytorch_enable_mps_fallback_env=_PYTORCH_ENABLE_MPS_FALLBACK_ENV,
    )
    return _SETTINGS


__all__ = [
    "_build_settings",
    "_resolve_settings_inputs",
    "apply_settings",
    "get_settings",
    "reload_settings",
    "resolve_profile_transcription_config",
    "settings_override",
]
