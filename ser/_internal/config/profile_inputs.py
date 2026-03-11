"""Profile-derived immutable config input helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from ser.profiles import ProfileCatalogEntry

type ArtifactProfileName = Literal["fast", "medium", "accurate", "accurate-research"]
type ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class RuntimeProfileSettingsInput:
    """Resolved runtime defaults for one profile."""

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


@dataclass(frozen=True, slots=True)
class FeatureRuntimeOverrideInput:
    """Resolved feature-runtime selector override for one backend."""

    backend_id: str
    device: str | None
    dtype: str | None


class ReadBoolEnvOptional(Protocol):
    """Typed callable contract for optional boolean env parsing."""

    def __call__(self, name: str | None, default: bool) -> bool: ...


class ReadIntEnvOptional(Protocol):
    """Typed callable contract for optional integer env parsing."""

    def __call__(
        self,
        name: str | None,
        default: int,
        *,
        minimum: int | None = None,
    ) -> int: ...


class ReadFloatEnvOptional(Protocol):
    """Typed callable contract for optional float env parsing."""

    def __call__(
        self,
        name: str | None,
        default: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float: ...


def build_feature_runtime_overrides(
    profile_catalog: Mapping[ArtifactProfileName, ProfileCatalogEntry],
) -> tuple[FeatureRuntimeOverrideInput, ...]:
    """Builds flattened runtime-selector overrides from profile defaults."""
    overrides: dict[str, FeatureRuntimeOverrideInput] = {}
    source_profiles: dict[str, ArtifactProfileName] = {}
    for profile_name, entry in profile_catalog.items():
        defaults = entry.feature_runtime_defaults
        if defaults is None:
            continue
        backend_id = entry.backend_id.strip().lower()
        candidate = FeatureRuntimeOverrideInput(
            backend_id=backend_id,
            device=defaults.torch_device,
            dtype=defaults.torch_dtype,
        )
        existing = overrides.get(backend_id)
        if existing is not None and existing != candidate:
            previous_profile = source_profiles[backend_id]
            raise RuntimeError(
                "Profile definitions contain conflicting feature runtime defaults for "
                f"backend_id={backend_id!r} across profiles "
                f"{previous_profile!r} and {profile_name!r}."
            )
        overrides[backend_id] = candidate
        source_profiles[backend_id] = profile_name
    return tuple(overrides[backend_id] for backend_id in sorted(overrides))


def resolve_runtime_profile_settings_input(
    entry: ProfileCatalogEntry,
    *,
    read_bool_env_optional: ReadBoolEnvOptional,
    read_int_env_optional: ReadIntEnvOptional,
    read_float_env_optional: ReadFloatEnvOptional,
) -> RuntimeProfileSettingsInput:
    """Resolves runtime-profile settings from one catalog entry and env overrides."""
    runtime_defaults = entry.runtime_defaults
    runtime_env = entry.runtime_env

    post_hysteresis_enter_confidence = read_float_env_optional(
        runtime_env.post_hysteresis_enter_confidence,
        runtime_defaults.post_hysteresis_enter_confidence,
        minimum=0.0,
        maximum=1.0,
    )
    post_hysteresis_exit_confidence = read_float_env_optional(
        runtime_env.post_hysteresis_exit_confidence,
        runtime_defaults.post_hysteresis_exit_confidence,
        minimum=0.0,
        maximum=1.0,
    )
    if post_hysteresis_enter_confidence < post_hysteresis_exit_confidence:
        post_hysteresis_enter_confidence = runtime_defaults.post_hysteresis_enter_confidence
        post_hysteresis_exit_confidence = runtime_defaults.post_hysteresis_exit_confidence

    return RuntimeProfileSettingsInput(
        timeout_seconds=read_float_env_optional(
            runtime_env.timeout_seconds,
            runtime_defaults.timeout_seconds,
            minimum=0.0,
        ),
        max_timeout_retries=read_int_env_optional(
            runtime_env.max_timeout_retries,
            runtime_defaults.max_timeout_retries,
            minimum=0,
        ),
        max_transient_retries=read_int_env_optional(
            runtime_env.max_transient_retries,
            runtime_defaults.max_transient_retries,
            minimum=0,
        ),
        retry_backoff_seconds=read_float_env_optional(
            runtime_env.retry_backoff_seconds,
            runtime_defaults.retry_backoff_seconds,
            minimum=0.0,
        ),
        pool_window_size_seconds=read_float_env_optional(
            runtime_env.pool_window_size_seconds,
            runtime_defaults.pool_window_size_seconds,
            minimum=0.05,
        ),
        pool_window_stride_seconds=read_float_env_optional(
            runtime_env.pool_window_stride_seconds,
            runtime_defaults.pool_window_stride_seconds,
            minimum=0.05,
        ),
        post_smoothing_window_frames=read_int_env_optional(
            runtime_env.post_smoothing_window_frames,
            runtime_defaults.post_smoothing_window_frames,
            minimum=1,
        ),
        post_hysteresis_enter_confidence=post_hysteresis_enter_confidence,
        post_hysteresis_exit_confidence=post_hysteresis_exit_confidence,
        post_min_segment_duration_seconds=read_float_env_optional(
            runtime_env.post_min_segment_duration_seconds,
            runtime_defaults.post_min_segment_duration_seconds,
            minimum=0.0,
        ),
        process_isolation=read_bool_env_optional(
            runtime_env.process_isolation,
            runtime_defaults.process_isolation,
        ),
    )
