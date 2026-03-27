"""Runtime-profile resolution helpers for transcription backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


class _RuntimeProfile(Protocol):
    """Minimal profile contract required by runtime-request construction."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns backend identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Returns backend model name."""
        ...

    @property
    def use_demucs(self) -> bool:
        """Returns whether Demucs preprocessing is enabled."""
        ...

    @property
    def use_vad(self) -> bool:
        """Returns whether VAD preprocessing is enabled."""
        ...


class _RuntimePolicy(Protocol):
    """Runtime policy contract returned by runtime-policy resolution."""

    @property
    def device_spec(self) -> str:
        """Returns resolved device spec string."""
        ...

    @property
    def device_type(self) -> str:
        """Returns resolved device family."""
        ...

    @property
    def precision_candidates(self) -> tuple[str, ...]:
        """Returns ordered precision fallback candidates."""
        ...

    @property
    def memory_tier(self) -> str:
        """Returns runtime memory-tier classification."""
        ...


type _ErrorFactory = Callable[[str], Exception]
type _SettingsResolver = Callable[[], AppConfig]
type _BackendIdResolver = Callable[[object], TranscriptionBackendId]


class _RuntimePolicyResolver(Protocol):
    """Callable contract used to resolve one transcription runtime policy."""

    def __call__(
        self,
        *,
        backend_id: TranscriptionBackendId,
        requested_device: str,
        requested_dtype: str,
        mps_low_memory_threshold_gb: float,
    ) -> _RuntimePolicy:
        """Returns resolved runtime policy for one backend/runtime request."""
        ...


def resolve_backend_id(
    raw_backend_id: object,
    *,
    error_factory: _ErrorFactory,
) -> TranscriptionBackendId:
    """Normalizes one backend id and validates supported values."""
    if raw_backend_id in {"stable_whisper", "faster_whisper"}:
        return cast(TranscriptionBackendId, raw_backend_id)
    raise error_factory(
        "Unsupported transcription backend id configured. "
        "Expected 'stable_whisper' or 'faster_whisper'."
    )


def resolve_transcription_profile(
    profile: _RuntimeProfile | None = None,
    *,
    settings_resolver: _SettingsResolver,
    profile_factory: Callable[..., _RuntimeProfile],
    backend_id_resolver: _BackendIdResolver,
) -> _RuntimeProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    if profile is not None:
        return profile
    settings: AppConfig = settings_resolver()
    return profile_factory(
        backend_id=backend_id_resolver(settings.transcription.backend_id),
        model_name=settings.models.whisper_model.name,
        use_demucs=settings.transcription.use_demucs,
        use_vad=settings.transcription.use_vad,
    )


def runtime_request_from_profile(
    active_profile: _RuntimeProfile,
    settings: AppConfig,
    *,
    runtime_policy_resolver: _RuntimePolicyResolver,
    default_mps_low_memory_threshold_gb: float,
) -> BackendRuntimeRequest:
    """Builds one backend runtime request from transcription profile settings."""
    from ser.transcript.backends.base import BackendRuntimeRequest

    torch_runtime = getattr(settings, "torch_runtime", None)
    transcription_settings = getattr(settings, "transcription", None)
    requested_device = getattr(torch_runtime, "device", "cpu")
    requested_dtype = getattr(torch_runtime, "dtype", "auto")
    requested_mps_low_memory_threshold = getattr(
        transcription_settings,
        "mps_low_memory_threshold_gb",
        default_mps_low_memory_threshold_gb,
    )
    runtime_policy = runtime_policy_resolver(
        backend_id=active_profile.backend_id,
        requested_device=(requested_device if isinstance(requested_device, str) else "cpu"),
        requested_dtype=requested_dtype if isinstance(requested_dtype, str) else "auto",
        mps_low_memory_threshold_gb=(
            requested_mps_low_memory_threshold
            if (
                isinstance(requested_mps_low_memory_threshold, int | float)
                and not isinstance(requested_mps_low_memory_threshold, bool)
            )
            else default_mps_low_memory_threshold_gb
        ),
    )
    return BackendRuntimeRequest(
        model_name=active_profile.model_name,
        use_demucs=active_profile.use_demucs,
        use_vad=active_profile.use_vad,
        device_spec=runtime_policy.device_spec,
        device_type=runtime_policy.device_type,
        precision_candidates=runtime_policy.precision_candidates,
        memory_tier=runtime_policy.memory_tier,
    )


__all__ = [
    "resolve_backend_id",
    "resolve_transcription_profile",
    "runtime_request_from_profile",
]
