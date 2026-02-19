"""Runtime registry for profile-to-backend capability resolution."""

from __future__ import annotations

from dataclasses import dataclass

from ser.config import AppConfig
from ser.profiles import ProfileName, resolve_profile_name


class UnsupportedProfileError(RuntimeError):
    """Raised when the selected runtime profile is not supported yet."""


@dataclass(frozen=True)
class RuntimeCapability:
    """Resolved runtime capability for a profile selection."""

    profile: ProfileName
    backend_id: str
    available: bool
    message: str | None = None


def resolve_runtime_capability(settings: AppConfig) -> RuntimeCapability:
    """Maps configured profile selection to runtime backend capability."""
    profile = resolve_profile_name(settings)
    if profile == "fast":
        return RuntimeCapability(
            profile="fast",
            backend_id="handcrafted",
            available=True,
        )
    if profile == "medium":
        return RuntimeCapability(
            profile="medium",
            backend_id="hf_xlsr",
            available=False,
            message=(
                "Runtime profile 'medium' is not implemented yet. "
                "Disable `SER_ENABLE_MEDIUM_PROFILE` or keep `fast` profile."
            ),
        )
    return RuntimeCapability(
        profile="accurate",
        backend_id="hf_whisper",
        available=False,
        message=(
            "Runtime profile 'accurate' is not implemented yet. "
            "Disable `SER_ENABLE_ACCURATE_PROFILE` or keep `fast` profile."
        ),
    )


def ensure_profile_supported(capability: RuntimeCapability) -> None:
    """Raises when the selected runtime capability is not yet available."""
    if capability.available:
        return
    raise UnsupportedProfileError(
        capability.message or "Selected runtime capability is unavailable."
    )
