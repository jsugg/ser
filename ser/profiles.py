"""Runtime profile resolution for staged pipeline rollout."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ser.config import AppConfig

type ProfileName = Literal["fast", "medium", "accurate"]


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved runtime profile configuration.

    Attributes:
        name: Canonical profile identifier.
        description: Short profile intent summary.
    """

    name: ProfileName
    description: str


_PROFILE_MAP = MappingProxyType(
    {
        "fast": RuntimeProfile(
            name="fast",
            description="Current CPU-first handcrafted pipeline.",
        ),
        "medium": RuntimeProfile(
            name="medium",
            description="Future SSL medium-latency pipeline profile.",
        ),
        "accurate": RuntimeProfile(
            name="accurate",
            description="Future highest-accuracy profile path.",
        ),
    }
)


def available_profiles() -> Mapping[str, RuntimeProfile]:
    """Returns immutable runtime profile definitions."""
    return _PROFILE_MAP


def resolve_profile_name(settings: AppConfig) -> ProfileName:
    """Resolves the active profile name from runtime flags."""
    runtime_flags = settings.runtime_flags
    if runtime_flags.accurate_profile:
        return "accurate"
    if runtime_flags.medium_profile:
        return "medium"
    return "fast"


def resolve_profile(settings: AppConfig) -> RuntimeProfile:
    """Resolves the full profile definition from runtime flags."""
    return _PROFILE_MAP[resolve_profile_name(settings)]
