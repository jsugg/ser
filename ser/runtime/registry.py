"""Runtime registry for profile-to-backend capability resolution."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from ser.config import AppConfig
from ser.profiles import (
    ProfileEnableFlag,
    ProfileName,
    get_profile_catalog,
    resolve_profile_name,
)


class UnsupportedProfileError(RuntimeError):
    """Raised when the selected runtime profile is not supported yet."""


@dataclass(frozen=True)
class RuntimeCapability:
    """Resolved runtime capability for a profile selection."""

    profile: ProfileName
    backend_id: str
    available: bool
    required_modules: tuple[str, ...] = ()
    missing_modules: tuple[str, ...] = ()
    implementation_ready: bool = True
    message: str | None = None


DEFAULT_IMPLEMENTED_BACKENDS: frozenset[str] = frozenset({"handcrafted"})


def _missing_optional_modules(required_modules: tuple[str, ...]) -> tuple[str, ...]:
    """Returns optional modules that are not import-resolvable in this environment."""
    return tuple(
        module_name
        for module_name in required_modules
        if importlib.util.find_spec(module_name) is None
    )


def _profile_requirements(
    profile: ProfileName,
) -> tuple[str, tuple[str, ...], ProfileEnableFlag | None]:
    """Returns backend id, optional dependencies, and controlling flag name."""
    entry = get_profile_catalog()[profile]
    return (entry.backend_id, entry.required_modules, entry.enable_flag)


def resolve_runtime_capability(
    settings: AppConfig,
    *,
    available_backend_hooks: frozenset[str] | None = None,
) -> RuntimeCapability:
    """Maps configured profile selection to runtime backend capability."""
    profile = resolve_profile_name(settings)
    backend_id, required_modules, profile_flag = _profile_requirements(profile)
    resolved_backend_hooks: frozenset[str] = (
        available_backend_hooks
        if available_backend_hooks is not None
        else DEFAULT_IMPLEMENTED_BACKENDS
    )
    implementation_ready: bool = backend_id in resolved_backend_hooks
    missing_modules: tuple[str, ...] = _missing_optional_modules(required_modules)
    available: bool = implementation_ready and not missing_modules
    if available:
        return RuntimeCapability(
            profile=profile,
            backend_id=backend_id,
            available=True,
            required_modules=required_modules,
            missing_modules=missing_modules,
            implementation_ready=implementation_ready,
        )

    if missing_modules:
        dependency_list: str = ", ".join(missing_modules)
        disable_hint = (
            f" Disable `{profile_flag}` or keep `fast` profile." if profile_flag else ""
        )
        return RuntimeCapability(
            profile=profile,
            backend_id=backend_id,
            available=False,
            required_modules=required_modules,
            missing_modules=missing_modules,
            implementation_ready=implementation_ready,
            message=(
                f"Runtime profile '{profile}' requires optional dependencies not "
                f"currently available: {dependency_list}."
                f"{disable_hint}"
            ),
        )

    disable_hint = (
        f" Disable `{profile_flag}` or keep `fast` profile." if profile_flag else ""
    )
    return RuntimeCapability(
        profile=profile,
        backend_id=backend_id,
        available=False,
        required_modules=required_modules,
        missing_modules=missing_modules,
        implementation_ready=implementation_ready,
        message=(
            f"Runtime profile '{profile}' dependencies are available, but the backend "
            "path is not implemented yet."
            f"{disable_hint}"
        ),
    )


def ensure_profile_supported(capability: RuntimeCapability) -> None:
    """Raises when the selected runtime capability is not yet available."""
    if capability.available:
        return
    raise UnsupportedProfileError(
        capability.message or "Selected runtime capability is unavailable."
    )
