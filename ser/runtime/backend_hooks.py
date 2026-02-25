"""Runtime backend hook registry for profile-specific inference routing."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Mapping
from typing import cast

from ser.config import AppConfig
from ser.license_check import (
    BackendConsentRecord,
    ensure_backend_access,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
)
from ser.runtime.contracts import BackendInferenceCallable, InferenceRequest
from ser.runtime.schema import InferenceResult

type MediumInferenceRunner = Callable[[InferenceRequest, AppConfig], InferenceResult]
type AccurateInferenceRunner = Callable[[InferenceRequest, AppConfig], InferenceResult]
type AccurateResearchInferenceRunner = Callable[
    [InferenceRequest, AppConfig], InferenceResult
]
type FastInferenceRunner = Callable[[InferenceRequest, AppConfig], InferenceResult]


def _missing_optional_modules(required_modules: tuple[str, ...]) -> tuple[str, ...]:
    """Returns missing optional modules required by a backend hook."""
    missing: list[str] = []
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return tuple(missing)


def _load_medium_inference_runner() -> MediumInferenceRunner | None:
    """Loads the concrete medium inference runner when implementation exists."""
    try:
        module = importlib.import_module("ser.runtime.medium_inference")
    except ModuleNotFoundError:
        return None

    runner = getattr(module, "run_medium_inference", None)
    if not callable(runner):
        return None
    return cast(MediumInferenceRunner, runner)


def _load_fast_inference_runner() -> FastInferenceRunner | None:
    """Loads the concrete fast inference runner when implementation exists."""
    try:
        module = importlib.import_module("ser.runtime.fast_inference")
    except ModuleNotFoundError:
        return None

    runner = getattr(module, "run_fast_inference", None)
    if not callable(runner):
        return None
    return cast(FastInferenceRunner, runner)


def _load_accurate_inference_runner() -> AccurateInferenceRunner | None:
    """Loads the concrete accurate inference runner when implementation exists."""
    try:
        module = importlib.import_module("ser.runtime.accurate_inference")
    except ModuleNotFoundError:
        return None

    runner = getattr(module, "run_accurate_inference", None)
    if not callable(runner):
        return None
    return cast(AccurateInferenceRunner, runner)


def _load_accurate_research_inference_runner() -> (
    AccurateResearchInferenceRunner | None
):
    """Loads the concrete accurate-research inference runner when available."""
    try:
        module = importlib.import_module("ser.runtime.accurate_research_inference")
    except ModuleNotFoundError:
        return None

    runner = getattr(module, "run_accurate_research_inference", None)
    if not callable(runner):
        return None
    return cast(AccurateResearchInferenceRunner, runner)


def _build_fast_hook(
    settings: AppConfig,
    *,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> BackendInferenceCallable | None:
    """Builds fast backend hook when implementation exists."""
    ensure_backend_access(
        backend_id="handcrafted",
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    runner = _load_fast_inference_runner()
    if runner is None:
        return None

    def fast_hook(request: InferenceRequest) -> InferenceResult:
        return runner(request, settings)

    return fast_hook


def _build_medium_hook(
    settings: AppConfig,
    *,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> BackendInferenceCallable | None:
    """Builds medium backend hook when flag, deps, and implementation are ready."""
    if not settings.runtime_flags.medium_profile:
        return None
    ensure_backend_access(
        backend_id="hf_xlsr",
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if _missing_optional_modules(("torch", "transformers")):
        return None

    runner = _load_medium_inference_runner()
    if runner is None:
        return None

    def medium_hook(request: InferenceRequest) -> InferenceResult:
        return runner(request, settings)

    return medium_hook


def _build_accurate_hook(
    settings: AppConfig,
    *,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> BackendInferenceCallable | None:
    """Builds accurate backend hook when flag, deps, and implementation are ready."""
    if not settings.runtime_flags.accurate_profile:
        return None
    ensure_backend_access(
        backend_id="hf_whisper",
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if _missing_optional_modules(("torch", "transformers")):
        return None

    runner = _load_accurate_inference_runner()
    if runner is None:
        return None

    def accurate_hook(request: InferenceRequest) -> InferenceResult:
        return runner(request, settings)

    return accurate_hook


def _build_accurate_research_hook(
    settings: AppConfig,
    *,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> BackendInferenceCallable | None:
    """Builds accurate-research hook when flag, policy, deps, and runner are ready."""
    if not settings.runtime_flags.accurate_research_profile:
        return None
    ensure_backend_access(
        backend_id="emotion2vec",
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if _missing_optional_modules(("torch", "transformers")):
        return None

    runner = _load_accurate_research_inference_runner()
    if runner is None:
        return None

    def accurate_research_hook(request: InferenceRequest) -> InferenceResult:
        return runner(request, settings)

    return accurate_research_hook


def build_backend_hooks(settings: AppConfig) -> dict[str, BackendInferenceCallable]:
    """Builds runtime backend hooks keyed by backend id."""
    hooks: dict[str, BackendInferenceCallable] = {}
    allowed_restricted_backends = parse_allowed_restricted_backends_env()
    persisted_consents = load_persisted_backend_consents(settings=settings)

    fast_hook = _build_fast_hook(
        settings,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if fast_hook is not None:
        hooks["handcrafted"] = fast_hook

    medium_hook = _build_medium_hook(
        settings,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if medium_hook is not None:
        hooks["hf_xlsr"] = medium_hook

    accurate_hook = _build_accurate_hook(
        settings,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if accurate_hook is not None:
        hooks["hf_whisper"] = accurate_hook

    accurate_research_hook = _build_accurate_research_hook(
        settings,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if accurate_research_hook is not None:
        hooks["emotion2vec"] = accurate_research_hook

    return hooks
