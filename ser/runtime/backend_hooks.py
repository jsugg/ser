"""Runtime backend hook registry for profile-specific inference routing."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable
from typing import cast

from ser.config import AppConfig
from ser.runtime.contracts import BackendInferenceCallable, InferenceRequest
from ser.runtime.schema import InferenceResult

type MediumInferenceRunner = Callable[[InferenceRequest, AppConfig], InferenceResult]


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


def _build_medium_hook(settings: AppConfig) -> BackendInferenceCallable | None:
    """Builds medium backend hook when flag, deps, and implementation are ready."""
    if not settings.runtime_flags.medium_profile:
        return None
    if _missing_optional_modules(("torch", "transformers")):
        return None

    runner = _load_medium_inference_runner()
    if runner is None:
        return None

    def medium_hook(request: InferenceRequest) -> InferenceResult:
        return runner(request, settings)

    return medium_hook


def build_backend_hooks(settings: AppConfig) -> dict[str, BackendInferenceCallable]:
    """Builds runtime backend hooks keyed by backend id."""
    hooks: dict[str, BackendInferenceCallable] = {}

    medium_hook = _build_medium_hook(settings)
    if medium_hook is not None:
        hooks["hf_xlsr"] = medium_hook

    return hooks
