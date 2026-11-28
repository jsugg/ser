"""Internal fast-profile training entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from ser._internal.models.artifact_envelope import LoadedModel
from ser._internal.models.fast_training import (
    FastTrainingHooks,
    train_fast_model,
    validate_fast_feature_contract,
)
from ser._internal.runtime.environment_plan import ProcessEnvDelta
from ser.config import AppConfig


class RuntimeEnvironmentPlanLike(Protocol):
    """Runtime environment plan required by fast training entrypoints."""

    @property
    def torch_runtime(self) -> ProcessEnvDelta: ...


def load_fast_model_for_evaluation(settings: AppConfig) -> LoadedModel:
    """Load a trusted trained artifact only if its feature semantics match this evaluator."""
    from ser._internal.models.emotion_model import load_model

    loaded = load_model(
        settings=settings, expected_profile="fast", expected_backend_id="handcrafted"
    )
    validate_fast_feature_contract(loaded.artifact_metadata, settings.feature_flags)
    return loaded


def train_model(
    settings: AppConfig | None = None,
    *,
    settings_resolver: Callable[[], AppConfig],
    build_hooks: Callable[[AppConfig], FastTrainingHooks],
    build_runtime_environment_plan_fn: Callable[[AppConfig], RuntimeEnvironmentPlanLike],
    temporary_process_env_fn: Callable[[ProcessEnvDelta], AbstractContextManager[None]],
) -> None:
    """Runs the fast-profile training entrypoint with explicit runtime env setup."""

    active_settings = settings if settings is not None else settings_resolver()
    hooks = build_hooks(active_settings)
    runtime_environment = build_runtime_environment_plan_fn(active_settings)
    with temporary_process_env_fn(runtime_environment.torch_runtime):
        train_fast_model(hooks=hooks)
