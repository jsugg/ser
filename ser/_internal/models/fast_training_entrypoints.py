"""Internal fast-profile training entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol

from ser._internal.runtime.environment_plan import ProcessEnvDelta
from ser.config import AppConfig
from ser.models.fast_training import FastTrainingHooks, train_fast_model


class RuntimeEnvironmentPlanLike(Protocol):
    """Runtime environment plan required by fast training entrypoints."""

    @property
    def torch_runtime(self) -> ProcessEnvDelta: ...


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
