"""Internal model-loading entrypoint orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from typing import Protocol, cast

from ser._internal.runtime.environment_plan import ProcessEnvDelta
from ser.config import AppConfig
from ser.models.artifact_envelope import LoadedModel
from ser.models.training_support import ModelCandidate

type ResolveModelFn = Callable[..., tuple[object, LoadedModel]]
type ResolveModelFromSettingsFn = Callable[..., tuple[ModelCandidate, LoadedModel]]


class RuntimeEnvironmentPlanLike(Protocol):
    """Runtime environment plan required by model loading entrypoints."""

    @property
    def torch_runtime(self) -> ProcessEnvDelta: ...


def load_model(
    settings: AppConfig | None = None,
    *,
    settings_resolver: Callable[[], AppConfig],
    resolve_model_factory: Callable[[AppConfig], ResolveModelFn],
    logger: logging.Logger,
    build_runtime_environment_plan_fn: Callable[[AppConfig], RuntimeEnvironmentPlanLike],
    temporary_process_env_fn: Callable[[ProcessEnvDelta], AbstractContextManager[None]],
    load_model_with_resolution_fn: Callable[..., LoadedModel],
    expected_backend_id: str | None = None,
    expected_profile: str | None = None,
    expected_backend_model_id: str | None = None,
) -> LoadedModel:
    """Loads one serialized model with explicit runtime env setup."""

    active_settings = settings if settings is not None else settings_resolver()
    runtime_environment = build_runtime_environment_plan_fn(active_settings)
    with temporary_process_env_fn(runtime_environment.torch_runtime):
        return load_model_with_resolution_fn(
            settings=active_settings,
            settings_resolver=settings_resolver,
            resolve_model=resolve_model_factory(active_settings),
            logger=logger,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=expected_backend_model_id,
        )


def resolve_model_for_loading_from_public_boundary(
    settings: AppConfig,
    *,
    resolve_model_for_loading_from_settings_fn: ResolveModelFromSettingsFn,
    load_secure_model_for_settings_fn: Callable[[ModelCandidate, AppConfig], LoadedModel],
    load_pickle_model_fn: Callable[[ModelCandidate], LoadedModel],
    logger: logging.Logger,
) -> ResolveModelFn:
    """Builds the public-boundary model resolver from injected collaborators."""

    return cast(
        ResolveModelFn,
        partial(
            resolve_model_for_loading_from_settings_fn,
            folder=settings.models.folder,
            secure_model_file=settings.models.secure_model_file,
            model_file=settings.models.model_file,
            candidate_factory=ModelCandidate,
            load_secure_model_for_settings=load_secure_model_for_settings_fn,
            load_pickle_model=load_pickle_model_fn,
            logger=logger,
        ),
    )
