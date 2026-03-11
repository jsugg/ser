"""Pre-lock execution-context preparation for medium runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ser.config import AppConfig, MediumRuntimeConfig
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_worker_operation import (
    MediumRetryOperationState,
    PreparedMediumOperation,
)
from ser.runtime.medium_worker_operation import _PayloadLike as _MediumPayloadLike

_PayloadT = TypeVar("_PayloadT", bound=_MediumPayloadLike)
_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")


@dataclass(frozen=True, slots=True)
class MediumExecutionContext(Generic[_PayloadT, _LoadedModelT, _BackendT]):
    """Resolved execution context required before entering the single-flight lock."""

    runtime_config: MediumRuntimeConfig
    expected_backend_model_id: str
    runtime_policy: FeatureRuntimePolicy
    use_process_isolation: bool
    retry_state: MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT]
    setup_started_at: float | None


def prepare_execution_context(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    enforce_timeout: bool,
    resolve_medium_model_id: Callable[[AppConfig], str],
    resolve_runtime_policy: Callable[[AppConfig], FeatureRuntimePolicy],
    log_selector_adjustment: Callable[[str, str, str], None],
    prepare_retry_state: Callable[
        ...,
        tuple[
            MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT],
            float | None,
        ],
    ],
    build_process_payload: Callable[[str, str, str], _PayloadT],
    prepare_in_process_operation: Callable[
        ...,
        PreparedMediumOperation[_LoadedModelT, _BackendT],
    ],
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
) -> MediumExecutionContext[_PayloadT, _LoadedModelT, _BackendT]:
    """Resolves runtime selectors and prepares retry state for medium inference."""

    runtime_config = settings.medium_runtime
    expected_backend_model_id = resolve_medium_model_id(settings)
    runtime_policy = resolve_runtime_policy(settings)
    if (
        runtime_policy.device != settings.torch_runtime.device
        or runtime_policy.dtype != settings.torch_runtime.dtype
    ):
        log_selector_adjustment(
            runtime_policy.device,
            runtime_policy.dtype,
            runtime_policy.reason,
        )

    use_process_isolation = (
        enforce_timeout
        and loaded_model is None
        and backend is None
        and settings.runtime_flags.profile_pipeline
        and runtime_config.process_isolation
    )
    retry_state, setup_started_at = prepare_retry_state(
        use_process_isolation=use_process_isolation,
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        expected_backend_model_id=expected_backend_model_id,
        policy_device=runtime_policy.device,
        policy_dtype=runtime_policy.dtype,
        logger=logger,
        profile=profile,
        setup_phase_name=setup_phase_name,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
        build_process_payload=lambda: build_process_payload(
            expected_backend_model_id,
            runtime_policy.device,
            runtime_policy.dtype,
        ),
        prepare_in_process_operation=prepare_in_process_operation,
    )
    return MediumExecutionContext(
        runtime_config=runtime_config,
        expected_backend_model_id=expected_backend_model_id,
        runtime_policy=runtime_policy,
        use_process_isolation=use_process_isolation,
        retry_state=retry_state,
        setup_started_at=setup_started_at,
    )


__all__ = ["MediumExecutionContext", "prepare_execution_context"]
