"""Lock-body execution orchestration for medium runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from ser.runtime.medium_execution_context import MediumExecutionContext
from ser.runtime.medium_worker_operation import (
    PreparedMediumOperation,
)
from ser.runtime.medium_worker_operation import _PayloadLike as _MediumPayloadLike

_PayloadT = TypeVar("_PayloadT", bound=_MediumPayloadLike)
_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


def execute_medium_inference_with_retry(
    *,
    execution_context: MediumExecutionContext[_PayloadT, _LoadedModelT, _BackendT],
    injected_backend: _BackendT | None,
    enforce_timeout: bool,
    allow_retries: bool,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    inference_phase_name: str,
    finalize_in_process_setup: Callable[..., None],
    prepare_medium_backend_runtime: Callable[[_BackendT], None],
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    run_inference_operation: Callable[..., _ResultT],
    run_with_process_timeout: Callable[[_PayloadT, float], _ResultT],
    run_process_operation: Callable[
        [PreparedMediumOperation[_LoadedModelT, _BackendT]],
        _ResultT,
    ],
    run_with_timeout: Callable[[Callable[[], _ResultT], float], _ResultT],
    build_transient_failure_handler: Callable[
        ...,
        Callable[[Exception, int, int], None],
    ],
    should_retry_on_cpu_after_transient_failure: Callable[[Exception], bool],
    summarize_transient_failure: Callable[[Exception], str],
    process_payload_cpu_fallback: Callable[[_PayloadT], _PayloadT],
    in_process_cpu_backend_builder: Callable[[], _BackendT],
    replace_prepared_backend: Callable[
        [PreparedMediumOperation[_LoadedModelT, _BackendT], _BackendT],
        PreparedMediumOperation[_LoadedModelT, _BackendT],
    ],
    run_retry_policy: Callable[..., _ResultT],
    retry_delay_seconds: Callable[..., float],
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    transient_exhausted_error: Callable[[Exception], Exception],
    run_with_retry_policy: Callable[..., _ResultT],
    passthrough_error_types: tuple[type[BaseException], ...],
    runtime_error_factory: Callable[[RuntimeError], Exception],
) -> _ResultT:
    """Finalizes setup and executes medium inference under retry policy."""
    retry_state = execution_context.retry_state
    finalize_in_process_setup(
        use_process_isolation=execution_context.use_process_isolation,
        state=retry_state,
        setup_started_at=execution_context.setup_started_at,
        logger=logger,
        profile=profile,
        setup_phase_name=setup_phase_name,
        prepare_medium_backend_runtime=prepare_medium_backend_runtime,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        runtime_error_factory=RuntimeError,
    )

    def operation() -> _ResultT:
        return run_inference_operation(
            enforce_timeout=enforce_timeout,
            use_process_isolation=execution_context.use_process_isolation,
            process_payload=retry_state.process_payload,
            prepared_operation=retry_state.prepared_operation,
            timeout_seconds=execution_context.runtime_config.timeout_seconds,
            logger=logger,
            profile=profile,
            inference_phase_name=inference_phase_name,
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_with_process_timeout=run_with_process_timeout,
            run_process_operation=run_process_operation,
            run_with_timeout=run_with_timeout,
            runtime_error_factory=RuntimeError,
        )

    on_transient_failure = build_transient_failure_handler(
        state=retry_state,
        use_process_isolation=execution_context.use_process_isolation,
        injected_backend=injected_backend,
        policy_device=execution_context.runtime_policy.device,
        logger=logger,
        should_retry_on_cpu_after_transient_failure=(should_retry_on_cpu_after_transient_failure),
        summarize_transient_failure=summarize_transient_failure,
        process_payload_cpu_fallback=process_payload_cpu_fallback,
        in_process_cpu_backend_builder=in_process_cpu_backend_builder,
        prepare_medium_backend_runtime=prepare_medium_backend_runtime,
        replace_prepared_backend=replace_prepared_backend,
        runtime_error_factory=RuntimeError,
    )

    return run_retry_policy(
        operation=operation,
        runtime_config=execution_context.runtime_config,
        allow_retries=allow_retries,
        profile_label=profile.capitalize(),
        timeout_error_type=timeout_error_type,
        transient_error_type=transient_error_type,
        transient_exhausted_error=transient_exhausted_error,
        retry_delay_seconds=retry_delay_seconds,
        logger=logger,
        on_transient_failure=on_transient_failure,
        run_with_retry_policy=run_with_retry_policy,
        passthrough_error_types=passthrough_error_types,
        runtime_error_factory=runtime_error_factory,
    )


__all__ = ["execute_medium_inference_with_retry"]
