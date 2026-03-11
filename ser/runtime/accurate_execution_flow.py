"""Lock-body execution orchestration for accurate runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TypeVar

from ser.config import AppConfig
from ser.runtime.accurate_worker_operation import (
    AccurateRetryOperationState,
    PreparedAccurateOperation,
)
from ser.runtime.accurate_worker_operation import _PayloadLike as _AccuratePayloadLike

_PayloadT = TypeVar("_PayloadT", bound=_AccuratePayloadLike)
_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


def run_accurate_retryable_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[_PayloadT, _BackendT],
    prepared_operation: PreparedAccurateOperation[_LoadedModelT, _BackendT] | None,
    timeout_seconds: float,
    expected_profile: str,
    logger: logging.Logger,
    inference_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    run_with_process_timeout: Callable[..., _ResultT],
    run_accurate_inference_once: Callable[..., _ResultT],
    run_with_timeout: Callable[..., _ResultT],
    run_inference_operation: Callable[..., _ResultT],
    timeout_error_factory: Callable[[str], Exception],
    runtime_error_factory: Callable[[str], Exception],
) -> _ResultT:
    """Runs one accurate inference attempt using the current retry state."""
    return run_inference_operation(
        enforce_timeout=enforce_timeout,
        use_process_isolation=use_process_isolation,
        process_payload=retry_state.process_payload,
        prepared_operation=prepared_operation,
        active_backend=retry_state.active_backend,
        timeout_seconds=timeout_seconds,
        expected_profile=expected_profile,
        logger=logger,
        inference_phase_name=inference_phase_name,
        log_phase_started=log_phase_started,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        run_with_process_timeout=partial(
            run_with_process_timeout,
            timeout_seconds=timeout_seconds,
        ),
        run_accurate_inference_once=run_accurate_inference_once,
        run_with_timeout=run_with_timeout,
        timeout_error_factory=timeout_error_factory,
        runtime_error_factory=runtime_error_factory,
    )


def execute_accurate_inference_with_retry(
    *,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[_PayloadT, _BackendT],
    prepared_operation: PreparedAccurateOperation[_LoadedModelT, _BackendT] | None,
    setup_started_at: float | None,
    settings: AppConfig,
    timeout_seconds: float,
    backend: _BackendT | None,
    expected_backend_id: str,
    expected_profile: str,
    allow_retries: bool,
    enforce_timeout: bool,
    cpu_backend_builder: Callable[[], _BackendT],
    logger: logging.Logger,
    setup_phase_name: str,
    finalize_in_process_setup: Callable[..., None],
    prepare_accurate_backend_runtime: Callable[[_BackendT], None],
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    build_transient_failure_handler: Callable[
        ...,
        Callable[[Exception, int, int], None],
    ],
    should_retry_on_cpu_after_transient_failure: Callable[[str], bool],
    summarize_transient_failure: Callable[[str], str],
    process_payload_cpu_fallback: Callable[[_PayloadT], _PayloadT],
    run_retry_policy: Callable[..., _ResultT],
    retry_delay_seconds: Callable[..., float],
    run_with_retry_policy: Callable[..., _ResultT],
    passthrough_error_types: tuple[type[BaseException], ...],
    run_accurate_retryable_operation: Callable[..., _ResultT],
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    transient_exhausted_error: Callable[[Exception], Exception],
    runtime_error_factory: Callable[[RuntimeError], Exception],
) -> _ResultT:
    """Finalizes setup and executes accurate inference under retry policy."""
    finalize_in_process_setup(
        use_process_isolation=use_process_isolation,
        state=retry_state,
        setup_started_at=setup_started_at,
        logger=logger,
        profile=expected_profile,
        setup_phase_name=setup_phase_name,
        prepare_accurate_backend_runtime=prepare_accurate_backend_runtime,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        runtime_error_factory=RuntimeError,
    )
    on_transient_failure = build_transient_failure_handler(
        state=retry_state,
        use_process_isolation=use_process_isolation,
        injected_backend=backend,
        expected_backend_id=expected_backend_id,
        policy_device=settings.torch_runtime.device,
        logger=logger,
        should_retry_on_cpu_after_transient_failure=(should_retry_on_cpu_after_transient_failure),
        summarize_transient_failure=summarize_transient_failure,
        process_payload_cpu_fallback=process_payload_cpu_fallback,
        in_process_cpu_backend_builder=cpu_backend_builder,
        prepare_accurate_backend_runtime=prepare_accurate_backend_runtime,
        runtime_error_factory=RuntimeError,
    )
    return run_retry_policy(
        operation=partial(
            run_accurate_retryable_operation,
            enforce_timeout=enforce_timeout,
            use_process_isolation=use_process_isolation,
            retry_state=retry_state,
            prepared_operation=prepared_operation,
            timeout_seconds=timeout_seconds,
            expected_profile=expected_profile,
        ),
        runtime_config=settings.accurate_runtime,
        allow_retries=allow_retries,
        profile_label="Accurate",
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


__all__ = [
    "execute_accurate_inference_with_retry",
    "run_accurate_retryable_operation",
]
