"""Retry-operation orchestration helpers for accurate runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from ser._internal.runtime.retry_scaffold import (
    run_retryable_operation as _run_retryable_operation_impl,
)

_PayloadT = TypeVar("_PayloadT")
_ResultT = TypeVar("_ResultT")


def run_accurate_retry_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    process_payload: _PayloadT | None,
    timeout_seconds: float,
    expected_profile: str,
    logger: logging.Logger,
    run_with_process_timeout: Callable[[_PayloadT], _ResultT],
    run_once_inprocess: Callable[[], _ResultT],
    run_with_timeout: Callable[..., _ResultT],
    timeout_error_factory: Callable[[str], Exception] | type[Exception],
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    phase_name: str,
) -> _ResultT:
    """Runs one accurate retry operation with timeout and phase logging."""
    return _run_retryable_operation_impl(
        enforce_timeout=enforce_timeout,
        use_process_isolation=use_process_isolation,
        process_payload=process_payload,
        timeout_seconds=timeout_seconds,
        logger=logger,
        profile=expected_profile,
        phase_name=phase_name,
        log_phase_started=log_phase_started,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        run_with_process_timeout=run_with_process_timeout,
        run_once_inprocess=run_once_inprocess,
        run_with_timeout=run_with_timeout,
        timeout_error_factory=timeout_error_factory,
        timeout_label="Accurate inference",
        missing_process_payload_message=(
            "Accurate process payload is missing for isolated execution."
        ),
        runtime_error_factory=RuntimeError,
    )


def run_accurate_inference_with_retry_policy(
    *,
    operation: Callable[[], _ResultT],
    runtime_config: object,
    allow_retries: bool,
    profile_label: str,
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    transient_exhausted_error: Callable[[Exception], Exception],
    retry_delay_seconds: Callable[..., float],
    logger: logging.Logger,
    on_transient_failure: Callable[[Exception, int, int], None],
    run_with_retry_policy: Callable[..., _ResultT],
    passthrough_error_types: tuple[type[BaseException], ...],
    runtime_error_factory: Callable[[RuntimeError], Exception],
) -> _ResultT:
    """Runs accurate retry-operation execution under retry policy handling."""

    try:
        return run_with_retry_policy(
            operation=operation,
            runtime_config=runtime_config,
            allow_retries=allow_retries,
            profile_label=profile_label,
            timeout_error_type=timeout_error_type,
            transient_error_type=transient_error_type,
            transient_exhausted_error=transient_exhausted_error,
            retry_delay_seconds=retry_delay_seconds,
            logger=logger,
            on_transient_failure=on_transient_failure,
        )
    except passthrough_error_types:
        raise
    except RuntimeError as err:
        raise runtime_error_factory(err) from err


__all__ = [
    "run_accurate_inference_with_retry_policy",
    "run_accurate_retry_operation",
]
