"""Retry-operation orchestration helpers for accurate runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

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
    if enforce_timeout:
        if use_process_isolation:
            if process_payload is None:
                raise RuntimeError(
                    "Accurate process payload is missing for isolated execution."
                )
            return run_with_process_timeout(process_payload)
        inference_started_at = log_phase_started(
            logger,
            phase_name=phase_name,
            profile=expected_profile,
        )
        try:
            result = run_with_timeout(
                operation=run_once_inprocess,
                timeout_seconds=timeout_seconds,
                timeout_error_factory=timeout_error_factory,
                timeout_label="Accurate inference",
            )
        except Exception:
            log_phase_failed(
                logger,
                phase_name=phase_name,
                started_at=inference_started_at,
                profile=expected_profile,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=phase_name,
            started_at=inference_started_at,
            profile=expected_profile,
        )
        return result

    inference_started_at = log_phase_started(
        logger,
        phase_name=phase_name,
        profile=expected_profile,
    )
    try:
        result = run_once_inprocess()
    except Exception:
        log_phase_failed(
            logger,
            phase_name=phase_name,
            started_at=inference_started_at,
            profile=expected_profile,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=phase_name,
        started_at=inference_started_at,
        profile=expected_profile,
    )
    return result


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
