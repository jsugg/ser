"""Shared retry/setup scaffolding for runtime inference operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

_PayloadT = TypeVar("_PayloadT")
_PreparedT = TypeVar("_PreparedT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


def prepare_retry_state(
    *,
    use_process_isolation: bool,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    build_process_payload: Callable[[], _PayloadT],
    prepare_in_process_operation: Callable[[], _PreparedT],
) -> tuple[_PayloadT | None, _PreparedT | None, float | None]:
    """Builds process payload or in-process prepared operation for one request."""
    if use_process_isolation:
        return build_process_payload(), None, None

    setup_started_at = log_phase_started(
        logger,
        phase_name=setup_phase_name,
        profile=profile,
    )
    try:
        prepared_operation = prepare_in_process_operation()
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    return None, prepared_operation, setup_started_at


def finalize_in_process_setup(
    *,
    use_process_isolation: bool,
    backend: _BackendT | None,
    setup_started_at: float | None,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    prepare_backend_runtime: Callable[[_BackendT], None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    missing_backend_message: str,
    runtime_error_factory: Callable[[str], Exception],
) -> None:
    """Finalizes in-process setup and emits setup phase diagnostics."""
    if use_process_isolation:
        return
    if backend is None:
        raise runtime_error_factory(missing_backend_message)
    try:
        prepare_backend_runtime(backend)
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    if setup_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=setup_phase_name,
            started_at=setup_started_at,
            profile=profile,
        )


def run_retryable_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    process_payload: _PayloadT | None,
    timeout_seconds: float,
    logger: logging.Logger,
    profile: str,
    phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    run_with_process_timeout: Callable[[_PayloadT], _ResultT],
    run_once_inprocess: Callable[[], _ResultT],
    run_with_timeout: Callable[..., _ResultT],
    timeout_error_factory: Callable[[str], Exception] | type[Exception],
    timeout_label: str,
    missing_process_payload_message: str,
    runtime_error_factory: Callable[[str], Exception],
) -> _ResultT:
    """Runs one retryable inference operation with shared phase orchestration."""
    if enforce_timeout:
        if use_process_isolation:
            if process_payload is None:
                raise runtime_error_factory(missing_process_payload_message)
            return run_with_process_timeout(process_payload)
        inference_started_at = log_phase_started(
            logger,
            phase_name=phase_name,
            profile=profile,
        )
        try:
            result = run_with_timeout(
                operation=run_once_inprocess,
                timeout_seconds=timeout_seconds,
                timeout_error_factory=timeout_error_factory,
                timeout_label=timeout_label,
            )
        except Exception:
            log_phase_failed(
                logger,
                phase_name=phase_name,
                started_at=inference_started_at,
                profile=profile,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=phase_name,
            started_at=inference_started_at,
            profile=profile,
        )
        return result

    inference_started_at = log_phase_started(
        logger,
        phase_name=phase_name,
        profile=profile,
    )
    try:
        result = run_once_inprocess()
    except Exception:
        log_phase_failed(
            logger,
            phase_name=phase_name,
            started_at=inference_started_at,
            profile=profile,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=phase_name,
        started_at=inference_started_at,
        profile=profile,
    )
    return result


__all__ = [
    "finalize_in_process_setup",
    "prepare_retry_state",
    "run_retryable_operation",
]
