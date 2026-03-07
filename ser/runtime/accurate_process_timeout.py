"""Process-timeout orchestration helpers for accurate runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, TypeVar

_WorkerMessageT = TypeVar("_WorkerMessageT")
_ResultT = TypeVar("_ResultT")


class AccurateProcessPayloadLike(Protocol):
    """Minimal payload contract used by process-timeout orchestration."""

    @property
    def expected_profile(self) -> str: ...


def run_with_process_timeout(
    *,
    payload: AccurateProcessPayloadLike,
    timeout_seconds: float,
    get_context: Callable[[str], object],
    logger: logging.Logger,
    setup_phase_name: str,
    inference_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    run_process_setup_compute_handshake: Callable[..., _WorkerMessageT],
    worker_target: Callable[..., None],
    recv_worker_message: Callable[..., _WorkerMessageT],
    is_setup_complete_message: Callable[[_WorkerMessageT], bool],
    terminate_worker_process: Callable[..., None],
    timeout_error_factory: Callable[[str], Exception] | type[Exception],
    execution_error_factory: Callable[[str], Exception] | type[Exception],
    worker_label: str,
    process_join_grace_seconds: float,
    parse_worker_completion_message: Callable[[_WorkerMessageT], _ResultT],
) -> _ResultT:
    """Runs one process-isolated attempt with timeout budget on compute phase."""
    setup_started_at = log_phase_started(
        logger,
        phase_name=setup_phase_name,
        profile=payload.expected_profile,
    )
    setup_completed = False
    inference_started_at: float | None = None
    context = get_context("spawn")

    def _on_setup_complete() -> None:
        nonlocal inference_started_at
        nonlocal setup_completed
        setup_completed = True
        log_phase_completed(
            logger,
            phase_name=setup_phase_name,
            started_at=setup_started_at,
            profile=payload.expected_profile,
        )
        inference_started_at = log_phase_started(
            logger,
            phase_name=inference_phase_name,
            profile=payload.expected_profile,
        )

    try:
        worker_message = run_process_setup_compute_handshake(
            context=context,
            worker_target=worker_target,
            payload=payload,
            timeout_seconds=timeout_seconds,
            recv_worker_message=recv_worker_message,
            is_setup_complete_message=is_setup_complete_message,
            terminate_worker_process=terminate_worker_process,
            timeout_error_factory=timeout_error_factory,
            execution_error_factory=execution_error_factory,
            worker_label=worker_label,
            process_join_grace_seconds=process_join_grace_seconds,
            on_setup_complete=_on_setup_complete,
        )
    except Exception:
        if inference_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=inference_phase_name,
                started_at=inference_started_at,
                profile=payload.expected_profile,
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=payload.expected_profile,
            )
        raise

    try:
        result = parse_worker_completion_message(worker_message)
    except Exception:
        if inference_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=inference_phase_name,
                started_at=inference_started_at,
                profile=payload.expected_profile,
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=payload.expected_profile,
            )
        raise
    if inference_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=inference_phase_name,
            started_at=inference_started_at,
            profile=payload.expected_profile,
        )
    return result


__all__ = ["run_with_process_timeout"]
