"""Process-timeout orchestration helpers for medium runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from ser._internal.runtime.process_timeout import (
    run_with_process_timeout as _run_with_process_timeout_impl,
)

_WorkerMessageT = TypeVar("_WorkerMessageT")
_ResultT = TypeVar("_ResultT")


def run_with_process_timeout(
    *,
    payload: object,
    profile: str,
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
    return _run_with_process_timeout_impl(
        payload=payload,
        resolve_profile=lambda _payload: profile,
        timeout_seconds=timeout_seconds,
        get_context=get_context,
        logger=logger,
        setup_phase_name=setup_phase_name,
        inference_phase_name=inference_phase_name,
        log_phase_started=log_phase_started,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        run_process_setup_compute_handshake=run_process_setup_compute_handshake,
        worker_target=worker_target,
        recv_worker_message=recv_worker_message,
        is_setup_complete_message=is_setup_complete_message,
        terminate_worker_process=terminate_worker_process,
        timeout_error_factory=timeout_error_factory,
        execution_error_factory=execution_error_factory,
        worker_label=worker_label,
        process_join_grace_seconds=process_join_grace_seconds,
        parse_worker_completion_message=parse_worker_completion_message,
    )


__all__ = ["run_with_process_timeout"]
