"""Worker lifecycle adapters for process-isolated medium inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import TypeVar

from ser._internal.runtime.worker_bindings import (
    is_setup_complete_message as _is_setup_complete_message_binding,
)
from ser._internal.runtime.worker_bindings import (
    parse_worker_completion_message as _parse_worker_completion_message_binding,
)
from ser._internal.runtime.worker_bindings import raise_worker_error as _raise_worker_error_binding
from ser._internal.runtime.worker_bindings import (
    recv_worker_message as _recv_worker_message_binding,
)
from ser._internal.runtime.worker_bindings import run_worker_entry as _run_worker_entry_binding
from ser._internal.runtime.worker_bindings import (
    terminate_worker_process as _terminate_worker_process_binding,
)
from ser.runtime import medium_process_timeout as medium_process_timeout_helpers

_PayloadT = TypeVar("_PayloadT")
_MessageT = TypeVar("_MessageT", bound=tuple[object, ...])
_PreparedOperationT = TypeVar("_PreparedOperationT")
_ResultT = TypeVar("_ResultT")


def run_with_process_timeout(
    payload: object,
    *,
    profile: str,
    timeout_seconds: float,
    get_context: Callable[[str], object],
    logger: logging.Logger,
    setup_phase_name: str,
    inference_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    run_process_setup_compute_handshake: Callable[..., _MessageT],
    worker_target: Callable[..., None],
    recv_worker_message: Callable[..., _MessageT],
    is_setup_complete_message: Callable[[_MessageT], bool],
    terminate_worker_process: Callable[..., None],
    timeout_error_factory: Callable[[str], Exception] | type[Exception],
    execution_error_factory: Callable[[str], Exception] | type[Exception],
    worker_label: str,
    process_join_grace_seconds: float,
    parse_worker_completion_message: Callable[[_MessageT], _ResultT],
) -> _ResultT:
    """Runs one isolated medium attempt with setup-aware timeout handling."""
    return medium_process_timeout_helpers.run_with_process_timeout(
        payload=payload,
        profile=profile,
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


def recv_worker_message(
    connection: Connection,
    *,
    stage: str,
    impl: Callable[..., _MessageT],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
) -> _MessageT:
    """Receives one worker message with medium-specific error mapping."""
    return _recv_worker_message_binding(
        connection=connection,
        stage=stage,
        impl=impl,
        worker_label=worker_label,
        error_factory=error_factory,
    )


def is_setup_complete_message(
    message: tuple[object, ...],
    *,
    impl: Callable[..., bool],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
) -> bool:
    """Checks whether one worker message marks setup completion."""
    return _is_setup_complete_message_binding(
        message=message,
        impl=impl,
        worker_label=worker_label,
        error_factory=error_factory,
    )


def parse_worker_completion_message(
    worker_message: tuple[object, ...],
    *,
    impl: Callable[..., _ResultT],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
    raise_worker_error: Callable[[str, str], None],
    result_type: type[_ResultT],
) -> _ResultT:
    """Parses one worker completion message for medium inference."""
    return _parse_worker_completion_message_binding(
        worker_message=worker_message,
        impl=impl,
        worker_label=worker_label,
        error_factory=error_factory,
        raise_worker_error=raise_worker_error,
        result_type=result_type,
    )


def run_worker_entry(
    payload: _PayloadT,
    connection: Connection,
    *,
    prepare_process_operation: Callable[[_PayloadT], _PreparedOperationT],
    run_process_operation: Callable[[_PreparedOperationT], object],
) -> None:
    """Executes one medium inference operation inside a child process."""
    _run_worker_entry_binding(
        payload=payload,
        connection=connection,
        prepare_process_operation=prepare_process_operation,
        run_process_operation=run_process_operation,
    )


def terminate_worker_process(
    process: BaseProcess,
    *,
    impl: Callable[..., None],
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Terminates a timed-out worker process with kill fallback."""
    _terminate_worker_process_binding(
        process=process,
        impl=impl,
        terminate_grace_seconds=terminate_grace_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )


def raise_worker_error(
    error_type: str,
    message: str,
    *,
    impl: Callable[..., None],
    known_error_factories: dict[str, Callable[[str], Exception]],
    unknown_error_factory: Callable[[str], Exception] | type[Exception],
    worker_label: str,
) -> None:
    """Rehydrates medium worker failures into runtime-domain exceptions."""
    _raise_worker_error_binding(
        error_type=error_type,
        message=message,
        impl=impl,
        known_error_factories=known_error_factories,
        unknown_error_factory=unknown_error_factory,
        worker_label=worker_label,
    )


__all__ = [
    "is_setup_complete_message",
    "parse_worker_completion_message",
    "raise_worker_error",
    "recv_worker_message",
    "run_with_process_timeout",
    "run_worker_entry",
    "terminate_worker_process",
]
