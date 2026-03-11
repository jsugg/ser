"""Shared adapter helpers for runtime worker lifecycle bindings."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, TypeVar

type WorkerMessage = tuple[Any, ...]
_ResultT = TypeVar("_ResultT")
_PayloadT = TypeVar("_PayloadT")
_PreparedT = TypeVar("_PreparedT")


def recv_worker_message(
    *,
    connection: Connection,
    stage: str,
    impl: Callable[..., WorkerMessage],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
) -> WorkerMessage:
    """Delegates worker-message reads with stable label/error wiring."""
    return impl(
        connection=connection,
        stage=stage,
        worker_label=worker_label,
        error_factory=error_factory,
    )


def is_setup_complete_message(
    *,
    message: WorkerMessage,
    impl: Callable[..., bool],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
) -> bool:
    """Delegates setup-complete message validation with shared wiring."""
    return impl(
        message=message,
        worker_label=worker_label,
        error_factory=error_factory,
    )


def parse_worker_completion_message(
    *,
    worker_message: WorkerMessage,
    impl: Callable[..., _ResultT],
    worker_label: str,
    error_factory: Callable[[str], Exception] | type[Exception],
    raise_worker_error: Callable[[str, str], None],
    result_type: type[_ResultT],
) -> _ResultT:
    """Delegates worker completion parsing with shared wiring."""
    return impl(
        worker_message=worker_message,
        worker_label=worker_label,
        error_factory=error_factory,
        raise_worker_error=raise_worker_error,
        result_type=result_type,
    )


def run_worker_entry(
    *,
    payload: _PayloadT,
    connection: Connection,
    prepare_process_operation: Callable[[_PayloadT], _PreparedT],
    run_process_operation: Callable[[_PreparedT], _ResultT],
) -> None:
    """Executes one process worker using the standard phase/ok/err envelope."""
    try:
        prepared_operation = prepare_process_operation(payload)
        connection.send(("phase", "setup_complete"))
        result = run_process_operation(prepared_operation)
        connection.send(("ok", result))
    except BaseException as err:
        connection.send(("err", type(err).__name__, str(err)))
    finally:
        connection.close()


def terminate_worker_process(
    *,
    process: BaseProcess,
    impl: Callable[..., None],
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Delegates worker termination with shared grace-period wiring."""
    impl(
        process=process,
        terminate_grace_seconds=terminate_grace_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )


def raise_worker_error(
    *,
    error_type: str,
    message: str,
    impl: Callable[..., None],
    known_error_factories: dict[str, Callable[[str], Exception]],
    unknown_error_factory: Callable[[str], Exception] | type[Exception],
    worker_label: str,
) -> None:
    """Delegates worker error hydration with shared mapping wiring."""
    impl(
        error_type=error_type,
        message=message,
        known_error_factories=known_error_factories,
        unknown_error_factory=unknown_error_factory,
        worker_label=worker_label,
    )


__all__ = [
    "is_setup_complete_message",
    "parse_worker_completion_message",
    "raise_worker_error",
    "recv_worker_message",
    "run_worker_entry",
    "terminate_worker_process",
]
