"""Shared worker message/lifecycle helpers for runtime process isolation."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Never, Protocol, TypeVar, cast

type _ErrorFactory = Callable[[str], Exception]
type _WorkerErrorRaiser = Callable[[str, str], None]
_TResult = TypeVar("_TResult")
_TPayload = TypeVar("_TPayload")
_TWorkerMessage = TypeVar(
    "_TWorkerMessage",
    bound=tuple[Any, ...],
    covariant=True,
)


class _WorkerMessageReader(Protocol[_TWorkerMessage]):
    """Callable protocol for worker message readers with stage annotation."""

    def __call__(
        self,
        connection: Connection,
        *,
        stage: str,
    ) -> _TWorkerMessage: ...


def recv_worker_message(
    *,
    connection: Connection,
    stage: str,
    worker_label: str,
    error_factory: _ErrorFactory,
) -> tuple[Any, ...]:
    """Receives one worker message and validates tuple envelope shape."""
    try:
        raw_message = connection.recv()
    except EOFError as err:
        raise error_factory(
            f"{worker_label} worker exited before sending {stage} payload."
        ) from err
    if not isinstance(raw_message, tuple) or not raw_message:
        raise error_factory(f"{worker_label} worker returned malformed payload.")
    return raw_message


def is_setup_complete_message(
    *,
    message: tuple[Any, ...],
    worker_label: str,
    error_factory: _ErrorFactory,
) -> bool:
    """Returns whether one worker message marks setup completion."""
    if message[0] != "phase":
        return False
    if len(message) != 2 or message[1] != "setup_complete":
        raise error_factory(f"{worker_label} worker returned malformed phase payload.")
    return True


def parse_worker_completion_message(
    *,
    worker_message: tuple[Any, ...],
    worker_label: str,
    error_factory: _ErrorFactory,
    raise_worker_error: _WorkerErrorRaiser,
    result_type: type[_TResult],
) -> _TResult:
    """Parses one worker completion message and returns inference result."""
    kind = worker_message[0]
    if kind == "ok":
        if len(worker_message) != 2:
            raise error_factory(
                f"{worker_label} worker returned malformed success payload."
            )
        result = worker_message[1]
        if not isinstance(result, result_type):
            raise error_factory(
                f"{worker_label} worker returned unexpected result type."
            )
        return result
    if kind == "phase":
        raise error_factory(
            f"{worker_label} worker returned setup phase without completion payload."
        )
    if kind == "err":
        if len(worker_message) != 3:
            raise error_factory(
                f"{worker_label} worker returned malformed error payload."
            )
        raise_worker_error(
            cast(str, worker_message[1]),
            cast(str, worker_message[2]),
        )
    raise error_factory(f"{worker_label} worker returned unknown payload status.")


def run_with_timeout(
    *,
    operation: Callable[[], _TResult],
    timeout_seconds: float,
    timeout_error_factory: _ErrorFactory,
    timeout_label: str,
) -> _TResult:
    """Runs one operation with best-effort timeout enforcement."""
    if timeout_seconds <= 0.0:
        return operation()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(operation)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as err:
        future.cancel()
        raise timeout_error_factory(
            f"{timeout_label} exceeded timeout budget ({timeout_seconds:.2f}s)."
        ) from err
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def raise_worker_error(
    *,
    error_type: str,
    message: str,
    known_error_factories: dict[str, _ErrorFactory],
    unknown_error_factory: _ErrorFactory,
    worker_label: str,
) -> Never:
    """Rehydrates one worker error payload into runtime-domain exceptions."""
    known_error_factory = known_error_factories.get(error_type)
    if known_error_factory is not None:
        raise known_error_factory(message)
    raise unknown_error_factory(
        f"{worker_label} worker failed with {error_type}: {message}"
    )


def terminate_worker_process(
    *,
    process: BaseProcess,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Terminates a timed-out worker process with kill fallback."""
    process.terminate()
    process.join(timeout=terminate_grace_seconds)
    if process.is_alive():
        process.kill()
        process.join(timeout=kill_grace_seconds)


def run_process_setup_compute_handshake(
    *,
    context: Any,
    worker_target: Callable[[_TPayload, Connection], None],
    payload: _TPayload,
    timeout_seconds: float,
    recv_worker_message: _WorkerMessageReader[_TWorkerMessage],
    is_setup_complete_message: Callable[[_TWorkerMessage], bool],
    terminate_worker_process: Callable[[BaseProcess], None],
    timeout_error_factory: _ErrorFactory,
    execution_error_factory: _ErrorFactory,
    worker_label: str,
    process_join_grace_seconds: float,
    on_setup_complete: Callable[[], None] | None = None,
) -> _TWorkerMessage:
    """Runs one setup+compute worker handshake with bounded compute wait."""
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(payload, child_conn),
        daemon=False,
    )
    process.start()
    child_conn.close()
    try:
        try:
            setup_message = recv_worker_message(parent_conn, stage="setup")
            if is_setup_complete_message(setup_message):
                if on_setup_complete is not None:
                    on_setup_complete()
                if timeout_seconds <= 0.0:
                    worker_message = recv_worker_message(parent_conn, stage="compute")
                else:
                    if not parent_conn.poll(timeout_seconds):
                        if process.is_alive():
                            terminate_worker_process(process)
                            raise timeout_error_factory(
                                f"{worker_label} exceeded timeout budget "
                                f"({timeout_seconds:.2f}s)."
                            )
                        raise execution_error_factory(
                            f"{worker_label} worker exited before sending compute result."
                        )
                    worker_message = recv_worker_message(parent_conn, stage="compute")
            else:
                worker_message = setup_message
            if process.is_alive():
                process.join(timeout=process_join_grace_seconds)
            if process.exitcode not in (0, None) and not parent_conn.poll():
                raise execution_error_factory(
                    f"{worker_label} worker exited without a result payload."
                )
        finally:
            parent_conn.close()
            if process.is_alive():
                terminate_worker_process(process)
    except Exception:
        raise
    return worker_message


__all__ = [
    "is_setup_complete_message",
    "parse_worker_completion_message",
    "raise_worker_error",
    "recv_worker_message",
    "run_process_setup_compute_handshake",
    "run_with_timeout",
    "terminate_worker_process",
]
