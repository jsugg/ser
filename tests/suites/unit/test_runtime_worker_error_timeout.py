"""Behavior tests for shared runtime worker timeout/error helpers."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ser._internal.runtime.worker_lifecycle import (
    raise_worker_error,
    run_process_setup_compute_handshake,
    run_with_timeout,
)

pytestmark = pytest.mark.unit


class _WorkerTimeoutError(TimeoutError):
    """Synthetic timeout error used for helper behavior tests."""


def test_run_with_timeout_returns_result_when_timeout_disabled() -> None:
    """Timeout helper should execute inline when timeout budget is non-positive."""
    resolved = run_with_timeout(
        operation=lambda: "ok",
        timeout_seconds=0.0,
        timeout_error_factory=_WorkerTimeoutError,
        timeout_label="Synthetic worker",
    )
    assert resolved == "ok"


def test_run_with_timeout_raises_timeout_error() -> None:
    """Timeout helper should raise configured timeout-domain error on budget breach."""

    def _slow_operation() -> str:
        time.sleep(0.05)
        return "late"

    with pytest.raises(_WorkerTimeoutError, match="Synthetic worker exceeded timeout"):
        run_with_timeout(
            operation=_slow_operation,
            timeout_seconds=0.001,
            timeout_error_factory=_WorkerTimeoutError,
            timeout_label="Synthetic worker",
        )


def test_raise_worker_error_maps_known_error_types() -> None:
    """Worker error helper should rehydrate mapped error types."""
    with pytest.raises(ValueError, match="bad input"):
        raise_worker_error(
            error_type="ValueError",
            message="bad input",
            known_error_factories={"ValueError": ValueError},
            unknown_error_factory=RuntimeError,
            worker_label="Synthetic",
        )


def test_raise_worker_error_maps_unknown_error_types_to_domain_error() -> None:
    """Worker error helper should raise unknown-domain error for unmapped types."""
    with pytest.raises(
        RuntimeError,
        match="Synthetic worker failed with UnknownType: boom",
    ):
        raise_worker_error(
            error_type="UnknownType",
            message="boom",
            known_error_factories={"ValueError": ValueError},
            unknown_error_factory=RuntimeError,
            worker_label="Synthetic",
        )


def test_run_process_setup_compute_handshake_waits_after_setup_only() -> None:
    """Process handshake timeout should apply to compute stage only."""
    poll_calls: list[float | None] = []
    expected = ("ok", "payload")

    class _ParentConnection:
        def __init__(self) -> None:
            self._messages: list[tuple[object, ...]] = [
                ("phase", "setup_complete"),
                expected,
            ]

        def recv(self) -> tuple[object, ...]:
            if not self._messages:
                raise EOFError
            return self._messages.pop(0)

        def poll(self, timeout: float | None = None) -> bool:
            poll_calls.append(timeout)
            return True

        def close(self) -> None:
            return None

    class _ChildConnection:
        def close(self) -> None:
            return None

    class _Process:
        exitcode = 0

        def __init__(self) -> None:
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def kill(self) -> None:
            self._alive = False

    class _Context:
        def __init__(self) -> None:
            self._parent = _ParentConnection()
            self._child = _ChildConnection()
            self._process = _Process()

        def Pipe(
            self, duplex: bool = False
        ) -> tuple[_ParentConnection, _ChildConnection]:  # noqa: N802
            del duplex
            return self._parent, self._child

        def Process(self, **_kwargs: object) -> _Process:  # noqa: N802
            return self._process

    setup_callbacks = 0

    def _on_setup_complete() -> None:
        nonlocal setup_callbacks
        setup_callbacks += 1

    resolved = run_process_setup_compute_handshake(
        context=cast(Any, _Context()),
        worker_target=lambda _payload, _conn: None,
        payload=SimpleNamespace(),
        timeout_seconds=7.0,
        recv_worker_message=lambda connection, stage: cast(
            tuple[str, object],
            connection.recv(),
        ),
        is_setup_complete_message=lambda message: message[0] == "phase",
        terminate_worker_process=lambda process: process.terminate(),
        timeout_error_factory=_WorkerTimeoutError,
        execution_error_factory=RuntimeError,
        worker_label="Synthetic",
        process_join_grace_seconds=0.5,
        on_setup_complete=_on_setup_complete,
    )

    assert resolved == expected
    assert setup_callbacks == 1
    assert poll_calls == [7.0]
