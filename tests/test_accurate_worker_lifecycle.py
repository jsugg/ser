"""Tests for accurate worker lifecycle adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import SimpleNamespace
from typing import cast

import pytest

from ser.runtime import accurate_worker_lifecycle


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    expected_profile: str = "accurate"


def test_run_with_process_timeout_delegates_to_timeout_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle owner should delegate timeout orchestration to shared helper."""
    captured: dict[str, object] = {}
    expected = object()
    connection = cast(Connection, SimpleNamespace())

    monkeypatch.setattr(
        accurate_worker_lifecycle.accurate_process_timeout_helpers,
        "run_with_process_timeout",
        lambda **kwargs: captured.update(kwargs) or expected,
    )

    result = accurate_worker_lifecycle.run_with_process_timeout(
        payload=_PayloadStub(),
        timeout_seconds=7.0,
        get_context=lambda _name: object(),
        logger=logging.getLogger("ser.tests.accurate_worker_lifecycle"),
        setup_phase_name="setup",
        inference_phase_name="inference",
        log_phase_started=lambda *_args, **_kwargs: None,
        log_phase_completed=lambda *_args, **_kwargs: None,
        log_phase_failed=lambda *_args, **_kwargs: None,
        run_process_setup_compute_handshake=lambda **_kwargs: ("ok", connection),
        worker_target=lambda _payload, _connection: None,
        recv_worker_message=lambda _connection: ("ok", "done"),
        is_setup_complete_message=lambda _message: True,
        terminate_worker_process=lambda _process: None,
        timeout_error_factory=TimeoutError,
        execution_error_factory=RuntimeError,
        worker_label="Accurate inference",
        process_join_grace_seconds=0.5,
        parse_worker_completion_message=lambda _message: expected,
    )

    assert result is expected
    assert captured["payload"] == _PayloadStub()
    assert captured["timeout_seconds"] == 7.0
    assert captured["worker_label"] == "Accurate inference"


def test_run_worker_entry_delegates_to_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle owner should delegate child-process execution to worker binding."""
    captured: dict[str, object] = {}
    payload = object()
    connection = cast(Connection, SimpleNamespace())

    monkeypatch.setattr(
        accurate_worker_lifecycle,
        "_run_worker_entry_binding",
        lambda **kwargs: captured.update(kwargs),
    )

    accurate_worker_lifecycle.run_worker_entry(
        payload,
        connection,
        prepare_process_operation=lambda _payload: "prepared",
        run_process_operation=lambda _prepared: None,
    )

    assert captured["payload"] is payload
    assert captured["connection"] is connection
    assert callable(captured["prepare_process_operation"])
    assert callable(captured["run_process_operation"])


def test_raise_worker_error_delegates_to_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle owner should wire accurate-specific worker error mapping."""
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        accurate_worker_lifecycle,
        "_raise_worker_error_binding",
        lambda **kwargs: captured.update(kwargs),
    )

    accurate_worker_lifecycle.raise_worker_error(
        "RuntimeError",
        "boom",
        impl=lambda **_kwargs: None,
        known_error_factories={"RuntimeError": RuntimeError},
        unknown_error_factory=ValueError,
        worker_label="Accurate inference",
    )

    assert captured["error_type"] == "RuntimeError"
    assert captured["message"] == "boom"
    assert captured["worker_label"] == "Accurate inference"
    assert captured["unknown_error_factory"] is ValueError


def test_terminate_worker_process_delegates_to_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lifecycle owner should delegate timed-out process termination policy."""
    captured: dict[str, object] = {}
    process = cast(BaseProcess, SimpleNamespace())

    monkeypatch.setattr(
        accurate_worker_lifecycle,
        "_terminate_worker_process_binding",
        lambda **kwargs: captured.update(kwargs),
    )

    accurate_worker_lifecycle.terminate_worker_process(
        process,
        impl=lambda **_kwargs: None,
        terminate_grace_seconds=0.5,
        kill_grace_seconds=0.25,
    )

    assert captured["process"] is process
    assert captured["terminate_grace_seconds"] == 0.5
    assert captured["kill_grace_seconds"] == 0.25
