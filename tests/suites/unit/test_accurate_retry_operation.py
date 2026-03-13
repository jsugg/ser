"""Tests for accurate retry-operation orchestration helper."""

from __future__ import annotations

import logging

import pytest

from ser.runtime.accurate_retry_operation import run_accurate_retry_operation

pytestmark = pytest.mark.unit


def test_run_accurate_retry_operation_requires_process_payload() -> None:
    """Process-isolated operation should fail fast when payload is absent."""
    with pytest.raises(
        RuntimeError,
        match="Accurate process payload is missing for isolated execution.",
    ):
        run_accurate_retry_operation(
            enforce_timeout=True,
            use_process_isolation=True,
            process_payload=None,
            timeout_seconds=12.0,
            expected_profile="accurate",
            logger=logging.getLogger(__name__),
            run_with_process_timeout=lambda _payload: "never",
            run_once_inprocess=lambda: "never",
            run_with_timeout=lambda **_kwargs: "never",
            timeout_error_factory=TimeoutError,
            log_phase_started=lambda *_args, **_kwargs: 1.0,
            log_phase_completed=lambda *_args, **_kwargs: None,
            log_phase_failed=lambda *_args, **_kwargs: None,
            phase_name="phase",
        )


def test_run_accurate_retry_operation_process_isolation_delegates() -> None:
    """Process-isolated timeout path should delegate to process-timeout runner."""
    captured: dict[str, object] = {}

    def _run_with_process_timeout(payload: object) -> str:
        captured["payload"] = payload
        return "ok"

    payload = object()
    result = run_accurate_retry_operation(
        enforce_timeout=True,
        use_process_isolation=True,
        process_payload=payload,
        timeout_seconds=33.0,
        expected_profile="accurate",
        logger=logging.getLogger(__name__),
        run_with_process_timeout=_run_with_process_timeout,
        run_once_inprocess=lambda: "never",
        run_with_timeout=lambda **_kwargs: "never",
        timeout_error_factory=TimeoutError,
        log_phase_started=lambda *_args, **_kwargs: 1.0,
        log_phase_completed=lambda *_args, **_kwargs: None,
        log_phase_failed=lambda *_args, **_kwargs: None,
        phase_name="phase",
    )

    assert result == "ok"
    assert captured == {"payload": payload}


def test_run_accurate_retry_operation_inprocess_timeout_path() -> None:
    """In-process timeout path should run through timeout wrapper and phase logs."""
    events: list[str] = []

    def _run_with_timeout(**kwargs: object) -> str:
        operation = kwargs["operation"]
        assert callable(operation)
        events.append("timeout_wrapper")
        return "timeout_ok"

    def _log_phase_started(*_args: object, **_kwargs: object) -> float:
        events.append("start")
        return 1.0

    result = run_accurate_retry_operation(
        enforce_timeout=True,
        use_process_isolation=False,
        process_payload=None,
        timeout_seconds=9.0,
        expected_profile="accurate",
        logger=logging.getLogger(__name__),
        run_with_process_timeout=lambda _payload: "never",
        run_once_inprocess=lambda: "run_once",
        run_with_timeout=_run_with_timeout,
        timeout_error_factory=TimeoutError,
        log_phase_started=_log_phase_started,
        log_phase_completed=lambda *_args, **_kwargs: events.append("complete"),
        log_phase_failed=lambda *_args, **_kwargs: events.append("failed"),
        phase_name="phase",
    )

    assert result == "timeout_ok"
    assert events == ["start", "timeout_wrapper", "complete"]


def test_run_accurate_retry_operation_inprocess_failure_logs_failed() -> None:
    """In-process failure should emit failed phase log and re-raise error."""
    events: list[str] = []

    def _log_phase_started(*_args: object, **_kwargs: object) -> float:
        events.append("start")
        return 1.0

    with pytest.raises(ValueError, match="boom"):
        run_accurate_retry_operation(
            enforce_timeout=False,
            use_process_isolation=False,
            process_payload=None,
            timeout_seconds=4.0,
            expected_profile="accurate",
            logger=logging.getLogger(__name__),
            run_with_process_timeout=lambda _payload: "never",
            run_once_inprocess=lambda: (_ for _ in ()).throw(ValueError("boom")),
            run_with_timeout=lambda **_kwargs: "never",
            timeout_error_factory=TimeoutError,
            log_phase_started=_log_phase_started,
            log_phase_completed=lambda *_args, **_kwargs: events.append("complete"),
            log_phase_failed=lambda *_args, **_kwargs: events.append("failed"),
            phase_name="phase",
        )

    assert events == ["start", "failed"]
