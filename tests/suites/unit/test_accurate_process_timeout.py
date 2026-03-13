"""Contract tests for accurate process-timeout orchestration helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from ser.runtime import accurate_process_timeout

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Payload:
    expected_profile: str


def test_run_with_process_timeout_completes_setup_and_inference_phases() -> None:
    """Helper should log setup/inference completion and return parsed result."""
    phase_started: list[str] = []
    phase_completed: list[str] = []
    handshake_kwargs: dict[str, object] = {}
    expected_result = {"status": "ok"}

    def _log_phase_started(_logger: object, *, phase_name: str, profile: str) -> float:
        phase_started.append(f"{phase_name}:{profile}")
        return float(len(phase_started))

    def _log_phase_completed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float | None,
        profile: str,
    ) -> None:
        del started_at
        phase_completed.append(f"{phase_name}:{profile}")

    def _log_phase_failed(**_kwargs: object) -> None:
        raise AssertionError("No phase failure expected on happy path.")

    def _run_handshake(**kwargs: object) -> tuple[str, dict[str, str]]:
        handshake_kwargs.update(kwargs)
        on_setup_complete = kwargs.get("on_setup_complete")
        assert callable(on_setup_complete)
        on_setup_complete()
        return ("ok", {"worker": "done"})

    result = accurate_process_timeout.run_with_process_timeout(
        payload=_Payload(expected_profile="accurate"),
        timeout_seconds=7.0,
        get_context=lambda _name: object(),
        logger=logging.getLogger("ser.tests.accurate_process_timeout"),
        setup_phase_name="setup",
        inference_phase_name="inference",
        log_phase_started=_log_phase_started,
        log_phase_completed=_log_phase_completed,
        log_phase_failed=_log_phase_failed,
        run_process_setup_compute_handshake=_run_handshake,
        worker_target=lambda *_args: None,
        recv_worker_message=lambda **_kwargs: ("ok", {"worker": "done"}),
        is_setup_complete_message=lambda _message: True,
        terminate_worker_process=lambda *_args, **_kwargs: None,
        timeout_error_factory=TimeoutError,
        execution_error_factory=RuntimeError,
        worker_label="Accurate inference",
        process_join_grace_seconds=0.5,
        parse_worker_completion_message=lambda _message: expected_result,
    )

    assert result == expected_result
    assert phase_started == ["setup:accurate", "inference:accurate"]
    assert phase_completed == ["setup:accurate", "inference:accurate"]
    assert handshake_kwargs["timeout_seconds"] == 7.0


def test_run_with_process_timeout_logs_inference_failure_on_parse_error() -> None:
    """Helper should record inference phase failure when completion parse fails."""
    phase_failed: list[str] = []

    def _log_phase_started(_logger: object, *, phase_name: str, profile: str) -> float:
        del profile
        return 1.0 if phase_name == "setup" else 2.0

    def _log_phase_completed(_logger: object, **_kwargs: object) -> None:
        return None

    def _log_phase_failed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float | None,
        profile: str,
    ) -> None:
        del started_at
        phase_failed.append(f"{phase_name}:{profile}")

    def _run_handshake(**kwargs: object) -> tuple[str, str]:
        on_setup_complete = kwargs.get("on_setup_complete")
        assert callable(on_setup_complete)
        on_setup_complete()
        return ("ok", "worker-message")

    with pytest.raises(RuntimeError, match="parse failed"):
        accurate_process_timeout.run_with_process_timeout(
            payload=_Payload(expected_profile="accurate"),
            timeout_seconds=5.0,
            get_context=lambda _name: object(),
            logger=logging.getLogger("ser.tests.accurate_process_timeout"),
            setup_phase_name="setup",
            inference_phase_name="inference",
            log_phase_started=_log_phase_started,
            log_phase_completed=_log_phase_completed,
            log_phase_failed=_log_phase_failed,
            run_process_setup_compute_handshake=_run_handshake,
            worker_target=lambda *_args: None,
            recv_worker_message=lambda **_kwargs: ("ok", "worker-message"),
            is_setup_complete_message=lambda _message: True,
            terminate_worker_process=lambda *_args, **_kwargs: None,
            timeout_error_factory=TimeoutError,
            execution_error_factory=RuntimeError,
            worker_label="Accurate inference",
            process_join_grace_seconds=0.5,
            parse_worker_completion_message=lambda _message: (_ for _ in ()).throw(
                RuntimeError("parse failed")
            ),
        )

    assert phase_failed == ["inference:accurate"]
