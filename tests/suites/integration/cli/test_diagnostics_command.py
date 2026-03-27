"""Tests for `ser doctor` command behavior."""

from __future__ import annotations

import pytest

import ser.config as config_module
from ser.diagnostics.command import run_doctor_command
from ser.diagnostics.domain import DiagnosticFinding, DiagnosticReport


def test_run_doctor_command_strict_returns_one_for_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode should fail with exit code 1 when only warnings are present."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_faster_whisper_mps_unsupported",
                severity="warning",
                message="Operational warning",
            ),
        )
    )

    monkeypatch.setattr(
        "ser.diagnostics.command.run_doctor_diagnostics",
        lambda **_kwargs: report,
    )

    exit_code = run_doctor_command(["--strict"], settings=settings)
    assert exit_code == 1


def test_run_doctor_command_strict_returns_zero_for_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode should not fail when report contains only informational findings."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_torio_ffmpeg_abi_mismatch",
                severity="info",
                message="Non-blocking informational advisory",
            ),
        )
    )

    monkeypatch.setattr(
        "ser.diagnostics.command.run_doctor_diagnostics",
        lambda **_kwargs: report,
    )

    exit_code = run_doctor_command(["--strict"], settings=settings)
    assert exit_code == 0


def test_run_doctor_command_returns_two_for_blocking_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blocking diagnostics findings should return exit code 2."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="runtime_capability_unavailable",
                severity="error",
                message="Missing dependency",
                blocking=True,
            ),
        )
    )

    monkeypatch.setattr(
        "ser.diagnostics.command.run_doctor_diagnostics",
        lambda **_kwargs: report,
    )

    exit_code = run_doctor_command([], settings=settings)
    assert exit_code == 2


def test_run_doctor_command_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """JSON output flag should print a JSON diagnostics payload."""
    settings = config_module.reload_settings()
    report = DiagnosticReport(findings=())

    monkeypatch.setattr(
        "ser.diagnostics.command.run_doctor_diagnostics",
        lambda **_kwargs: report,
    )

    exit_code = run_doctor_command(["--format", "json"], settings=settings)
    output = capsys.readouterr().out
    assert exit_code == 0
    assert '"summary"' in output
    assert '"findings"' in output
