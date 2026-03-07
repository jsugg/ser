"""Diagnostics-focused public API helpers for library and CLI orchestration."""

from __future__ import annotations

from typing import Literal

from ser.config import AppConfig, resolve_profile_transcription_config
from ser.diagnostics.domain import DiagnosticReport, PreflightMode
from ser.profiles import resolve_profile_name

type PreflightLogLevel = Literal["info", "error"]
type PreflightLogRecord = tuple[PreflightLogLevel, str]


def preflight_command_requested(
    *,
    train: bool,
    file_path: str | None,
    calibrate_transcription_runtime: bool,
) -> bool:
    """Returns whether one CLI invocation requests executable workflow actions."""
    return bool(train or file_path or calibrate_transcription_runtime)


def preflight_includes_transcription_checks(
    *,
    file_path: str | None,
    no_transcript: bool,
    calibrate_transcription_runtime: bool,
) -> bool:
    """Returns whether startup preflight should include transcription checks."""
    return bool(calibrate_transcription_runtime or (file_path and not no_transcript))


def resolve_doctor_command(*, profile: str | None) -> str:
    """Returns doctor command hint string for the selected CLI profile."""
    profile_hint = f" --profile {profile}" if isinstance(profile, str) else ""
    return f"ser doctor{profile_hint}"


def suppress_preflight_transcription_operational_relogs(
    *,
    settings: AppConfig,
    report: DiagnosticReport,
) -> None:
    """Pre-marks startup-reported transcription operational issues as emitted once."""
    issue_codes = tuple(
        finding.code.removeprefix("transcription_operational_")
        for finding in report.findings
        if finding.code.startswith("transcription_operational_")
    )
    if not issue_codes:
        return
    profile_name = resolve_profile_name(settings)
    backend_id, _model_name, _use_demucs, _use_vad = (
        resolve_profile_transcription_config(profile_name)
    )
    from ser.transcript.transcript_extractor import mark_compatibility_issues_as_emitted

    mark_compatibility_issues_as_emitted(
        backend_id=backend_id,
        issue_kind="operational",
        issue_codes=issue_codes,
    )


def run_doctor_command(argv: list[str], *, settings: AppConfig | None = None) -> int:
    """Runs `ser doctor ...` command argv via the public API boundary."""
    from ser.diagnostics.command import run_doctor_command as _run_doctor_command

    return _run_doctor_command(argv, settings=settings)


def parse_preflight_mode(raw_mode: str) -> PreflightMode:
    """Parses one startup preflight mode string."""
    from ser.diagnostics.service import parse_preflight_mode as _parse_preflight_mode

    return _parse_preflight_mode(raw_mode)


def run_startup_preflight(
    *,
    settings: AppConfig,
    include_transcription_checks: bool,
) -> DiagnosticReport:
    """Runs one startup preflight diagnostics check suite."""
    from ser.diagnostics.service import run_startup_preflight as _run_startup_preflight

    return _run_startup_preflight(
        settings=settings,
        include_transcription_checks=include_transcription_checks,
    )


def should_fail_preflight(*, report: DiagnosticReport, mode: PreflightMode) -> bool:
    """Returns whether one preflight report should block execution."""
    from ser.diagnostics.service import should_fail_preflight as _should_fail_preflight

    return _should_fail_preflight(report=report, mode=mode)


def format_startup_preflight_one_liner(
    report: DiagnosticReport,
    *,
    doctor_command: str,
) -> str:
    """Formats one concise one-line startup preflight summary."""
    from ser.diagnostics.service import (
        format_startup_preflight_one_liner as _format_startup_preflight_one_liner,
    )

    return _format_startup_preflight_one_liner(
        report,
        doctor_command=doctor_command,
    )


def run_startup_preflight_cli_gate(
    *,
    settings: AppConfig,
    mode: PreflightMode,
    profile: str | None,
    train_requested: bool,
    file_path: str | None,
    no_transcript: bool,
    calibrate_transcription_runtime: bool,
) -> tuple[tuple[PreflightLogRecord, ...], int | None]:
    """Evaluates startup preflight policy and returns log records plus optional exit."""
    if mode == "off":
        return ((), None)
    if not preflight_command_requested(
        train=train_requested,
        file_path=file_path,
        calibrate_transcription_runtime=calibrate_transcription_runtime,
    ):
        return ((), None)
    report = run_startup_preflight(
        settings=settings,
        include_transcription_checks=preflight_includes_transcription_checks(
            file_path=file_path,
            no_transcript=no_transcript,
            calibrate_transcription_runtime=calibrate_transcription_runtime,
        ),
    )
    doctor_command = resolve_doctor_command(profile=profile)
    fail_preflight = should_fail_preflight(report=report, mode=mode)
    logs: list[PreflightLogRecord] = []
    if report.findings:
        preflight_summary = format_startup_preflight_one_liner(
            report,
            doctor_command=doctor_command,
        )
        logs.append(("error" if fail_preflight else "info", preflight_summary))
        if not fail_preflight:
            suppress_preflight_transcription_operational_relogs(
                settings=settings,
                report=report,
            )
    if fail_preflight:
        logs.append(("error", f"Startup preflight failed (mode={mode})."))
        return (tuple(logs), 2)
    return (tuple(logs), None)


__all__ = [
    "format_startup_preflight_one_liner",
    "parse_preflight_mode",
    "preflight_command_requested",
    "preflight_includes_transcription_checks",
    "resolve_doctor_command",
    "run_doctor_command",
    "run_startup_preflight_cli_gate",
    "run_startup_preflight",
    "suppress_preflight_transcription_operational_relogs",
    "should_fail_preflight",
]
