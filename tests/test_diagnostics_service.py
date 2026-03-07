"""Tests for diagnostics report semantics and formatting."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ser.config as config_module
import ser.diagnostics.service as diagnostics_service
from ser.diagnostics.domain import DiagnosticFinding, DiagnosticReport
from ser.diagnostics.service import (
    format_report_brief,
    format_report_json,
    format_report_text,
    parse_preflight_mode,
    run_doctor_diagnostics,
    run_startup_preflight,
    should_fail_preflight,
)
from ser.transcript.backends import CompatibilityIssue, CompatibilityReport
from ser.utils.transcription_compat import FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE


def test_parse_preflight_mode_falls_back_to_warn_for_invalid_values() -> None:
    """Invalid preflight mode inputs should resolve to warn."""
    assert parse_preflight_mode("strict") == "strict"
    assert parse_preflight_mode("off") == "off"
    assert parse_preflight_mode("warn") == "warn"
    assert parse_preflight_mode("invalid-mode") == "warn"


def test_should_fail_preflight_obeys_mode_semantics() -> None:
    """Preflight failure logic should reflect off/warn/strict contracts."""
    blocking_report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="runtime_capability_unavailable",
                severity="error",
                message="Backend unavailable",
                blocking=True,
            ),
        )
    )
    warning_report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_faster_whisper_mps_unsupported",
                severity="warning",
                message="Operational warning",
            ),
        )
    )
    assert should_fail_preflight(report=blocking_report, mode="off") is False
    assert should_fail_preflight(report=blocking_report, mode="warn") is True
    assert should_fail_preflight(report=warning_report, mode="warn") is False
    assert should_fail_preflight(report=warning_report, mode="strict") is True


def test_format_report_text_and_json_include_remediation_details() -> None:
    """Report formatters should preserve remediation and summary details."""
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="ffmpeg_binary_missing",
                severity="error",
                message="ffmpeg executable was not found on PATH.",
                remediation=("Install ffmpeg.",),
                blocking=True,
            ),
        )
    )
    text_output = format_report_text(report)
    json_output = format_report_json(report)
    payload = json.loads(json_output)

    assert "SER diagnostics report" in text_output
    assert "[ERROR] ffmpeg_binary_missing: blocking" in text_output
    assert "remediation: Install ffmpeg." in text_output
    assert payload["summary"]["has_blocking_findings"] is True
    assert payload["findings"][0]["code"] == "ffmpeg_binary_missing"
    assert payload["findings"][0]["remediation"] == ["Install ffmpeg."]


def test_format_report_text_marks_warnings_as_advisory_when_non_blocking() -> None:
    """Non-blocking warnings should be rendered as advisory in text output."""
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_faster_whisper_mps_unsupported",
                severity="warning",
                message="MPS is unsupported and CPU fallback will be used.",
            ),
        )
    )

    text_output = format_report_text(report)
    assert (
        "[WARNING] transcription_operational_faster_whisper_mps_unsupported: advisory"
        in text_output
    )


def test_format_report_text_marks_info_as_informational() -> None:
    """Informational findings should be explicitly marked as informational."""
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_torio_ffmpeg_abi_mismatch",
                severity="info",
                message="Non-blocking runtime advisory.",
            ),
        )
    )

    text_output = format_report_text(report)
    assert (
        "[INFO] transcription_operational_torio_ffmpeg_abi_mismatch: informational"
        in text_output
    )


def test_format_report_brief_condenses_long_messages() -> None:
    """Compact preflight formatter should avoid long repeated remediation logs."""
    report = DiagnosticReport(
        findings=(
            DiagnosticFinding(
                code="transcription_operational_faster_whisper_mps_unsupported",
                severity="warning",
                message=(
                    "faster-whisper backend does not support MPS runtime; "
                    "transcription will use CPU fallback and throughput may "
                    "be lower for long clips."
                ),
            ),
        )
    )
    brief = format_report_brief(report, max_message_chars=90)
    assert "SER diagnostics preflight summary" in brief
    assert (
        "[WARNING] transcription_operational_faster_whisper_mps_unsupported (advisory):"
        in brief
    )
    assert "..." in brief


def test_fast_openmp_conflict_is_info_for_cli_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known harmless faster-whisper OpenMP conflicts should be informational."""
    settings = config_module.reload_settings()
    settings = replace(
        settings,
        runtime_flags=replace(
            settings.runtime_flags,
            profile_pipeline=True,
            medium_profile=False,
            accurate_profile=False,
            accurate_research_profile=False,
        ),
    )

    class _FakeAdapter:
        def check_compatibility(
            self,
            *,
            runtime_request: object,
            settings: object,
        ) -> CompatibilityReport:
            del runtime_request
            del settings
            return CompatibilityReport(
                backend_id="faster_whisper",
                functional_issues=(
                    CompatibilityIssue(
                        code=FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE,
                        message="OpenMP conflict detected.",
                    ),
                ),
            )

    monkeypatch.setattr(
        diagnostics_service,
        "resolve_runtime_capability",
        lambda *args, **kwargs: SimpleNamespace(available=True),
    )
    monkeypatch.setattr(
        diagnostics_service.shutil, "which", lambda _name: "/usr/bin/ffmpeg"
    )
    monkeypatch.setattr(
        diagnostics_service,
        "resolve_profile_transcription_config",
        lambda _profile: ("faster_whisper", "distil-large-v3", False, True),
    )
    monkeypatch.setattr(
        diagnostics_service,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: _FakeAdapter(),
    )

    report = run_startup_preflight(
        settings=settings,
        include_transcription_checks=True,
    )

    assert report.has_blocking_findings is False
    assert any(
        finding.code
        == "transcription_operational_faster_whisper_openmp_runtime_conflict"
        and finding.severity == "info"
        for finding in report.findings
    )
    assert should_fail_preflight(report=report, mode="warn") is False


def test_run_doctor_diagnostics_includes_dataset_registry_health_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doctor diagnostics should surface dataset registry provenance issues."""
    settings = config_module.reload_settings()
    monkeypatch.setattr(
        diagnostics_service,
        "resolve_runtime_capability",
        lambda *args, **kwargs: SimpleNamespace(available=True, message=None),
    )
    monkeypatch.setattr(
        diagnostics_service,
        "collect_dataset_registry_health_issues",
        lambda **kwargs: (
            SimpleNamespace(
                dataset_id="msp-podcast",
                code="source_provenance_mismatch",
                message="Source mismatch.",
            ),
        ),
    )

    report = run_doctor_diagnostics(
        settings=settings,
        include_transcription_checks=False,
        include_noise_findings=False,
    )

    assert any(
        finding.code == "dataset_registry_source_provenance_mismatch"
        and finding.severity == "error"
        for finding in report.findings
    )
