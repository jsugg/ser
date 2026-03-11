"""Diagnostics service for startup preflight and doctor command workflows."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from typing import Literal, cast

from ser._internal.config.bootstrap import resolve_profile_transcription_config
from ser.config import AppConfig
from ser.data.dataset_prepare import collect_dataset_registry_health_issues
from ser.diagnostics.domain import (
    DiagnosticFinding,
    DiagnosticReport,
    PreflightMode,
)
from ser.profiles import ProfileName, get_profile_catalog, resolve_profile_name
from ser.runtime.registry import resolve_runtime_capability
from ser.transcript.backends import (
    BackendRuntimeRequest,
    CompatibilityIssueImpact,
    resolve_transcription_backend_adapter,
)
from ser.transcript.runtime_policy import (
    DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    resolve_transcription_runtime_policy,
)
from ser.utils.transcription_compat import (
    FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE,
    resolve_transcription_compatibility_lane,
)

_SUPPORTED_PREFLIGHT_MODES: frozenset[str] = frozenset({"off", "warn", "strict"})


def parse_preflight_mode(raw_mode: str) -> PreflightMode:
    """Parses one user-facing preflight mode string."""
    normalized = raw_mode.strip().lower()
    if normalized in _SUPPORTED_PREFLIGHT_MODES:
        return cast(PreflightMode, normalized)
    return "warn"


def apply_profile_override_for_diagnostics(
    settings: AppConfig,
    *,
    profile: ProfileName | None,
) -> AppConfig:
    """Returns settings with runtime profile flags overridden for diagnostics only."""
    if profile is None:
        return settings
    runtime_flags = replace(
        settings.runtime_flags,
        profile_pipeline=True,
        medium_profile=profile == "medium",
        accurate_profile=profile == "accurate",
        accurate_research_profile=profile == "accurate-research",
        restricted_backends=bool(settings.runtime_flags.restricted_backends),
    )
    return replace(settings, runtime_flags=runtime_flags)


def run_startup_preflight(
    *,
    settings: AppConfig,
    include_transcription_checks: bool,
) -> DiagnosticReport:
    """Runs one fast startup preflight report for command execution gating."""
    findings = _run_checks(
        settings=settings,
        include_transcription_checks=include_transcription_checks,
        include_noise_findings=False,
        include_lane_info=False,
    )
    return DiagnosticReport(findings=tuple(findings))


def run_doctor_diagnostics(
    *,
    settings: AppConfig,
    include_transcription_checks: bool,
    include_noise_findings: bool,
) -> DiagnosticReport:
    """Runs one comprehensive diagnostics report for interactive troubleshooting."""
    findings = _run_checks(
        settings=settings,
        include_transcription_checks=include_transcription_checks,
        include_noise_findings=include_noise_findings,
        include_lane_info=True,
        include_dataset_registry_checks=True,
    )
    return DiagnosticReport(findings=tuple(findings))


def should_fail_preflight(*, report: DiagnosticReport, mode: PreflightMode) -> bool:
    """Returns whether one startup preflight report should block command execution."""
    if mode == "off":
        return False
    if report.has_blocking_findings:
        return True
    return mode == "strict" and report.has_warning_or_higher


def format_report_text(report: DiagnosticReport) -> str:
    """Formats one diagnostic report into human-readable CLI text."""
    counts = report.counts_by_severity()
    lines: list[str] = [
        "SER diagnostics report",
        ("summary: " f"info={counts['info']} warning={counts['warning']} error={counts['error']}"),
    ]
    if not report.findings:
        lines.append("status: ok (no findings)")
        return "\n".join(lines)
    for finding in report.findings:
        level = finding.severity.upper()
        status_label = (
            " blocking"
            if finding.blocking
            else (
                " advisory"
                if finding.severity == "warning"
                else (" informational" if finding.severity == "info" else "")
            )
        )
        lines.append(f"[{level}] {finding.code}:{status_label} {finding.message}")
        for remediation in finding.remediation:
            lines.append(f"  remediation: {remediation}")
    return "\n".join(lines)


def format_report_brief(
    report: DiagnosticReport,
    *,
    max_findings: int = 3,
    max_message_chars: int = 180,
) -> str:
    """Formats one compact report variant for startup log hygiene."""
    counts = report.counts_by_severity()
    lines: list[str] = [
        "SER diagnostics preflight summary",
        ("summary: " f"info={counts['info']} warning={counts['warning']} error={counts['error']}"),
    ]
    if not report.findings:
        lines.append("status: ok (no findings)")
        return "\n".join(lines)
    for finding in report.findings[:max_findings]:
        status_label = (
            "blocking"
            if finding.blocking
            else ("advisory" if finding.severity == "warning" else "info")
        )
        lines.append(
            f"[{finding.severity.upper()}] {finding.code} ({status_label}): "
            f"{_condense_message(finding.message, max_chars=max_message_chars)}"
        )
    omitted_count = len(report.findings) - max_findings
    if omitted_count > 0:
        lines.append(f"... {omitted_count} additional finding(s) omitted.")
    return "\n".join(lines)


def format_startup_preflight_one_liner(
    report: DiagnosticReport,
    *,
    doctor_command: str,
) -> str:
    """Formats one single-line preflight summary for concise startup logging."""
    if not report.findings:
        return "Startup preflight: no findings."
    primary = report.findings[0]
    additional = len(report.findings) - 1
    additional_suffix = f" (+{additional} more)" if additional > 0 else ""
    if primary.blocking or report.has_error:
        return (
            f"Startup preflight blocking issue [{primary.code}]{additional_suffix}. "
            f"Execution halted. Run `{doctor_command}` for full details."
        )
    return (
        f"Startup preflight advisory [{primary.code}]{additional_suffix}. "
        f"Runtime will continue. Run `{doctor_command}` for full details."
    )


def format_report_json(report: DiagnosticReport) -> str:
    """Formats one diagnostic report as stable JSON output."""
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)


def _run_checks(
    *,
    settings: AppConfig,
    include_transcription_checks: bool,
    include_noise_findings: bool,
    include_lane_info: bool,
    include_dataset_registry_checks: bool = False,
) -> list[DiagnosticFinding]:
    """Runs one deterministic diagnostics check suite."""
    findings: list[DiagnosticFinding] = []
    if include_lane_info:
        lane = resolve_transcription_compatibility_lane()
        findings.append(
            DiagnosticFinding(
                code="environment_transcription_lane",
                severity="info",
                message=f"Resolved transcription compatibility lane: {lane}.",
            )
        )
    findings.extend(_check_runtime_capability(settings))
    if include_transcription_checks:
        findings.extend(_check_ffmpeg_binary())
        findings.extend(
            _check_transcription_backend_compatibility(
                settings=settings,
                include_noise_findings=include_noise_findings,
            )
        )
    if include_dataset_registry_checks:
        findings.extend(_check_dataset_registry_health(settings=settings))
    return findings


def _check_runtime_capability(settings: AppConfig) -> tuple[DiagnosticFinding, ...]:
    """Validates runtime profile dependency availability for selected profile."""
    known_backend_ids: frozenset[str] = frozenset(
        {"handcrafted", *(entry.backend_id for entry in get_profile_catalog().values())}
    )
    capability = resolve_runtime_capability(
        settings,
        available_backend_hooks=known_backend_ids,
    )
    if capability.available:
        return ()
    return (
        DiagnosticFinding(
            code="runtime_capability_unavailable",
            severity="error",
            message=capability.message or "Selected runtime profile capability is unavailable.",
            blocking=True,
        ),
    )


def _check_ffmpeg_binary() -> tuple[DiagnosticFinding, ...]:
    """Ensures one ffmpeg executable is available on PATH for transcription workflows."""
    if shutil.which("ffmpeg") is not None:
        return ()
    return (
        DiagnosticFinding(
            code="ffmpeg_binary_missing",
            severity="error",
            message=(
                "ffmpeg executable was not found on PATH; transcription workflows "
                "require ffmpeg."
            ),
            remediation=("Install ffmpeg and rerun diagnostics (`brew install ffmpeg` on macOS).",),
            blocking=True,
        ),
    )


def _check_transcription_backend_compatibility(
    *,
    settings: AppConfig,
    include_noise_findings: bool,
) -> tuple[DiagnosticFinding, ...]:
    """Collects transcription backend compatibility findings for active profile."""
    profile_name = resolve_profile_name(settings)
    backend_id, model_name, use_demucs, use_vad = resolve_profile_transcription_config(profile_name)
    torch_runtime = settings.torch_runtime
    requested_device = torch_runtime.device if isinstance(torch_runtime.device, str) else "cpu"
    requested_dtype = torch_runtime.dtype if isinstance(torch_runtime.dtype, str) else "auto"
    requested_threshold = settings.transcription.mps_low_memory_threshold_gb
    runtime_policy = resolve_transcription_runtime_policy(
        backend_id=backend_id,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
        mps_low_memory_threshold_gb=(
            requested_threshold
            if (
                isinstance(requested_threshold, int | float)
                and not isinstance(requested_threshold, bool)
            )
            else DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB
        ),
    )
    runtime_request = BackendRuntimeRequest(
        model_name=model_name,
        use_demucs=use_demucs,
        use_vad=use_vad,
        device_spec=runtime_policy.device_spec,
        device_type=runtime_policy.device_type,
        precision_candidates=runtime_policy.precision_candidates,
        memory_tier=runtime_policy.memory_tier,
    )
    adapter = resolve_transcription_backend_adapter(backend_id)
    report = adapter.check_compatibility(
        runtime_request=runtime_request,
        settings=settings,
    )
    findings: list[DiagnosticFinding] = []
    for issue in report.functional_issues:
        if _is_advisory_faster_whisper_openmp_conflict(
            backend_id=backend_id,
            issue_code=issue.code,
        ):
            findings.append(
                DiagnosticFinding(
                    code=f"transcription_operational_{issue.code}",
                    severity="info",
                    message=issue.message,
                )
            )
            continue
        findings.append(
            DiagnosticFinding(
                code=f"transcription_functional_{issue.code}",
                severity="error",
                message=issue.message,
                blocking=True,
            )
        )
    for issue in report.operational_issues:
        findings.append(
            DiagnosticFinding(
                code=f"transcription_operational_{issue.code}",
                severity=_resolve_operational_issue_severity(issue.impact),
                message=issue.message,
            )
        )
    if include_noise_findings:
        for issue in report.noise_issues:
            findings.append(
                DiagnosticFinding(
                    code=f"transcription_noise_{issue.code}",
                    severity="info",
                    message=issue.message,
                )
            )
    return tuple(findings)


def _is_advisory_faster_whisper_openmp_conflict(
    *,
    backend_id: str,
    issue_code: str,
) -> bool:
    """Returns whether one faster-whisper OpenMP conflict is non-blocking in CLI runtime."""
    return (
        backend_id == "faster_whisper" and issue_code == FASTER_WHISPER_OPENMP_CONFLICT_ISSUE_CODE
    )


def _resolve_operational_issue_severity(
    issue_impact: CompatibilityIssueImpact,
) -> Literal["info", "warning"]:
    """Maps one operational compatibility impact to diagnostics severity."""
    if issue_impact == "informational":
        return "info"
    return "warning"


def _condense_message(message: str, *, max_chars: int) -> str:
    """Returns one single-line message summary bounded by max chars."""
    normalized = " ".join(message.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _check_dataset_registry_health(
    *,
    settings: AppConfig,
) -> tuple[DiagnosticFinding, ...]:
    """Collects dataset registry consistency findings for diagnostics."""

    issues = collect_dataset_registry_health_issues(settings=settings)
    return tuple(
        DiagnosticFinding(
            code=f"dataset_registry_{issue.code}",
            severity="error",
            message=f"[{issue.dataset_id}] {issue.message}",
        )
        for issue in issues
    )
