"""Diagnostics package exports."""

from .command import run_doctor_command
from .domain import (
    DiagnosticFinding,
    DiagnosticReport,
    DiagnosticSeverity,
    PreflightMode,
)
from .service import (
    apply_profile_override_for_diagnostics,
    format_report_brief,
    format_report_json,
    format_report_text,
    format_startup_preflight_one_liner,
    parse_preflight_mode,
    run_doctor_diagnostics,
    run_startup_preflight,
    should_fail_preflight,
)

__all__ = [
    "DiagnosticFinding",
    "DiagnosticReport",
    "DiagnosticSeverity",
    "PreflightMode",
    "apply_profile_override_for_diagnostics",
    "format_report_brief",
    "format_report_json",
    "format_startup_preflight_one_liner",
    "format_report_text",
    "parse_preflight_mode",
    "run_doctor_command",
    "run_doctor_diagnostics",
    "run_startup_preflight",
    "should_fail_preflight",
]
