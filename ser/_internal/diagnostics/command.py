"""CLI command helpers for SER runtime diagnostics."""

from __future__ import annotations

import argparse
from typing import cast

from ser.config import AppConfig, reload_settings
from ser.diagnostics.service import (
    apply_profile_override_for_diagnostics,
    format_report_json,
    format_report_text,
    run_doctor_diagnostics,
)
from ser.profiles import ProfileName

_PROFILE_CHOICES: tuple[ProfileName, ...] = (
    "fast",
    "medium",
    "accurate",
    "accurate-research",
)


def _resolve_boundary_settings(settings: AppConfig | None = None) -> AppConfig:
    """Returns explicit settings or reloads a diagnostics CLI snapshot."""
    return settings if settings is not None else reload_settings()


def run_doctor_command(argv: list[str], *, settings: AppConfig | None = None) -> int:
    """Runs `ser doctor ...` command."""
    parser = argparse.ArgumentParser(prog="ser doctor")
    parser.add_argument(
        "--profile",
        choices=_PROFILE_CHOICES,
        default=None,
        help=("Profile context for diagnostics " "(fast, medium, accurate, accurate-research)."),
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for diagnostics findings.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when warning or error findings are present.",
    )
    parser.add_argument(
        "--include-noise-findings",
        action="store_true",
        help="Include informational dependency-noise findings in output.",
    )
    parser.add_argument(
        "--skip-transcription-checks",
        action="store_true",
        help="Skip transcription backend checks.",
    )
    args = parser.parse_args(argv)

    active_settings = _resolve_boundary_settings(settings)
    active_settings = apply_profile_override_for_diagnostics(
        active_settings,
        profile=cast(ProfileName | None, args.profile),
    )
    report = run_doctor_diagnostics(
        settings=active_settings,
        include_transcription_checks=not bool(args.skip_transcription_checks),
        include_noise_findings=bool(args.include_noise_findings),
    )
    formatted_output = (
        format_report_json(report) if args.format == "json" else format_report_text(report)
    )
    print(formatted_output)
    if report.has_blocking_findings or report.has_error:
        return 2
    if args.strict and report.has_warning_or_higher:
        return 1
    return 0
