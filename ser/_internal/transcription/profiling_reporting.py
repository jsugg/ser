"""Reporting helpers for transcription profiling and runtime calibration flows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ser.transcript.profiling import ProfilingResult, RuntimeCalibrationResult


def persist_profile_report(path: Path, payload: dict[str, object]) -> Path:
    """Persists one profiling payload as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def profiling_summary_lines(result: ProfilingResult) -> tuple[str, ...]:
    """Builds a concise default-profiling summary for terminal output."""
    lines = [
        "",
        "Transcription default profiling summary",
        f"Reference files: {result.reference_files}",
        (
            "Minimum mean accuracy gate: "
            f"{result.gate.minimum_mean_accuracy:.4f} "
            f"(baseline={result.gate.baseline_mean_accuracy:.4f}, "
            f"allowed_drop={result.gate.maximum_accuracy_drop:.4f}, "
            f"floor={result.gate.absolute_accuracy_floor:.4f})"
        ),
        (
            "profile | mean_accuracy | mean_wer | exact_match_rate | "
            "avg_latency_s | failed | error"
        ),
    ]
    for summary in result.summaries:
        lines.append(
            f"{summary.profile.name} | "
            f"{summary.mean_accuracy:.4f} | "
            f"{summary.mean_word_error_rate:.4f} | "
            f"{summary.exact_match_rate:.4f} | "
            f"{summary.average_latency_seconds:.4f} | "
            f"{summary.failed_samples} | "
            f"{summary.error_message or '-'}"
        )
    lines.extend(
        [
            "",
            "Recommendation",
            f"Change defaults: {result.recommendation.should_change_defaults}",
            f"Selected profile: {result.recommendation.selected_profile}",
            f"Reason: {result.recommendation.reason}",
            f"Report: {result.report_path}",
        ]
    )
    return tuple(lines)


def runtime_calibration_summary_lines(
    result: RuntimeCalibrationResult,
) -> tuple[str, ...]:
    """Builds concise runtime-calibration summary lines for terminal output."""
    lines = [
        "",
        "Transcription runtime calibration summary",
        (
            "profile | backend | model | recommendation | confidence | "
            "mps_loaded | mps_completed | failover | failed | mean_latency_s | reason"
        ),
    ]
    for recommendation in result.recommendations:
        metrics = recommendation.metrics
        lines.append(
            f"{recommendation.profile.source_profile} | "
            f"{recommendation.profile.backend_id} | "
            f"{recommendation.profile.model_name} | "
            f"{recommendation.recommendation} | "
            f"{recommendation.confidence} | "
            f"{metrics.mps_loaded_runs}/{metrics.iterations} | "
            f"{metrics.mps_completed_runs}/{metrics.iterations} | "
            f"{metrics.mps_to_cpu_failover_runs} | "
            f"{metrics.failed_runs} | "
            f"{metrics.mean_latency_seconds:.4f} | "
            f"{recommendation.reason}"
        )
    lines.append("")
    lines.append(f"Report: {result.report_path}")
    return tuple(lines)


__all__ = [
    "persist_profile_report",
    "profiling_summary_lines",
    "runtime_calibration_summary_lines",
]
