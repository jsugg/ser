"""Runtime-calibration orchestration helpers for transcription profiling."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generic, TypeVar

SettingsT = TypeVar("SettingsT")
ProfileNameT = TypeVar("ProfileNameT", bound=str)
CandidateT = TypeVar("CandidateT")
MetricsT = TypeVar("MetricsT")
RecommendationT = TypeVar("RecommendationT")
RecommendationValueT = TypeVar("RecommendationValueT", bound=str)
ConfidenceT = TypeVar("ConfidenceT", bound=str)


@dataclass(frozen=True)
class RuntimeCalibrationExecution(Generic[RecommendationT]):
    """Runtime-calibration outputs ready for public result mapping."""

    recommendations: tuple[RecommendationT, ...]
    report_path: Path


def _utc_now() -> datetime:
    """Return the current UTC timestamp."""

    return datetime.now(tz=UTC)


def execute_runtime_calibration(
    *,
    active_settings: SettingsT,
    calibration_file: Path,
    language: str,
    iterations_per_profile: int,
    profile_names: tuple[ProfileNameT, ...],
    report_path: Path | None,
    build_runtime_calibration_settings: Callable[[SettingsT], SettingsT],
    settings_override: Callable[[SettingsT], AbstractContextManager[object]],
    runtime_calibration_candidates: Callable[[tuple[ProfileNameT, ...]], tuple[CandidateT, ...]],
    calibrate_candidate: Callable[[CandidateT, Path, str, int], MetricsT],
    derive_runtime_recommendation: Callable[
        [MetricsT],
        tuple[RecommendationValueT, ConfidenceT, str],
    ],
    recommendation_factory: Callable[
        [CandidateT, RecommendationValueT, ConfidenceT, str, MetricsT],
        RecommendationT,
    ],
    runtime_calibration_report_path: Callable[[SettingsT], Path],
    persist_profile_report: Callable[[Path, dict[str, object]], Path],
    serialize_recommendation: Callable[[RecommendationT], dict[str, object]],
    now_utc: Callable[[], datetime] = _utc_now,
) -> RuntimeCalibrationExecution[RecommendationT]:
    """Run calibration probes and persist their recommendation report."""

    if iterations_per_profile <= 0:
        raise ValueError("iterations_per_profile must be greater than zero.")
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration audio file not found: {calibration_file}")

    calibration_settings = build_runtime_calibration_settings(active_settings)
    recommendations: list[RecommendationT] = []
    with settings_override(calibration_settings):
        for candidate in runtime_calibration_candidates(profile_names):
            metrics = calibrate_candidate(
                candidate,
                calibration_file,
                language,
                iterations_per_profile,
            )
            recommendation, confidence, reason = derive_runtime_recommendation(metrics)
            recommendations.append(
                recommendation_factory(
                    candidate,
                    recommendation,
                    confidence,
                    reason,
                    metrics,
                )
            )

    output_path = (
        runtime_calibration_report_path(active_settings) if report_path is None else report_path
    )
    payload: dict[str, object] = {
        "created_at_utc": now_utc().isoformat(),
        "calibration_file": str(calibration_file),
        "iterations_per_profile": iterations_per_profile,
        "profiles": [
            serialize_recommendation(recommendation) for recommendation in recommendations
        ],
    }
    persisted_path = persist_profile_report(output_path, payload)

    return RuntimeCalibrationExecution(
        recommendations=tuple(recommendations),
        report_path=persisted_path,
    )


__all__ = [
    "RuntimeCalibrationExecution",
    "execute_runtime_calibration",
]
