"""Policy helpers for quality-gate threshold and profile comparison logic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


class ThresholdsLike(Protocol):
    """Structural contract for quality-gate threshold values."""

    @property
    def minimum_uar_delta(self) -> float: ...

    @property
    def minimum_macro_f1_delta(self) -> float: ...

    @property
    def maximum_medium_segments_per_minute(self) -> float | None: ...

    @property
    def minimum_medium_median_segment_duration_seconds(self) -> float | None: ...


class TemporalStabilityLike(Protocol):
    """Structural contract for temporal-stability metrics."""

    @property
    def segment_count_per_minute(self) -> float: ...

    @property
    def median_segment_duration_seconds(self) -> float: ...


class ProfileSummaryLike(Protocol):
    """Structural contract for profile summary values used in comparison."""

    @property
    def metrics(self) -> Mapping[str, object]: ...

    @property
    def temporal_stability(self) -> TemporalStabilityLike: ...


@dataclass(frozen=True)
class ProfileComparisonResult:
    """Normalized comparison result used by quality-gate wrappers."""

    medium_minus_fast_uar: float
    medium_minus_fast_macro_f1: float
    medium_segments_per_minute: float
    medium_median_segment_duration_seconds: float
    passes_quality_gate: bool
    failure_reasons: tuple[str, ...]


def metric_as_float(metrics: Mapping[str, object], key: str) -> float:
    """Reads one numeric metric from a metrics payload with validation."""
    value = metrics.get(key)
    if not isinstance(value, float | int):
        raise ValueError(f"metrics payload is missing numeric key: {key}")
    return float(value)


def validate_thresholds(thresholds: ThresholdsLike) -> None:
    """Validates quality-gate threshold bounds."""
    if not math.isfinite(thresholds.minimum_uar_delta):
        raise ValueError("minimum_uar_delta must be finite.")
    if thresholds.minimum_uar_delta < 0.0:
        raise ValueError("minimum_uar_delta must be >= 0.")
    if not math.isfinite(thresholds.minimum_macro_f1_delta):
        raise ValueError("minimum_macro_f1_delta must be finite.")
    if thresholds.minimum_macro_f1_delta < 0.0:
        raise ValueError("minimum_macro_f1_delta must be >= 0.")
    if thresholds.maximum_medium_segments_per_minute is not None:
        if not math.isfinite(thresholds.maximum_medium_segments_per_minute):
            raise ValueError("maximum_medium_segments_per_minute must be finite.")
        if thresholds.maximum_medium_segments_per_minute <= 0.0:
            raise ValueError("maximum_medium_segments_per_minute must be positive.")
    if thresholds.minimum_medium_median_segment_duration_seconds is not None:
        if not math.isfinite(thresholds.minimum_medium_median_segment_duration_seconds):
            raise ValueError(
                "minimum_medium_median_segment_duration_seconds must be finite."
            )
        if thresholds.minimum_medium_median_segment_duration_seconds < 0.0:
            raise ValueError(
                "minimum_medium_median_segment_duration_seconds must be >= 0."
            )


def compare_profiles(
    *,
    fast: ProfileSummaryLike,
    medium: ProfileSummaryLike,
    thresholds: ThresholdsLike,
) -> ProfileComparisonResult:
    """Compares medium versus fast against configured quality thresholds."""
    validate_thresholds(thresholds)
    fast_uar = metric_as_float(fast.metrics, "uar")
    medium_uar = metric_as_float(medium.metrics, "uar")
    fast_macro_f1 = metric_as_float(fast.metrics, "macro_f1")
    medium_macro_f1 = metric_as_float(medium.metrics, "macro_f1")
    medium_segments_per_minute = medium.temporal_stability.segment_count_per_minute
    medium_median_segment_duration = (
        medium.temporal_stability.median_segment_duration_seconds
    )

    uar_delta = medium_uar - fast_uar
    macro_f1_delta = medium_macro_f1 - fast_macro_f1
    failure_reasons: list[str] = []
    if uar_delta < thresholds.minimum_uar_delta:
        failure_reasons.append(
            "medium_minus_fast_uar below minimum threshold: "
            f"{uar_delta:.4f} < {thresholds.minimum_uar_delta:.4f}"
        )
    if macro_f1_delta < thresholds.minimum_macro_f1_delta:
        failure_reasons.append(
            "medium_minus_fast_macro_f1 below minimum threshold: "
            f"{macro_f1_delta:.4f} < {thresholds.minimum_macro_f1_delta:.4f}"
        )
    if thresholds.maximum_medium_segments_per_minute is not None:
        if medium_segments_per_minute > thresholds.maximum_medium_segments_per_minute:
            failure_reasons.append(
                "medium_segments_per_minute exceeds maximum threshold: "
                f"{medium_segments_per_minute:.4f} > "
                f"{thresholds.maximum_medium_segments_per_minute:.4f}"
            )
    if thresholds.minimum_medium_median_segment_duration_seconds is not None:
        if (
            medium_median_segment_duration
            < thresholds.minimum_medium_median_segment_duration_seconds
        ):
            failure_reasons.append(
                "medium_median_segment_duration_seconds below minimum threshold: "
                f"{medium_median_segment_duration:.4f} < "
                f"{thresholds.minimum_medium_median_segment_duration_seconds:.4f}"
            )

    return ProfileComparisonResult(
        medium_minus_fast_uar=uar_delta,
        medium_minus_fast_macro_f1=macro_f1_delta,
        medium_segments_per_minute=medium_segments_per_minute,
        medium_median_segment_duration_seconds=medium_median_segment_duration,
        passes_quality_gate=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
    )
