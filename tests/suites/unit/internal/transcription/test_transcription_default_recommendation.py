"""Tests for internal default-profile recommendation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ser._internal.transcription import default_recommendation as helpers


@dataclass(frozen=True)
class _Profile:
    name: str


@dataclass(frozen=True)
class _Summary:
    profile: _Profile
    evaluated_samples: int
    error_message: str | None
    mean_accuracy: float
    average_latency_seconds: float


@dataclass(frozen=True)
class _Gate:
    baseline_mean_accuracy: float
    minimum_mean_accuracy: float
    maximum_accuracy_drop: float
    absolute_accuracy_floor: float


@dataclass(frozen=True)
class _Recommendation:
    baseline_profile: str
    selected_profile: str
    should_change_defaults: bool
    reason: str
    selected_mean_accuracy: float
    selected_average_latency_seconds: float
    selected_speedup_vs_baseline: float
    minimum_required_samples: int


def test_derive_accuracy_gate_applies_floor_and_allowed_drop() -> None:
    """Gate helper should choose the stricter of absolute floor and baseline drop."""

    gate = helpers.derive_accuracy_gate(
        _Summary(
            profile=_Profile(name="baseline"),
            evaluated_samples=12,
            error_message=None,
            mean_accuracy=0.94,
            average_latency_seconds=4.0,
        ),
        absolute_accuracy_floor=0.90,
        maximum_accuracy_drop=0.02,
        gate_factory=_Gate,
    )

    assert gate.minimum_mean_accuracy == pytest.approx(0.92)


def test_recommend_default_profile_selects_fastest_passing_candidate() -> None:
    """Recommendation helper should choose the fastest candidate that clears the gate."""

    baseline = _Summary(
        profile=_Profile(name="baseline"),
        evaluated_samples=20,
        error_message=None,
        mean_accuracy=0.96,
        average_latency_seconds=4.0,
    )
    faster = _Summary(
        profile=_Profile(name="candidate-fast"),
        evaluated_samples=20,
        error_message=None,
        mean_accuracy=0.95,
        average_latency_seconds=2.5,
    )
    accurate_but_slow = _Summary(
        profile=_Profile(name="candidate-slow"),
        evaluated_samples=20,
        error_message=None,
        mean_accuracy=0.97,
        average_latency_seconds=4.5,
    )

    recommendation = helpers.recommend_default_profile(
        (baseline, faster, accurate_but_slow),
        _Gate(
            baseline_mean_accuracy=0.96,
            minimum_mean_accuracy=0.94,
            maximum_accuracy_drop=0.02,
            absolute_accuracy_floor=0.90,
        ),
        minimum_speedup_ratio=1.10,
        minimum_required_samples=1,
        recommendation_factory=_Recommendation,
    )

    assert recommendation.should_change_defaults is True
    assert recommendation.selected_profile == "candidate-fast"


def test_recommend_default_profile_keeps_baseline_for_low_sample_runs() -> None:
    """Recommendation helper should keep the baseline when sample count is too low."""

    baseline = _Summary(
        profile=_Profile(name="baseline"),
        evaluated_samples=12,
        error_message=None,
        mean_accuracy=0.96,
        average_latency_seconds=4.0,
    )
    faster = _Summary(
        profile=_Profile(name="candidate-fast"),
        evaluated_samples=12,
        error_message=None,
        mean_accuracy=0.95,
        average_latency_seconds=2.0,
    )

    recommendation = helpers.recommend_default_profile(
        (baseline, faster),
        _Gate(
            baseline_mean_accuracy=0.96,
            minimum_mean_accuracy=0.94,
            maximum_accuracy_drop=0.02,
            absolute_accuracy_floor=0.90,
        ),
        minimum_speedup_ratio=1.10,
        minimum_required_samples=100,
        recommendation_factory=_Recommendation,
    )

    assert recommendation.should_change_defaults is False
    assert recommendation.selected_profile == "baseline"


def test_recommend_default_profile_keeps_baseline_when_gate_fails() -> None:
    """Recommendation helper should reject candidates below the accuracy gate."""

    baseline = _Summary(
        profile=_Profile(name="baseline"),
        evaluated_samples=20,
        error_message=None,
        mean_accuracy=0.96,
        average_latency_seconds=4.0,
    )
    low_accuracy = _Summary(
        profile=_Profile(name="candidate-low-accuracy"),
        evaluated_samples=20,
        error_message=None,
        mean_accuracy=0.80,
        average_latency_seconds=1.5,
    )

    recommendation = helpers.recommend_default_profile(
        (baseline, low_accuracy),
        _Gate(
            baseline_mean_accuracy=0.96,
            minimum_mean_accuracy=0.94,
            maximum_accuracy_drop=0.02,
            absolute_accuracy_floor=0.90,
        ),
        minimum_speedup_ratio=1.10,
        minimum_required_samples=1,
        recommendation_factory=_Recommendation,
    )

    assert recommendation.should_change_defaults is False
    assert recommendation.selected_profile == "baseline"
