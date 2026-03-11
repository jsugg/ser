"""Default-profile recommendation policy helpers for transcription profiling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar


class _ProfileLike(Protocol):
    """Minimal profile contract required by recommendation policy."""

    @property
    def name(self) -> str: ...


class _ProfileBenchmarkSummaryLike(Protocol):
    """Benchmark summary contract required by recommendation policy."""

    @property
    def profile(self) -> _ProfileLike: ...

    @property
    def evaluated_samples(self) -> int: ...

    @property
    def error_message(self) -> str | None: ...

    @property
    def mean_accuracy(self) -> float: ...

    @property
    def average_latency_seconds(self) -> float: ...


class _AccuracyGateLike(Protocol):
    """Accuracy gate contract used during recommendation selection."""

    @property
    def minimum_mean_accuracy(self) -> float: ...


GateT = TypeVar("GateT")
SummaryT = TypeVar("SummaryT", bound=_ProfileBenchmarkSummaryLike)
RecommendationT = TypeVar("RecommendationT")


def derive_accuracy_gate(
    baseline_summary: _ProfileBenchmarkSummaryLike,
    *,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
    gate_factory: Callable[[float, float, float, float], GateT],
) -> GateT:
    """Derive the minimum acceptable accuracy gate from baseline results."""

    minimum_mean_accuracy = max(
        absolute_accuracy_floor,
        baseline_summary.mean_accuracy - maximum_accuracy_drop,
    )
    return gate_factory(
        baseline_summary.mean_accuracy,
        minimum_mean_accuracy,
        maximum_accuracy_drop,
        absolute_accuracy_floor,
    )


def recommend_default_profile(
    summaries: tuple[SummaryT, ...],
    gate: _AccuracyGateLike,
    *,
    minimum_speedup_ratio: float,
    minimum_required_samples: int,
    recommendation_factory: Callable[
        [str, str, bool, str, float, float, float, int],
        RecommendationT,
    ],
) -> RecommendationT:
    """Select a default profile only when it is faster and accuracy-safe."""

    baseline = summaries[0]
    if baseline.evaluated_samples < minimum_required_samples:
        return recommendation_factory(
            baseline.profile.name,
            baseline.profile.name,
            False,
            (
                "Insufficient sample size for safe default changes. "
                f"Need at least {minimum_required_samples} evaluated samples."
            ),
            baseline.mean_accuracy,
            baseline.average_latency_seconds,
            1.0,
            minimum_required_samples,
        )

    selected = baseline
    selected_speedup = 1.0

    for summary in summaries[1:]:
        if summary.error_message is not None or summary.evaluated_samples == 0:
            continue
        if summary.mean_accuracy < gate.minimum_mean_accuracy:
            continue
        if summary.average_latency_seconds <= 0.0:
            continue
        speedup = baseline.average_latency_seconds / summary.average_latency_seconds
        if speedup >= minimum_speedup_ratio and speedup > selected_speedup:
            selected = summary
            selected_speedup = speedup

    if selected.profile.name == baseline.profile.name:
        return recommendation_factory(
            baseline.profile.name,
            baseline.profile.name,
            False,
            (
                "No candidate met both the accuracy gate and required speedup; "
                "keep current defaults."
            ),
            baseline.mean_accuracy,
            baseline.average_latency_seconds,
            1.0,
            minimum_required_samples,
        )

    return recommendation_factory(
        baseline.profile.name,
        selected.profile.name,
        True,
        "Candidate met the accuracy gate and exceeded required speedup.",
        selected.mean_accuracy,
        selected.average_latency_seconds,
        selected_speedup,
        minimum_required_samples,
    )


__all__ = [
    "derive_accuracy_gate",
    "recommend_default_profile",
]
