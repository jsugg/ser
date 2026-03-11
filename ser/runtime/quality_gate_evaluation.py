"""Evaluation helpers for profile quality-gate temporal/latency metrics."""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type IndexArray = NDArray[np.int64]
type FoldIndices = tuple[IndexArray, IndexArray]
type LabeledAudioSample = tuple[str, str]


class SegmentLike(Protocol):
    """Minimal segment shape required by gate evaluation helpers."""

    @property
    def emotion(self) -> str: ...

    @property
    def start_seconds(self) -> float: ...

    @property
    def end_seconds(self) -> float: ...


type SegmentPredictor = Callable[[str], Sequence[SegmentLike]]
type CanonicalizeSegments = Callable[
    [Sequence[SegmentLike]],
    Sequence[SegmentLike],
]


class ComputeSerMetrics(Protocol):
    """Callable protocol for SER metric payload generation."""

    def __call__(
        self,
        *,
        y_true: Sequence[str],
        y_pred: Sequence[str],
    ) -> dict[str, object]: ...


@dataclass(frozen=True)
class NormalizedSegment:
    """Validated internal segment representation."""

    emotion: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ProfileEvaluationResult:
    """Normalized per-profile evaluation aggregates prior to report wrapping."""

    profile: str
    evaluated_clips: int
    failed_clips: int
    metrics: dict[str, object]
    latency_mean_seconds: float
    latency_median_seconds: float
    latency_p95_seconds: float
    segment_count_per_minute: float
    median_segment_duration_seconds: float


def normalize_segments(
    segments: Sequence[SegmentLike],
    *,
    canonicalize_segments: CanonicalizeSegments,
) -> list[NormalizedSegment]:
    """Normalizes runtime segments into a validated immutable representation."""
    return [
        NormalizedSegment(
            emotion=segment.emotion,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in canonicalize_segments(list(segments))
    ]


def segment_duration(segment: NormalizedSegment) -> float:
    """Returns non-negative segment duration in seconds."""
    return max(0.0, segment.end_seconds - segment.start_seconds)


def clip_label_from_segments(
    segments: Sequence[NormalizedSegment],
    *,
    unknown_label: str,
) -> str:
    """Returns a duration-weighted clip label derived from segment predictions."""
    if not segments:
        return unknown_label

    weighted_votes: dict[str, float] = defaultdict(float)
    for segment in segments:
        duration = segment_duration(segment)
        weighted_votes[segment.emotion] += duration if duration > 0.0 else 1e-6

    winner = min(weighted_votes, key=lambda label: (-weighted_votes[label], label))
    return winner


def clip_stability_metrics(
    segments: Sequence[NormalizedSegment],
) -> tuple[float, list[float]]:
    """Returns segment-count-per-minute and per-segment durations."""
    if not segments:
        return 0.0, []

    clip_start = min(segment.start_seconds for segment in segments)
    clip_end = max(segment.end_seconds for segment in segments)
    clip_duration = max(0.0, clip_end - clip_start)
    segment_count_per_minute = (
        (float(len(segments)) * 60.0) / clip_duration if clip_duration > 0.0 else 0.0
    )
    segment_durations = [
        duration
        for duration in (segment_duration(segment) for segment in segments)
        if duration > 0.0
    ]
    return segment_count_per_minute, segment_durations


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Returns nearest-rank percentile for a non-empty value sequence."""
    if not values:
        return 0.0
    if not 0.0 <= percentile_value <= 1.0:
        raise ValueError("percentile must be between 0 and 1.")
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        int(round(percentile_value * float(len(sorted_values) - 1))),
    )
    return float(sorted_values[index])


def evaluate_profile(
    *,
    profile_name: str,
    samples: Sequence[LabeledAudioSample],
    folds: Sequence[FoldIndices],
    predictor: SegmentPredictor,
    unknown_label: str,
    canonicalize_segments: CanonicalizeSegments,
    compute_ser_metrics: ComputeSerMetrics,
    progress_every: int | None = None,
) -> ProfileEvaluationResult:
    """Evaluates one profile over shared grouped folds."""
    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    segment_counts_per_minute: list[float] = []
    segment_durations: list[float] = []
    failed_clips = 0
    processed_clips = 0
    total_clips = sum(int(test_indices.size) for _, test_indices in folds)

    for _, test_indices in folds:
        for sample_index in test_indices.tolist():
            audio_path, expected_label = samples[int(sample_index)]
            start_time = time.perf_counter()
            try:
                segments = predictor(audio_path)
            except Exception:
                failed_clips += 1
                processed_clips += 1
                if (
                    progress_every is not None
                    and progress_every > 0
                    and processed_clips % progress_every == 0
                ):
                    print(
                        f"[quality-gate:{profile_name}] "
                        f"{processed_clips}/{total_clips} clips "
                        f"(failed={failed_clips})",
                        flush=True,
                    )
                continue
            latencies.append(time.perf_counter() - start_time)

            normalized_segments = normalize_segments(
                segments,
                canonicalize_segments=canonicalize_segments,
            )
            predicted_label = clip_label_from_segments(
                normalized_segments,
                unknown_label=unknown_label,
            )
            segments_per_minute, durations = clip_stability_metrics(normalized_segments)
            segment_counts_per_minute.append(segments_per_minute)
            segment_durations.extend(durations)
            y_true.append(expected_label)
            y_pred.append(predicted_label)
            processed_clips += 1
            if (
                progress_every is not None
                and progress_every > 0
                and processed_clips % progress_every == 0
            ):
                print(
                    f"[quality-gate:{profile_name}] "
                    f"{processed_clips}/{total_clips} clips "
                    f"(failed={failed_clips})",
                    flush=True,
                )

    if not y_true:
        raise RuntimeError(f"Profile '{profile_name}' produced no successful clip predictions.")

    metrics = compute_ser_metrics(y_true=y_true, y_pred=y_pred)
    return ProfileEvaluationResult(
        profile=profile_name,
        evaluated_clips=len(y_true),
        failed_clips=failed_clips,
        metrics=metrics,
        latency_mean_seconds=float(statistics.fmean(latencies)),
        latency_median_seconds=float(statistics.median(latencies)),
        latency_p95_seconds=percentile(latencies, 0.95),
        segment_count_per_minute=(
            float(statistics.fmean(segment_counts_per_minute)) if segment_counts_per_minute else 0.0
        ),
        median_segment_duration_seconds=(
            float(statistics.median(segment_durations)) if segment_durations else 0.0
        ),
    )
