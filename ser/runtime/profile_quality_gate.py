"""Cross-profile quality gate harness for fast-versus-medium evaluation."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import statistics
import time
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

import ser.config as config
from ser.config import get_settings
from ser.data.data_loader import extract_ravdess_speaker_id_from_path
from ser.models.emotion_model import LoadedModel, load_model, predict_emotions
from ser.repr import XLSRBackend
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_inference import run_medium_inference
from ser.train.eval import grouped_train_test_split, speaker_independent_cv
from ser.train.metrics import compute_ser_metrics
from ser.utils.segment_canonicalization import canonicalize_segments

type IndexArray = NDArray[np.int64]
type FoldIndices = tuple[IndexArray, IndexArray]
type LabeledAudioSample = tuple[str, str]


class SegmentLike(Protocol):
    """Minimal segment shape required by gate evaluation."""

    @property
    def emotion(self) -> str:
        """Emotion label assigned to the segment."""
        ...

    @property
    def start_seconds(self) -> float:
        """Segment start timestamp in seconds."""
        ...

    @property
    def end_seconds(self) -> float:
        """Segment end timestamp in seconds."""
        ...


type SegmentPredictor = Callable[[str], Sequence[SegmentLike]]


@dataclass(frozen=True)
class QualityGateThresholds:
    """Pass/fail thresholds for medium-versus-fast quality gate checks."""

    minimum_uar_delta: float = 0.0
    minimum_macro_f1_delta: float = 0.0
    maximum_medium_segments_per_minute: float | None = None
    minimum_medium_median_segment_duration_seconds: float | None = None


@dataclass(frozen=True)
class LatencySummary:
    """Latency aggregate statistics for one profile."""

    mean_seconds: float
    median_seconds: float
    p95_seconds: float


@dataclass(frozen=True)
class TemporalStabilitySummary:
    """Temporal stability aggregates for one profile."""

    segment_count_per_minute: float
    median_segment_duration_seconds: float


@dataclass(frozen=True)
class ProfileQualitySummary:
    """Per-profile evaluation results over shared folds."""

    profile: str
    evaluated_clips: int
    failed_clips: int
    metrics: dict[str, object]
    temporal_stability: TemporalStabilitySummary
    latency: LatencySummary


@dataclass(frozen=True)
class QualityGateComparison:
    """Cross-profile comparison and pass/fail decision."""

    medium_minus_fast_uar: float
    medium_minus_fast_macro_f1: float
    medium_segments_per_minute: float
    medium_median_segment_duration_seconds: float
    passes_quality_gate: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True)
class GroupedEvaluationSummary:
    """Speaker-grouped evaluation diagnostics for report traceability."""

    unique_speakers: int
    min_samples_per_speaker: int
    max_samples_per_speaker: int
    fold_speaker_overlap_counts: tuple[int, ...]


@dataclass(frozen=True)
class ProfileQualityGateReport:
    """Full quality-gate report artifact."""

    generated_at_utc: str
    dataset_glob_pattern: str
    fold_strategy: str
    folds_evaluated: int
    grouped_evaluation: GroupedEvaluationSummary
    thresholds: QualityGateThresholds
    fast: ProfileQualitySummary
    medium: ProfileQualitySummary
    comparison: QualityGateComparison


def _extract_emotion_code(file_name: str) -> str | None:
    """Extracts the RAVDESS emotion code token from a filename."""
    parts = Path(file_name).name.split("-")
    return parts[2] if len(parts) >= 3 else None


def collect_labeled_samples(
    *,
    dataset_glob_pattern: str,
    emotion_map: Mapping[str, str],
    max_files: int | None = None,
) -> list[LabeledAudioSample]:
    """Collects labeled samples from dataset glob pattern.

    Args:
        dataset_glob_pattern: Glob expression for dataset audio files.
        emotion_map: Mapping from filename emotion-code to canonical label.
        max_files: Optional maximum number of collected labeled samples.

    Returns:
        Sorted `(file_path, label)` pairs accepted by the configured emotion map.

    Raises:
        ValueError: If the glob yields no usable labeled samples.
    """
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be greater than zero when provided.")

    resolved_files = sorted(glob.glob(dataset_glob_pattern))
    samples: list[LabeledAudioSample] = []
    for file_path in resolved_files:
        emotion_code = _extract_emotion_code(file_path)
        if emotion_code is None:
            continue
        label = emotion_map.get(emotion_code)
        if label is None:
            continue
        samples.append((file_path, label))
        if max_files is not None and len(samples) >= max_files:
            break

    if not samples:
        raise ValueError(
            "No labeled audio samples were collected from dataset_glob_pattern."
        )
    return samples


def _resolve_speaker_ids(samples: Sequence[LabeledAudioSample]) -> list[str]:
    """Resolves one speaker-id per sample and enforces grouped-eval readiness."""
    speaker_ids: list[str] = []
    for audio_path, _ in samples:
        speaker_id = extract_ravdess_speaker_id_from_path(audio_path)
        if speaker_id is None:
            raise ValueError(
                "Speaker ID extraction failed for sample path; grouped evaluation "
                f"requires RAVDESS-style filenames: {audio_path}"
            )
        speaker_ids.append(speaker_id)
    if len(set(speaker_ids)) < 2:
        raise ValueError("At least two distinct speaker IDs are required.")
    return speaker_ids


def _build_grouped_folds(
    *,
    labels: Sequence[str],
    speaker_ids: Sequence[str],
    n_splits: int,
    random_state: int,
    fallback_test_size: float,
) -> tuple[str, tuple[FoldIndices, ...]]:
    """Builds grouped folds, falling back to one grouped holdout when necessary."""
    if n_splits < 2:
        raise ValueError("n_splits must be greater than or equal to 2.")
    if not 0.0 < fallback_test_size < 1.0:
        raise ValueError("fallback_test_size must be between 0 and 1.")

    features = np.zeros((len(labels), 1), dtype=np.float64)
    try:
        folds = speaker_independent_cv(
            features,
            labels,
            speaker_ids,
            n_splits=n_splits,
            random_state=random_state,
        )
        return "stratified_group_kfold", folds
    except ValueError:
        grouped_split = grouped_train_test_split(
            features,
            labels,
            speaker_ids,
            test_size=fallback_test_size,
            random_state=random_state,
        )
        single_fold: FoldIndices = (
            grouped_split.train_indices,
            grouped_split.test_indices,
        )
        return "group_shuffle_holdout", (single_fold,)


def _build_grouped_evaluation_summary(
    *,
    speaker_ids: Sequence[str],
    folds: Sequence[FoldIndices],
) -> GroupedEvaluationSummary:
    """Builds grouped-evaluation diagnostics and enforces zero speaker overlap."""
    if not speaker_ids:
        raise ValueError("speaker_ids cannot be empty.")
    counts_by_speaker: defaultdict[str, int] = defaultdict(int)
    for speaker_id in speaker_ids:
        counts_by_speaker[str(speaker_id)] += 1

    overlap_counts: list[int] = []
    for train_indices, test_indices in folds:
        train_speakers = {speaker_ids[int(index)] for index in train_indices.tolist()}
        test_speakers = {speaker_ids[int(index)] for index in test_indices.tolist()}
        overlap_count = len(train_speakers.intersection(test_speakers))
        if overlap_count > 0:
            raise RuntimeError(
                "Grouped quality-gate folds contain train/test speaker overlap."
            )
        overlap_counts.append(overlap_count)

    samples_per_speaker = tuple(counts_by_speaker.values())
    return GroupedEvaluationSummary(
        unique_speakers=len(counts_by_speaker),
        min_samples_per_speaker=min(samples_per_speaker),
        max_samples_per_speaker=max(samples_per_speaker),
        fold_speaker_overlap_counts=tuple(overlap_counts),
    )


@dataclass(frozen=True)
class _NormalizedSegment:
    """Validated internal segment representation."""

    emotion: str
    start_seconds: float
    end_seconds: float


def _normalize_segments(segments: Sequence[SegmentLike]) -> list[_NormalizedSegment]:
    """Normalizes runtime segments into a validated immutable representation."""
    return [
        _NormalizedSegment(
            emotion=segment.emotion,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in canonicalize_segments(list(segments))
    ]


def _segment_duration(segment: _NormalizedSegment) -> float:
    """Returns non-negative segment duration in seconds."""
    return max(0.0, segment.end_seconds - segment.start_seconds)


def _clip_label_from_segments(
    segments: Sequence[_NormalizedSegment], *, unknown_label: str
) -> str:
    """Returns a duration-weighted clip label derived from segment predictions."""
    if not segments:
        return unknown_label

    weighted_votes: dict[str, float] = defaultdict(float)
    for segment in segments:
        duration = _segment_duration(segment)
        weighted_votes[segment.emotion] += duration if duration > 0.0 else 1e-6

    winner = min(weighted_votes, key=lambda label: (-weighted_votes[label], label))
    return winner


def _clip_stability_metrics(
    segments: Sequence[_NormalizedSegment],
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
        for duration in (_segment_duration(segment) for segment in segments)
        if duration > 0.0
    ]
    return segment_count_per_minute, segment_durations


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Returns nearest-rank percentile for a non-empty value sequence."""
    if not values:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1.")
    sorted_values = sorted(values)
    index = min(
        len(sorted_values) - 1,
        int(round(percentile * float(len(sorted_values) - 1))),
    )
    return float(sorted_values[index])


def _evaluate_profile(
    *,
    profile_name: str,
    samples: Sequence[LabeledAudioSample],
    folds: Sequence[FoldIndices],
    predictor: SegmentPredictor,
    unknown_label: str,
    progress_every: int | None = None,
) -> ProfileQualitySummary:
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

            normalized_segments = _normalize_segments(segments)
            predicted_label = _clip_label_from_segments(
                normalized_segments,
                unknown_label=unknown_label,
            )
            segments_per_minute, durations = _clip_stability_metrics(
                normalized_segments
            )
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
        raise RuntimeError(
            f"Profile '{profile_name}' produced no successful clip predictions."
        )

    metrics = compute_ser_metrics(y_true=y_true, y_pred=y_pred)
    latency = LatencySummary(
        mean_seconds=float(statistics.fmean(latencies)),
        median_seconds=float(statistics.median(latencies)),
        p95_seconds=_percentile(latencies, 0.95),
    )
    temporal_stability = TemporalStabilitySummary(
        segment_count_per_minute=(
            float(statistics.fmean(segment_counts_per_minute))
            if segment_counts_per_minute
            else 0.0
        ),
        median_segment_duration_seconds=(
            float(statistics.median(segment_durations)) if segment_durations else 0.0
        ),
    )
    return ProfileQualitySummary(
        profile=profile_name,
        evaluated_clips=len(y_true),
        failed_clips=failed_clips,
        metrics=metrics,
        temporal_stability=temporal_stability,
        latency=latency,
    )


def _metric_as_float(metrics: Mapping[str, object], key: str) -> float:
    """Reads one numeric metric from a metrics payload with validation."""
    value = metrics.get(key)
    if not isinstance(value, float | int):
        raise ValueError(f"metrics payload is missing numeric key: {key}")
    return float(value)


def _validate_thresholds(thresholds: QualityGateThresholds) -> None:
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


def _compare_profiles(
    *,
    fast: ProfileQualitySummary,
    medium: ProfileQualitySummary,
    thresholds: QualityGateThresholds,
) -> QualityGateComparison:
    """Compares medium versus fast against configured quality thresholds."""
    _validate_thresholds(thresholds)
    fast_uar = _metric_as_float(fast.metrics, "uar")
    medium_uar = _metric_as_float(medium.metrics, "uar")
    fast_macro_f1 = _metric_as_float(fast.metrics, "macro_f1")
    medium_macro_f1 = _metric_as_float(medium.metrics, "macro_f1")
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

    return QualityGateComparison(
        medium_minus_fast_uar=uar_delta,
        medium_minus_fast_macro_f1=macro_f1_delta,
        medium_segments_per_minute=medium_segments_per_minute,
        medium_median_segment_duration_seconds=medium_median_segment_duration,
        passes_quality_gate=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
    )


def evaluate_profile_quality_gate(
    *,
    samples: Sequence[LabeledAudioSample],
    fast_predictor: SegmentPredictor,
    medium_predictor: SegmentPredictor,
    dataset_glob_pattern: str,
    thresholds: QualityGateThresholds,
    n_splits: int = 5,
    random_state: int = 42,
    fallback_test_size: float = 0.25,
    unknown_label: str = "unknown",
    progress_every: int | None = None,
) -> ProfileQualityGateReport:
    """Evaluates fast and medium profiles on shared grouped folds.

    Args:
        samples: Labeled `(audio_path, label)` tuples.
        fast_predictor: Inference callable for fast profile.
        medium_predictor: Inference callable for medium profile.
        dataset_glob_pattern: Source dataset glob used for report traceability.
        thresholds: Quality-gate acceptance thresholds.
        n_splits: Number of grouped CV folds.
        random_state: Deterministic fold seed.
        fallback_test_size: Holdout split ratio when grouped CV is not feasible.
        unknown_label: Label emitted when segment predictions are empty.
        progress_every: Optional clip cadence for progress logs per profile.

    Returns:
        A full quality-gate report with per-profile metrics and gate decision.
    """
    if len(samples) < 2:
        raise ValueError("At least two labeled samples are required.")

    labels = [label for _, label in samples]
    speaker_ids = _resolve_speaker_ids(samples)
    fold_strategy, folds = _build_grouped_folds(
        labels=labels,
        speaker_ids=speaker_ids,
        n_splits=n_splits,
        random_state=random_state,
        fallback_test_size=fallback_test_size,
    )
    grouped_evaluation = _build_grouped_evaluation_summary(
        speaker_ids=speaker_ids,
        folds=folds,
    )
    fast_summary = _evaluate_profile(
        profile_name="fast",
        samples=samples,
        folds=folds,
        predictor=fast_predictor,
        unknown_label=unknown_label,
        progress_every=progress_every,
    )
    medium_summary = _evaluate_profile(
        profile_name="medium",
        samples=samples,
        folds=folds,
        predictor=medium_predictor,
        unknown_label=unknown_label,
        progress_every=progress_every,
    )
    comparison = _compare_profiles(
        fast=fast_summary,
        medium=medium_summary,
        thresholds=thresholds,
    )
    return ProfileQualityGateReport(
        generated_at_utc=datetime.now(tz=UTC).isoformat(),
        dataset_glob_pattern=dataset_glob_pattern,
        fold_strategy=fold_strategy,
        folds_evaluated=len(folds),
        grouped_evaluation=grouped_evaluation,
        thresholds=thresholds,
        fast=fast_summary,
        medium=medium_summary,
        comparison=comparison,
    )


@contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily applies environment overrides and refreshes runtime settings."""
    previous_values: dict[str, str | None] = {key: os.getenv(key) for key in overrides}
    for key, value in overrides.items():
        os.environ[key] = value
    config.reload_settings()
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
        config.reload_settings()


def _build_fast_predictor(
    *,
    model_file_name: str,
    secure_model_file_name: str,
    training_report_file_name: str,
) -> SegmentPredictor:
    """Builds fast predictor with deterministic model artifact selection."""
    overrides = {
        "SER_MODEL_FILE_NAME": model_file_name,
        "SER_SECURE_MODEL_FILE_NAME": secure_model_file_name,
        "SER_TRAINING_REPORT_FILE_NAME": training_report_file_name,
    }
    loaded_model: LoadedModel
    with _temporary_env(overrides):
        loaded_model = load_model()

    def _predict(audio_path: str) -> Sequence[SegmentLike]:
        return predict_emotions(audio_path, loaded_model=loaded_model)

    return _predict


def _build_medium_predictor(
    *,
    model_file_name: str,
    secure_model_file_name: str,
    training_report_file_name: str,
    language: str | None,
) -> SegmentPredictor:
    """Builds medium predictor with deterministic model artifact selection."""
    overrides = {
        "SER_MODEL_FILE_NAME": model_file_name,
        "SER_SECURE_MODEL_FILE_NAME": secure_model_file_name,
        "SER_TRAINING_REPORT_FILE_NAME": training_report_file_name,
    }
    loaded_model: LoadedModel
    settings = get_settings()
    default_language = settings.default_language
    backend = XLSRBackend(cache_dir=settings.models.huggingface_cache_root)
    with _temporary_env(overrides):
        loaded_model = load_model()

    def _predict(audio_path: str) -> Sequence[SegmentLike]:
        result = run_medium_inference(
            InferenceRequest(
                file_path=audio_path,
                language=language or default_language,
                save_transcript=False,
            ),
            settings,
            loaded_model=loaded_model,
            backend=backend,
            enforce_timeout=False,
            allow_retries=False,
        )
        return result.segments

    return _predict


def build_report_payload(report: ProfileQualityGateReport) -> dict[str, object]:
    """Converts quality-gate report dataclasses into a JSON-safe dictionary."""
    return asdict(report)


def enforce_quality_gate(
    report: ProfileQualityGateReport,
    *,
    require_pass: bool,
) -> None:
    """Raises a terminal error when gate enforcement is required and failed."""
    if not require_pass:
        return
    if report.comparison.passes_quality_gate:
        return
    reasons = "; ".join(report.comparison.failure_reasons)
    raise SystemExit(f"Quality gate failed: {reasons}")


def _configure_cli_noise_controls() -> None:
    """Suppresses non-actionable warning/log noise for long gate executions."""
    warnings.filterwarnings(
        "ignore",
        message=r"n_fft=\d+ is too large for input signal of length=.*",
        category=UserWarning,
        module=r"librosa\.core\.spectrum",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Trying to estimate tuning from empty frequency set\.",
        category=UserWarning,
        module=r"librosa\.core\.pitch",
    )
    logging.getLogger("ser.models.emotion_model").setLevel(logging.WARNING)
    logging.getLogger("ser.features.feature_extractor").setLevel(logging.ERROR)
    logging.getLogger("ser.runtime.medium_inference").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for profile quality gate evaluation."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="SER fast-vs-medium profile quality gate harness"
    )
    parser.add_argument(
        "--dataset-glob",
        default=settings.dataset.glob_pattern,
        help="Dataset glob pattern used for shared evaluation set.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files from dataset glob.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Grouped CV fold count before fallback to grouped holdout.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=settings.training.test_size,
        help="Grouped holdout test-size fallback when CV split is infeasible.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=settings.training.random_state,
        help="Deterministic seed for split and fallback behavior.",
    )
    parser.add_argument(
        "--fast-model-file-name",
        default=settings.models.model_file_name,
        help="Model artifact filename used for fast-profile inference.",
    )
    parser.add_argument(
        "--fast-secure-model-file-name",
        default=settings.models.secure_model_file_name,
        help="Secure model artifact filename used for fast-profile inference.",
    )
    parser.add_argument(
        "--fast-training-report-file-name",
        default=settings.models.training_report_file_name,
        help="Training report filename used for fast-profile feature-size hints.",
    )
    parser.add_argument(
        "--medium-model-file-name",
        required=True,
        help="Model artifact filename used for medium-profile inference.",
    )
    parser.add_argument(
        "--medium-secure-model-file-name",
        default=settings.models.secure_model_file_name,
        help="Secure model artifact filename used for medium-profile inference.",
    )
    parser.add_argument(
        "--medium-training-report-file-name",
        default=settings.models.training_report_file_name,
        help="Training report filename used for medium-profile feature-size hints.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language passed through to medium inference request.",
    )
    parser.add_argument(
        "--min-uar-delta",
        type=float,
        default=settings.quality_gate.min_uar_delta,
        help="Minimum required (medium - fast) UAR delta.",
    )
    parser.add_argument(
        "--min-macro-f1-delta",
        type=float,
        default=settings.quality_gate.min_macro_f1_delta,
        help="Minimum required (medium - fast) macro-F1 delta.",
    )
    parser.add_argument(
        "--max-medium-segments-per-minute",
        type=float,
        default=settings.quality_gate.max_medium_segments_per_minute,
        help="Optional upper bound for medium segment count per minute.",
    )
    parser.add_argument(
        "--min-medium-median-segment-duration",
        type=float,
        default=settings.quality_gate.min_medium_median_segment_duration_seconds,
        help="Optional lower bound for medium median segment duration in seconds.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for JSON quality-gate report.",
    )
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Exit with code 1 when medium profile does not pass thresholds.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Emit progress every N evaluated clips per profile (0 disables).",
    )
    return parser.parse_args()


def main() -> None:
    """Runs quality-gate harness and writes JSON report to stdout or disk."""
    _configure_cli_noise_controls()
    args = _parse_args()
    settings = get_settings()
    thresholds = QualityGateThresholds(
        minimum_uar_delta=args.min_uar_delta,
        minimum_macro_f1_delta=args.min_macro_f1_delta,
        maximum_medium_segments_per_minute=args.max_medium_segments_per_minute,
        minimum_medium_median_segment_duration_seconds=(
            args.min_medium_median_segment_duration
        ),
    )
    samples = collect_labeled_samples(
        dataset_glob_pattern=args.dataset_glob,
        emotion_map=settings.emotions,
        max_files=args.max_files,
    )
    fast_predictor = _build_fast_predictor(
        model_file_name=args.fast_model_file_name,
        secure_model_file_name=args.fast_secure_model_file_name,
        training_report_file_name=args.fast_training_report_file_name,
    )
    medium_predictor = _build_medium_predictor(
        model_file_name=args.medium_model_file_name,
        secure_model_file_name=args.medium_secure_model_file_name,
        training_report_file_name=args.medium_training_report_file_name,
        language=args.language,
    )
    report = evaluate_profile_quality_gate(
        samples=samples,
        fast_predictor=fast_predictor,
        medium_predictor=medium_predictor,
        dataset_glob_pattern=args.dataset_glob,
        thresholds=thresholds,
        n_splits=args.n_splits,
        random_state=args.random_state,
        fallback_test_size=args.test_size,
        progress_every=args.progress_every if args.progress_every > 0 else None,
    )
    payload = build_report_payload(report)
    serialized = json.dumps(payload, indent=2, sort_keys=True)

    output_path = (
        Path(args.out)
        if args.out is not None
        else settings.models.folder / "profile_quality_gate_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    enforce_quality_gate(report, require_pass=args.require_pass)
    print(serialized)


if __name__ == "__main__":
    main()
