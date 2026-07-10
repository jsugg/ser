"""Cross-profile quality gate harness for fast-versus-medium evaluation."""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ser._internal.runtime.environment_plan import build_runtime_environment_plan
from ser._internal.runtime.process_env import temporary_process_env
from ser.config import AppConfig, reload_settings
from ser.data.data_loader import extract_ravdess_speaker_id_from_path
from ser.models.emotion_model import load_model, predict_emotions
from ser.repr import XLSRBackend
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_inference import run_medium_inference
from ser.runtime.quality_gate_cli import (
    QualityGateCliDefaults,
)
from ser.runtime.quality_gate_cli import (
    configure_cli_noise_controls as _gate_configure_cli_noise_controls,
)
from ser.runtime.quality_gate_cli import normalize_progress_every as _gate_normalize_progress_every
from ser.runtime.quality_gate_cli import parse_args as _gate_parse_args
from ser.runtime.quality_gate_evaluation import (
    NormalizedSegment,
    ProfileEvaluationResult,
)
from ser.runtime.quality_gate_evaluation import (
    clip_label_from_segments as _gate_clip_label_from_segments,
)
from ser.runtime.quality_gate_evaluation import (
    clip_stability_metrics as _gate_clip_stability_metrics,
)
from ser.runtime.quality_gate_evaluation import evaluate_profile as _gate_evaluate_profile
from ser.runtime.quality_gate_evaluation import normalize_segments as _gate_normalize_segments
from ser.runtime.quality_gate_evaluation import percentile as _gate_percentile
from ser.runtime.quality_gate_evaluation import segment_duration as _gate_segment_duration
from ser.runtime.quality_gate_policy import (
    ProfileComparisonResult,
    compare_profiles,
    metric_as_float,
    validate_thresholds,
)
from ser.runtime.quality_gate_reporting import build_report_payload as _gate_build_report_payload
from ser.runtime.quality_gate_reporting import enforce_quality_gate as _gate_enforce_quality_gate
from ser.runtime.quality_gate_reporting import (
    resolve_report_output_path as _gate_resolve_report_output_path,
)
from ser.runtime.quality_gate_reporting import (
    serialize_report_payload as _gate_serialize_report_payload,
)
from ser.runtime.quality_gate_reporting import (
    write_serialized_report as _gate_write_serialized_report,
)
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
        raise ValueError("No labeled audio samples were collected from dataset_glob_pattern.")
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
            raise RuntimeError("Grouped quality-gate folds contain train/test speaker overlap.")
        overlap_counts.append(overlap_count)

    samples_per_speaker = tuple(counts_by_speaker.values())
    return GroupedEvaluationSummary(
        unique_speakers=len(counts_by_speaker),
        min_samples_per_speaker=min(samples_per_speaker),
        max_samples_per_speaker=max(samples_per_speaker),
        fold_speaker_overlap_counts=tuple(overlap_counts),
    )


type _NormalizedSegment = NormalizedSegment


def _normalize_segments(segments: Sequence[SegmentLike]) -> list[_NormalizedSegment]:
    """Normalizes runtime segments into a validated immutable representation."""
    return _gate_normalize_segments(
        segments,
        canonicalize_segments=canonicalize_segments,
    )


def _segment_duration(segment: _NormalizedSegment) -> float:
    """Returns non-negative segment duration in seconds."""
    return _gate_segment_duration(segment)


def _clip_label_from_segments(segments: Sequence[_NormalizedSegment], *, unknown_label: str) -> str:
    """Returns a duration-weighted clip label derived from segment predictions."""
    return _gate_clip_label_from_segments(segments, unknown_label=unknown_label)


def _clip_stability_metrics(
    segments: Sequence[_NormalizedSegment],
) -> tuple[float, list[float]]:
    """Returns segment-count-per-minute and per-segment durations."""
    return _gate_clip_stability_metrics(segments)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Returns nearest-rank percentile for a non-empty value sequence."""
    return _gate_percentile(values, percentile)


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
    evaluation: ProfileEvaluationResult = _gate_evaluate_profile(
        profile_name=profile_name,
        samples=samples,
        folds=folds,
        predictor=predictor,
        unknown_label=unknown_label,
        canonicalize_segments=canonicalize_segments,
        compute_ser_metrics=compute_ser_metrics,
        progress_every=progress_every,
    )
    return ProfileQualitySummary(
        profile=evaluation.profile,
        evaluated_clips=evaluation.evaluated_clips,
        failed_clips=evaluation.failed_clips,
        metrics=evaluation.metrics,
        temporal_stability=TemporalStabilitySummary(
            segment_count_per_minute=evaluation.segment_count_per_minute,
            median_segment_duration_seconds=(evaluation.median_segment_duration_seconds),
        ),
        latency=LatencySummary(
            mean_seconds=evaluation.latency_mean_seconds,
            median_seconds=evaluation.latency_median_seconds,
            p95_seconds=evaluation.latency_p95_seconds,
        ),
    )


def _metric_as_float(metrics: Mapping[str, object], key: str) -> float:
    """Reads one numeric metric from a metrics payload with validation."""
    return metric_as_float(metrics, key)


def _validate_thresholds(thresholds: QualityGateThresholds) -> None:
    """Validates quality-gate threshold bounds."""
    validate_thresholds(thresholds)


def _compare_profiles(
    *,
    fast: ProfileQualitySummary,
    medium: ProfileQualitySummary,
    thresholds: QualityGateThresholds,
) -> QualityGateComparison:
    """Compares medium versus fast against configured quality thresholds."""
    comparison: ProfileComparisonResult = compare_profiles(
        fast=fast,
        medium=medium,
        thresholds=thresholds,
    )
    return QualityGateComparison(
        medium_minus_fast_uar=comparison.medium_minus_fast_uar,
        medium_minus_fast_macro_f1=comparison.medium_minus_fast_macro_f1,
        medium_segments_per_minute=comparison.medium_segments_per_minute,
        medium_median_segment_duration_seconds=(comparison.medium_median_segment_duration_seconds),
        passes_quality_gate=comparison.passes_quality_gate,
        failure_reasons=comparison.failure_reasons,
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


def _settings_with_artifact_files(
    *,
    settings: AppConfig,
    model_file_name: str,
    secure_model_file_name: str,
    training_report_file_name: str,
) -> AppConfig:
    """Builds one explicit settings snapshot for quality-gate artifact selection."""
    return replace(
        settings,
        models=replace(
            settings.models,
            model_file_name=model_file_name,
            secure_model_file_name=secure_model_file_name,
            training_report_file_name=training_report_file_name,
        ),
    )


def _build_fast_predictor(
    *,
    model_file_name: str,
    secure_model_file_name: str,
    training_report_file_name: str,
    settings: AppConfig,
) -> SegmentPredictor:
    """Builds fast predictor with deterministic model artifact selection."""
    artifact_settings = _settings_with_artifact_files(
        settings=settings,
        model_file_name=model_file_name,
        secure_model_file_name=secure_model_file_name,
        training_report_file_name=training_report_file_name,
    )
    loaded_model = load_model(settings=artifact_settings)

    def _predict(audio_path: str) -> Sequence[SegmentLike]:
        return predict_emotions(audio_path, loaded_model=loaded_model)

    return _predict


def _build_medium_predictor(
    *,
    model_file_name: str,
    secure_model_file_name: str,
    training_report_file_name: str,
    language: str | None,
    settings: AppConfig,
) -> SegmentPredictor:
    """Builds medium predictor with deterministic model artifact selection."""
    artifact_settings = _settings_with_artifact_files(
        settings=settings,
        model_file_name=model_file_name,
        secure_model_file_name=secure_model_file_name,
        training_report_file_name=training_report_file_name,
    )
    default_language = settings.default_language
    backend = XLSRBackend(cache_dir=settings.models.huggingface_cache_root)
    loaded_model = load_model(settings=artifact_settings)
    runtime_environment = build_runtime_environment_plan(artifact_settings)

    def _predict(audio_path: str) -> Sequence[SegmentLike]:
        with temporary_process_env(runtime_environment.torch_runtime):
            result = run_medium_inference(
                InferenceRequest(
                    file_path=audio_path,
                    language=language or default_language,
                    save_transcript=False,
                ),
                artifact_settings,
                loaded_model=loaded_model,
                backend=backend,
                enforce_timeout=False,
                allow_retries=False,
            )
        return result.segments

    return _predict


def build_report_payload(report: ProfileQualityGateReport) -> dict[str, object]:
    """Converts quality-gate report dataclasses into a JSON-safe dictionary."""
    return _gate_build_report_payload(report)


def enforce_quality_gate(
    report: ProfileQualityGateReport,
    *,
    require_pass: bool,
) -> None:
    """Raises a terminal error when gate enforcement is required and failed."""
    _gate_enforce_quality_gate(report, require_pass=require_pass)


def _configure_cli_noise_controls() -> None:
    """Suppresses non-actionable warning/log noise for long gate executions."""
    _gate_configure_cli_noise_controls()


def _parse_args(settings: AppConfig) -> argparse.Namespace:
    """Parses command-line arguments for profile quality gate evaluation."""
    defaults = QualityGateCliDefaults(
        dataset_glob=settings.dataset.glob_pattern,
        test_size=settings.training.test_size,
        random_state=settings.training.random_state,
        fast_model_file_name=settings.models.model_file_name,
        fast_secure_model_file_name=settings.models.secure_model_file_name,
        fast_training_report_file_name=settings.models.training_report_file_name,
        medium_secure_model_file_name=settings.models.secure_model_file_name,
        medium_training_report_file_name=settings.models.training_report_file_name,
        min_uar_delta=settings.quality_gate.min_uar_delta,
        min_macro_f1_delta=settings.quality_gate.min_macro_f1_delta,
        max_medium_segments_per_minute=(settings.quality_gate.max_medium_segments_per_minute),
        min_medium_median_segment_duration_seconds=(
            settings.quality_gate.min_medium_median_segment_duration_seconds
        ),
    )
    return _gate_parse_args(defaults=defaults)


def _resolve_boundary_settings(settings: AppConfig | None = None) -> AppConfig:
    """Returns explicit settings or reloads a quality-gate CLI snapshot."""
    return settings if settings is not None else reload_settings()


def main() -> None:
    """Runs quality-gate harness and writes JSON report to stdout or disk."""
    _configure_cli_noise_controls()
    settings = _resolve_boundary_settings()
    args = _parse_args(settings)
    thresholds = QualityGateThresholds(
        minimum_uar_delta=args.min_uar_delta,
        minimum_macro_f1_delta=args.min_macro_f1_delta,
        maximum_medium_segments_per_minute=args.max_medium_segments_per_minute,
        minimum_medium_median_segment_duration_seconds=(args.min_medium_median_segment_duration),
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
        settings=settings,
    )
    medium_predictor = _build_medium_predictor(
        model_file_name=args.medium_model_file_name,
        secure_model_file_name=args.medium_secure_model_file_name,
        training_report_file_name=args.medium_training_report_file_name,
        language=args.language,
        settings=settings,
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
        progress_every=_gate_normalize_progress_every(args.progress_every),
    )
    payload = build_report_payload(report)
    serialized = _gate_serialize_report_payload(payload)
    output_path = _gate_resolve_report_output_path(
        output_path=args.out,
        default_directory=settings.models.folder,
    )
    _gate_write_serialized_report(serialized=serialized, output_path=output_path)
    enforce_quality_gate(report, require_pass=args.require_pass)
    print(serialized)


if __name__ == "__main__":
    main()
