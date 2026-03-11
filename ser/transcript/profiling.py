"""Profiling utilities for Whisper transcription defaults on RAVDESS audio."""

from __future__ import annotations

import glob
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

from ser._internal.transcription import default_benchmark as default_benchmark_helpers
from ser._internal.transcription import default_profiling as default_profiling_helpers
from ser._internal.transcription import default_recommendation as default_recommendation_helpers
from ser._internal.transcription import profile_candidates as profile_candidate_helpers
from ser._internal.transcription import profiling_entrypoints as profiling_entrypoint_helpers
from ser._internal.transcription import profiling_reporting as profiling_reporting_helpers
from ser._internal.transcription import ravdess_references as ravdess_reference_helpers
from ser._internal.transcription import runtime_calibration as runtime_calibration_helpers
from ser._internal.transcription import (
    runtime_calibration_workflow as runtime_calibration_workflow_helpers,
)
from ser._internal.transcription import text_metrics as text_metric_helpers
from ser._internal.transcription.public_boundary_profiling import (
    build_profile_candidates_from_public_boundary as _build_profile_candidates_boundary_impl,
)
from ser._internal.transcription.public_boundary_profiling import (
    run_cli_from_public_boundary as _run_cli_from_public_boundary_impl,
)
from ser._internal.transcription.public_boundary_profiling import (
    run_default_profile_benchmark_from_public_boundary as _run_default_profile_benchmark_boundary_impl,
)
from ser._internal.transcription.public_boundary_profiling import (
    run_runtime_calibration_from_public_boundary as _run_runtime_calibration_boundary_impl,
)
from ser.config import (
    AppConfig,
    ArtifactProfileName,
    get_settings,
    settings_override,
)
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.transcript.transcript_extractor import (
    TranscriptionProfile,
    load_whisper_model,
    transcribe_with_model,
)
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)
_PROFILING_OWNER_MODULES = (
    default_benchmark_helpers,
    runtime_calibration_workflow_helpers,
)

RAVDESS_REFERENCE_GLOB: Final[str] = "ser/dataset/ravdess/**/*.wav"
RAVDESS_STATEMENT_TEXT: Final[dict[str, str]] = {
    "01": "kids are talking by the door",
    "02": "dogs are sitting by the door",
}
DEFAULT_BENCHMARK_PROFILES: Final[
    tuple[ArtifactProfileName, ArtifactProfileName, ArtifactProfileName]
] = (
    "accurate",
    "medium",
    "fast",
)
DEFAULT_CALIBRATION_PROFILES: Final[
    tuple[
        ArtifactProfileName,
        ArtifactProfileName,
        ArtifactProfileName,
        ArtifactProfileName,
    ]
] = ("accurate", "medium", "accurate-research", "fast")

type RecommendationConfidence = Literal["high", "medium", "low"]
type RuntimeRecommendation = Literal["prefer_cpu", "prefer_mps", "mps_with_failover"]


@dataclass(frozen=True)
class TranscriptionProfileCandidate:
    """Candidate runtime profile to benchmark for default selection."""

    name: str
    source_profile: str
    backend_id: TranscriptionBackendId
    model_name: str
    use_demucs: bool
    use_vad: bool


@dataclass(frozen=True)
class ProfileBenchmarkSummary:
    """Aggregate metrics for one transcription runtime profile."""

    profile: TranscriptionProfileCandidate
    evaluated_samples: int
    failed_samples: int
    exact_match_rate: float
    mean_word_error_rate: float
    median_word_error_rate: float
    p90_word_error_rate: float
    mean_accuracy: float
    average_latency_seconds: float
    total_runtime_seconds: float
    error_message: str | None = None


@dataclass(frozen=True)
class AccuracyGate:
    """Minimum acceptable transcription accuracy for default proposals."""

    baseline_mean_accuracy: float
    minimum_mean_accuracy: float
    maximum_accuracy_drop: float
    absolute_accuracy_floor: float


@dataclass(frozen=True)
class DefaultRecommendation:
    """Recommendation outcome for whether defaults should change."""

    baseline_profile: str
    selected_profile: str
    should_change_defaults: bool
    reason: str
    selected_mean_accuracy: float
    selected_average_latency_seconds: float
    selected_speedup_vs_baseline: float
    minimum_required_samples: int


@dataclass(frozen=True)
class ProfilingResult:
    """All profiling outputs for reporting and CLI rendering."""

    reference_files: int
    gate: AccuracyGate
    summaries: tuple[ProfileBenchmarkSummary, ...]
    recommendation: DefaultRecommendation
    report_path: Path


@dataclass(frozen=True)
class RuntimeCalibrationMetrics:
    """Empirical runtime outcomes collected for one profile/model candidate."""

    profile: TranscriptionProfileCandidate
    iterations: int
    successful_runs: int
    failed_runs: int
    mps_loaded_runs: int
    mps_completed_runs: int
    mps_to_cpu_failover_runs: int
    hard_mps_oom_runs: int
    mean_latency_seconds: float
    error_messages: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeCalibrationRecommendation:
    """Runtime recommendation with confidence for one profile/model candidate."""

    profile: TranscriptionProfileCandidate
    recommendation: RuntimeRecommendation
    confidence: RecommendationConfidence
    reason: str
    metrics: RuntimeCalibrationMetrics


@dataclass(frozen=True)
class RuntimeCalibrationResult:
    """Calibration output including recommendation report path."""

    recommendations: tuple[RuntimeCalibrationRecommendation, ...]
    report_path: Path


@dataclass(frozen=True)
class RavdessMetadata:
    """Parsed metadata fields from a RAVDESS filename."""

    emotion_code: str
    statement_code: str
    actor_id: str


def _candidate_name(
    *,
    source_profile: str,
    backend_id: TranscriptionBackendId,
    model_name: str,
    use_demucs: bool,
    use_vad: bool,
) -> str:
    """Builds one deterministic benchmark candidate identifier."""
    return profile_candidate_helpers.candidate_name(
        source_profile=source_profile,
        backend_id=backend_id,
        model_name=model_name,
        use_demucs=use_demucs,
        use_vad=use_vad,
    )


def default_profile_candidates() -> tuple[TranscriptionProfileCandidate, ...]:
    """Returns benchmark candidates aligned to effective runtime transcription defaults."""
    return _build_profile_candidates_boundary_impl(
        profiles=DEFAULT_BENCHMARK_PROFILES,
        candidate_factory=TranscriptionProfileCandidate,
    )


def runtime_calibration_candidates(
    profiles: tuple[ArtifactProfileName, ...] = DEFAULT_CALIBRATION_PROFILES,
) -> tuple[TranscriptionProfileCandidate, ...]:
    """Returns calibration candidates for selected artifact profiles."""
    return _build_profile_candidates_boundary_impl(
        profiles=profiles,
        candidate_factory=TranscriptionProfileCandidate,
    )


def ravdess_reference_text(file_path: Path) -> str | None:
    """Returns ground-truth transcript text from a RAVDESS filename."""
    return ravdess_reference_helpers.reference_text(
        file_path,
        statement_text=RAVDESS_STATEMENT_TEXT,
    )


def _parse_ravdess_metadata(file_path: Path) -> RavdessMetadata | None:
    """Extracts actor/emotion/statement metadata from a RAVDESS filename."""
    metadata = ravdess_reference_helpers.parse_metadata(file_path)
    if metadata is None:
        return None
    return RavdessMetadata(
        emotion_code=metadata.emotion_code,
        statement_code=metadata.statement_code,
        actor_id=metadata.actor_id,
    )


def _stratified_reference_subset(
    references: list[Path],
    *,
    limit: int,
    random_seed: int,
) -> list[Path]:
    """Returns a deterministic near-uniform subset across actor+statement strata."""
    return ravdess_reference_helpers.stratified_reference_subset(
        references,
        limit=limit,
        random_seed=random_seed,
    )


def _summarize_subset_coverage(files: list[Path]) -> dict[str, int]:
    """Summarizes actor/emotion/statement diversity in selected references."""
    return ravdess_reference_helpers.summarize_subset_coverage(files)


def collect_ravdess_reference_files(
    limit: int | None = None,
    *,
    sampling_strategy: str = "stratified",
    random_seed: int = 42,
) -> list[Path]:
    """Collects RAVDESS files with known reference transcripts."""
    return ravdess_reference_helpers.collect_reference_files(
        glob_pattern=RAVDESS_REFERENCE_GLOB,
        statement_text=RAVDESS_STATEMENT_TEXT,
        limit=limit,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        glob_paths=lambda pattern, recursive: glob.glob(pattern, recursive=recursive),
    )


def _normalize_words(text: str) -> list[str]:
    """Normalizes transcript text into comparable token lists."""
    return text_metric_helpers.normalize_words(text)


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Computes token-level Levenshtein distance."""
    return text_metric_helpers.levenshtein_distance(reference, hypothesis)


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Computes word error rate using normalized token sequences."""
    return text_metric_helpers.compute_word_error_rate(reference_text, hypothesis_text)


def transcript_words_to_text(words: list[TranscriptWord]) -> str:
    """Converts per-word transcript entries into plain normalized text."""
    return text_metric_helpers.transcript_words_to_text(words)


def _percentile(values: list[float], percentile: float) -> float:
    """Returns a nearest-rank percentile for non-empty numeric samples."""
    return text_metric_helpers.percentile(values, percentile)


def profile_transcription_candidate(
    candidate: TranscriptionProfileCandidate,
    files: list[Path],
    language: str,
) -> ProfileBenchmarkSummary:
    """Profiles one candidate profile over a list of reference audio files."""
    return profiling_entrypoint_helpers.profile_transcription_candidate(
        candidate=candidate,
        files=files,
        language=language,
        profile_factory=TranscriptionProfile,
        profile_candidate_transcriptions_fn=(
            default_profiling_helpers.profile_candidate_transcriptions
        ),
        load_model=load_whisper_model,
        transcribe=transcribe_with_model,
        resolve_reference_text=ravdess_reference_text,
        words_to_text=transcript_words_to_text,
        compute_word_error_rate=word_error_rate,
        percentile=_percentile,
        logger=logger,
        summary_factory=ProfileBenchmarkSummary,
    )


def derive_accuracy_gate(
    baseline_summary: ProfileBenchmarkSummary,
    *,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
) -> AccuracyGate:
    """Derives the minimum acceptable mean accuracy from baseline results."""
    return default_recommendation_helpers.derive_accuracy_gate(
        baseline_summary,
        absolute_accuracy_floor=absolute_accuracy_floor,
        maximum_accuracy_drop=maximum_accuracy_drop,
        gate_factory=AccuracyGate,
    )


def recommend_default_profile(
    summaries: tuple[ProfileBenchmarkSummary, ...],
    gate: AccuracyGate,
    *,
    minimum_speedup_ratio: float = 1.10,
    minimum_required_samples: int = 100,
) -> DefaultRecommendation:
    """Selects a default profile only when it is faster and accuracy-safe."""
    return default_recommendation_helpers.recommend_default_profile(
        summaries,
        gate,
        minimum_speedup_ratio=minimum_speedup_ratio,
        minimum_required_samples=minimum_required_samples,
        recommendation_factory=DefaultRecommendation,
    )


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or falls back to ambient public-boundary config."""
    return settings if settings is not None else get_settings()


def run_default_profile_benchmark(
    *,
    language: str,
    sample_limit: int | None,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
    minimum_required_samples_for_recommendation: int = 100,
    sampling_strategy: str = "stratified",
    random_seed: int = 42,
    report_path: Path | None = None,
    settings: AppConfig | None = None,
) -> ProfilingResult:
    """Profiles default candidates and computes recommendation thresholds."""
    return _run_default_profile_benchmark_boundary_impl(
        language=language,
        sample_limit=sample_limit,
        absolute_accuracy_floor=absolute_accuracy_floor,
        maximum_accuracy_drop=maximum_accuracy_drop,
        minimum_required_samples_for_recommendation=(minimum_required_samples_for_recommendation),
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
        report_path=report_path,
        active_settings=_resolve_boundary_settings(settings),
        reference_glob=RAVDESS_REFERENCE_GLOB,
        collect_reference_files=lambda limit, sampling_strategy_name, seed: collect_ravdess_reference_files(
            limit=limit,
            sampling_strategy=sampling_strategy_name,
            random_seed=seed,
        ),
        default_profile_candidates=default_profile_candidates,
        profile_candidate=profile_transcription_candidate,
        derive_accuracy_gate=lambda baseline_summary, floor, drop: derive_accuracy_gate(
            baseline_summary,
            absolute_accuracy_floor=floor,
            maximum_accuracy_drop=drop,
        ),
        recommend_default_profile=lambda summaries, gate, minimum_required_samples: recommend_default_profile(
            summaries,
            gate,
            minimum_required_samples=minimum_required_samples,
        ),
        summarize_subset_coverage=_summarize_subset_coverage,
        serialize_gate=lambda gate: asdict(gate),
        serialize_summary=lambda summary: asdict(summary),
        serialize_recommendation=lambda recommendation: asdict(recommendation),
        result_factory=lambda reference_files, gate, summaries, recommendation, resolved_report_path: ProfilingResult(
            reference_files=reference_files,
            gate=gate,
            summaries=summaries,
            recommendation=recommendation,
            report_path=resolved_report_path,
        ),
    )


def parse_calibration_profiles(raw_profiles: str) -> tuple[ArtifactProfileName, ...]:
    """Parses and validates calibration profile names from CLI input."""
    return runtime_calibration_helpers.normalize_calibration_profile_csv(raw_profiles)


def derive_runtime_recommendation(
    metrics: RuntimeCalibrationMetrics,
) -> tuple[RuntimeRecommendation, RecommendationConfidence, str]:
    """Derives runtime recommendation and confidence from calibration metrics."""
    return runtime_calibration_helpers.derive_runtime_recommendation_from_metrics(metrics)


def _calibrate_runtime_candidate(
    *,
    candidate: TranscriptionProfileCandidate,
    calibration_file: Path,
    language: str,
    iterations: int,
) -> RuntimeCalibrationMetrics:
    """Runs iterative runtime probes for one profile/model candidate."""
    return profiling_entrypoint_helpers.calibrate_runtime_candidate(
        candidate=candidate,
        calibration_file=calibration_file,
        language=language,
        iterations=iterations,
        profile_factory=TranscriptionProfile,
        run_runtime_calibration_probes_fn=(
            runtime_calibration_helpers.run_runtime_calibration_probes
        ),
        load_model=load_whisper_model,
        transcribe=transcribe_with_model,
        metrics_factory=RuntimeCalibrationMetrics,
    )


def _serialize_runtime_calibration_recommendation(
    recommendation: RuntimeCalibrationRecommendation,
) -> dict[str, object]:
    """Serializes one runtime recommendation for report persistence."""
    return {
        "profile": recommendation.profile.source_profile,
        "backend_id": recommendation.profile.backend_id,
        "model_name": recommendation.profile.model_name,
        "recommendation": recommendation.recommendation,
        "confidence": recommendation.confidence,
        "reason": recommendation.reason,
        "metrics": asdict(recommendation.metrics),
    }


def run_transcription_runtime_calibration(
    *,
    calibration_file: Path,
    language: str,
    iterations_per_profile: int = 2,
    profile_names: tuple[ArtifactProfileName, ...] = DEFAULT_CALIBRATION_PROFILES,
    report_path: Path | None = None,
    settings: AppConfig | None = None,
) -> RuntimeCalibrationResult:
    """Runs runtime calibration probes and emits confidence-scored recommendations."""
    return _run_runtime_calibration_boundary_impl(
        calibration_file=calibration_file,
        language=language,
        iterations_per_profile=iterations_per_profile,
        profile_names=profile_names,
        report_path=report_path,
        active_settings=_resolve_boundary_settings(settings),
        settings_override=settings_override,
        runtime_calibration_candidates=runtime_calibration_candidates,
        calibrate_candidate=lambda candidate, calibration_file, language, iterations: _calibrate_runtime_candidate(
            candidate=candidate,
            calibration_file=calibration_file,
            language=language,
            iterations=iterations,
        ),
        derive_runtime_recommendation=derive_runtime_recommendation,
        recommendation_factory=lambda candidate, recommendation, confidence, reason, metrics: RuntimeCalibrationRecommendation(
            profile=candidate,
            recommendation=recommendation,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
        ),
        serialize_recommendation=_serialize_runtime_calibration_recommendation,
        result_factory=lambda recommendations, resolved_report_path: RuntimeCalibrationResult(
            recommendations=recommendations,
            report_path=resolved_report_path,
        ),
    )


def main() -> None:
    """Runs the internal transcription-default profiling workflow."""
    _run_cli_from_public_boundary_impl(
        run_default_profile_benchmark=run_default_profile_benchmark,
        run_runtime_calibration=run_transcription_runtime_calibration,
        parse_calibration_profiles=parse_calibration_profiles,
        profiling_summary_lines=profiling_reporting_helpers.profiling_summary_lines,
        runtime_calibration_summary_lines=(
            profiling_reporting_helpers.runtime_calibration_summary_lines
        ),
    )


if __name__ == "__main__":
    main()
