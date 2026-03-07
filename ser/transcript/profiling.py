"""Profiling utilities for Whisper transcription defaults on RAVDESS audio."""

from __future__ import annotations

import glob
import logging
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from ser._internal.transcription import default_profiling as default_profiling_helpers
from ser._internal.transcription import (
    profiling_reporting as profiling_reporting_helpers,
)
from ser._internal.transcription import (
    runtime_calibration as runtime_calibration_helpers,
)
from ser.config import (
    AppConfig,
    ArtifactProfileName,
    get_settings,
    resolve_profile_transcription_config,
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
    demucs_label = "demucs" if use_demucs else "no_demucs"
    vad_label = "vad" if use_vad else "no_vad"
    return f"{source_profile}_{backend_id}_{model_name}_{demucs_label}_{vad_label}"


def default_profile_candidates() -> tuple[TranscriptionProfileCandidate, ...]:
    """Returns benchmark candidates aligned to effective runtime transcription defaults."""
    candidates: list[TranscriptionProfileCandidate] = []
    for profile_name in DEFAULT_BENCHMARK_PROFILES:
        backend_id, model_name, use_demucs, use_vad = (
            resolve_profile_transcription_config(profile_name)
        )
        candidates.append(
            TranscriptionProfileCandidate(
                name=_candidate_name(
                    source_profile=profile_name,
                    backend_id=backend_id,
                    model_name=model_name,
                    use_demucs=use_demucs,
                    use_vad=use_vad,
                ),
                source_profile=profile_name,
                backend_id=backend_id,
                model_name=model_name,
                use_demucs=use_demucs,
                use_vad=use_vad,
            )
        )
    return tuple(candidates)


def runtime_calibration_candidates(
    profiles: tuple[ArtifactProfileName, ...] = DEFAULT_CALIBRATION_PROFILES,
) -> tuple[TranscriptionProfileCandidate, ...]:
    """Returns calibration candidates for selected artifact profiles."""
    candidates: list[TranscriptionProfileCandidate] = []
    for profile_name in profiles:
        backend_id, model_name, use_demucs, use_vad = (
            resolve_profile_transcription_config(profile_name)
        )
        candidates.append(
            TranscriptionProfileCandidate(
                name=_candidate_name(
                    source_profile=profile_name,
                    backend_id=backend_id,
                    model_name=model_name,
                    use_demucs=use_demucs,
                    use_vad=use_vad,
                ),
                source_profile=profile_name,
                backend_id=backend_id,
                model_name=model_name,
                use_demucs=use_demucs,
                use_vad=use_vad,
            )
        )
    return tuple(candidates)


def ravdess_reference_text(file_path: Path) -> str | None:
    """Returns ground-truth transcript text from a RAVDESS filename."""
    parts = file_path.stem.split("-")
    if len(parts) < 5:
        return None
    statement_code = parts[4]
    return RAVDESS_STATEMENT_TEXT.get(statement_code)


def _parse_ravdess_metadata(file_path: Path) -> RavdessMetadata | None:
    """Extracts actor/emotion/statement metadata from a RAVDESS filename."""
    parts = file_path.stem.split("-")
    if len(parts) < 7:
        return None
    return RavdessMetadata(
        emotion_code=parts[2],
        statement_code=parts[4],
        actor_id=parts[6],
    )


def _stratified_reference_subset(
    references: list[Path],
    *,
    limit: int,
    random_seed: int,
) -> list[Path]:
    """Returns a deterministic near-uniform subset across actor+statement strata."""
    if limit >= len(references):
        return references

    strata: dict[tuple[str, str], list[Path]] = {}
    for file_path in references:
        metadata = _parse_ravdess_metadata(file_path)
        if metadata is None:
            continue
        key = (metadata.actor_id, metadata.statement_code)
        strata.setdefault(key, []).append(file_path)

    if not strata:
        return references[:limit]

    rng = random.Random(random_seed)
    keys = sorted(strata.keys())
    rng.shuffle(keys)
    for key in keys:
        strata[key] = sorted(strata[key])
        rng.shuffle(strata[key])

    selected: list[Path] = []
    consumed: dict[tuple[str, str], int] = {key: 0 for key in keys}

    while len(selected) < limit:
        progressed = False
        for key in keys:
            group = strata[key]
            index = consumed[key]
            if index >= len(group):
                continue
            selected.append(group[index])
            consumed[key] = index + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return sorted(selected)


def _summarize_subset_coverage(files: list[Path]) -> dict[str, int]:
    """Summarizes actor/emotion/statement diversity in selected references."""
    actors: set[str] = set()
    emotions: set[str] = set()
    statements: set[str] = set()
    for file_path in files:
        metadata = _parse_ravdess_metadata(file_path)
        if metadata is None:
            continue
        actors.add(metadata.actor_id)
        emotions.add(metadata.emotion_code)
        statements.add(metadata.statement_code)
    return {
        "actors": len(actors),
        "emotions": len(emotions),
        "statements": len(statements),
    }


def collect_ravdess_reference_files(
    limit: int | None = None,
    *,
    sampling_strategy: str = "stratified",
    random_seed: int = 42,
) -> list[Path]:
    """Collects RAVDESS files with known reference transcripts."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    files = sorted(
        Path(raw_path) for raw_path in glob.glob(RAVDESS_REFERENCE_GLOB, recursive=True)
    )
    references = [path for path in files if ravdess_reference_text(path) is not None]
    if limit is not None:
        if sampling_strategy == "head":
            return references[:limit]
        if sampling_strategy == "stratified":
            return _stratified_reference_subset(
                references,
                limit=limit,
                random_seed=random_seed,
            )
        raise ValueError("sampling_strategy must be one of: 'stratified', 'head'.")
    return references


def _normalize_words(text: str) -> list[str]:
    """Normalizes transcript text into comparable token lists."""
    lowered = text.strip().lower()
    normalized = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return [token for token in normalized.split() if token]


def _levenshtein_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Computes token-level Levenshtein distance."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous_row = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current_row = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            insert_cost = current_row[hyp_index - 1] + 1
            delete_cost = previous_row[hyp_index] + 1
            substitute_cost = previous_row[hyp_index - 1] + (
                0 if ref_token == hyp_token else 1
            )
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Computes word error rate using normalized token sequences."""
    reference_tokens = _normalize_words(reference_text)
    hypothesis_tokens = _normalize_words(hypothesis_text)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
    return distance / float(len(reference_tokens))


def transcript_words_to_text(words: list[TranscriptWord]) -> str:
    """Converts per-word transcript entries into plain normalized text."""
    return " ".join(word.word.strip() for word in words if word.word.strip())


def _percentile(values: list[float], percentile: float) -> float:
    """Returns a nearest-rank percentile for non-empty numeric samples."""
    if not values:
        return 1.0
    rank = max(0, math.ceil(percentile * len(values)) - 1)
    return sorted(values)[rank]


def profile_transcription_candidate(
    candidate: TranscriptionProfileCandidate,
    files: list[Path],
    language: str,
) -> ProfileBenchmarkSummary:
    """Profiles one candidate profile over a list of reference audio files."""
    profile = TranscriptionProfile(
        backend_id=candidate.backend_id,
        model_name=candidate.model_name,
        use_demucs=candidate.use_demucs,
        use_vad=candidate.use_vad,
    )
    stats = default_profiling_helpers.profile_candidate_transcriptions(
        candidate_name=candidate.name,
        profile=profile,
        files=files,
        language=language,
        load_model=load_whisper_model,
        transcribe=transcribe_with_model,
        resolve_reference_text=ravdess_reference_text,
        words_to_text=transcript_words_to_text,
        compute_word_error_rate=word_error_rate,
        percentile=_percentile,
        logger=logger,
    )

    return ProfileBenchmarkSummary(
        profile=candidate,
        evaluated_samples=stats.evaluated_samples,
        failed_samples=stats.failed_samples,
        exact_match_rate=stats.exact_match_rate,
        mean_word_error_rate=stats.mean_word_error_rate,
        median_word_error_rate=stats.median_word_error_rate,
        p90_word_error_rate=stats.p90_word_error_rate,
        mean_accuracy=stats.mean_accuracy,
        average_latency_seconds=stats.average_latency_seconds,
        total_runtime_seconds=stats.total_runtime_seconds,
        error_message=stats.error_message,
    )


def derive_accuracy_gate(
    baseline_summary: ProfileBenchmarkSummary,
    *,
    absolute_accuracy_floor: float,
    maximum_accuracy_drop: float,
) -> AccuracyGate:
    """Derives the minimum acceptable mean accuracy from baseline results."""
    minimum_mean_accuracy = max(
        absolute_accuracy_floor,
        baseline_summary.mean_accuracy - maximum_accuracy_drop,
    )
    return AccuracyGate(
        baseline_mean_accuracy=baseline_summary.mean_accuracy,
        minimum_mean_accuracy=minimum_mean_accuracy,
        maximum_accuracy_drop=maximum_accuracy_drop,
        absolute_accuracy_floor=absolute_accuracy_floor,
    )


def recommend_default_profile(
    summaries: tuple[ProfileBenchmarkSummary, ...],
    gate: AccuracyGate,
    *,
    minimum_speedup_ratio: float = 1.10,
    minimum_required_samples: int = 100,
) -> DefaultRecommendation:
    """Selects a default profile only when it is faster and accuracy-safe."""
    baseline = summaries[0]
    if baseline.evaluated_samples < minimum_required_samples:
        return DefaultRecommendation(
            baseline_profile=baseline.profile.name,
            selected_profile=baseline.profile.name,
            should_change_defaults=False,
            reason=(
                "Insufficient sample size for safe default changes. "
                f"Need at least {minimum_required_samples} evaluated samples."
            ),
            selected_mean_accuracy=baseline.mean_accuracy,
            selected_average_latency_seconds=baseline.average_latency_seconds,
            selected_speedup_vs_baseline=1.0,
            minimum_required_samples=minimum_required_samples,
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
        return DefaultRecommendation(
            baseline_profile=baseline.profile.name,
            selected_profile=baseline.profile.name,
            should_change_defaults=False,
            reason=(
                "No candidate met both the accuracy gate and required speedup; "
                "keep current defaults."
            ),
            selected_mean_accuracy=baseline.mean_accuracy,
            selected_average_latency_seconds=baseline.average_latency_seconds,
            selected_speedup_vs_baseline=1.0,
            minimum_required_samples=minimum_required_samples,
        )

    return DefaultRecommendation(
        baseline_profile=baseline.profile.name,
        selected_profile=selected.profile.name,
        should_change_defaults=True,
        reason="Candidate met the accuracy gate and exceeded required speedup.",
        selected_mean_accuracy=selected.mean_accuracy,
        selected_average_latency_seconds=selected.average_latency_seconds,
        selected_speedup_vs_baseline=selected_speedup,
        minimum_required_samples=minimum_required_samples,
    )


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
    if not 0.0 <= absolute_accuracy_floor <= 1.0:
        raise ValueError("absolute_accuracy_floor must be between 0 and 1.")
    if not 0.0 <= maximum_accuracy_drop <= 1.0:
        raise ValueError("maximum_accuracy_drop must be between 0 and 1.")
    if minimum_required_samples_for_recommendation <= 0:
        raise ValueError(
            "minimum_required_samples_for_recommendation must be greater than zero."
        )

    active_settings = settings if settings is not None else get_settings()
    reference_files = collect_ravdess_reference_files(
        limit=sample_limit,
        sampling_strategy=sampling_strategy,
        random_seed=random_seed,
    )
    if not reference_files:
        raise RuntimeError(
            f"No RAVDESS reference files found under {RAVDESS_REFERENCE_GLOB}."
        )

    candidates = default_profile_candidates()
    summaries = tuple(
        profile_transcription_candidate(candidate, reference_files, language)
        for candidate in candidates
    )
    baseline_summary = summaries[0]
    gate = derive_accuracy_gate(
        baseline_summary,
        absolute_accuracy_floor=absolute_accuracy_floor,
        maximum_accuracy_drop=maximum_accuracy_drop,
    )
    recommendation = recommend_default_profile(
        summaries,
        gate,
        minimum_required_samples=minimum_required_samples_for_recommendation,
    )

    output_path = (
        active_settings.models.folder / "transcription_profile_report.json"
        if report_path is None
        else report_path
    )
    coverage = _summarize_subset_coverage(reference_files)
    payload: dict[str, object] = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "reference_glob": RAVDESS_REFERENCE_GLOB,
        "reference_files": len(reference_files),
        "sampling_strategy": sampling_strategy,
        "random_seed": random_seed,
        "subset_coverage": coverage,
        "accuracy_gate": asdict(gate),
        "minimum_required_samples_for_recommendation": (
            minimum_required_samples_for_recommendation
        ),
        "profiles": [asdict(summary) for summary in summaries],
        "recommendation": asdict(recommendation),
    }
    persisted_path = profiling_reporting_helpers.persist_profile_report(
        output_path,
        payload,
    )

    return ProfilingResult(
        reference_files=len(reference_files),
        gate=gate,
        summaries=summaries,
        recommendation=recommendation,
        report_path=persisted_path,
    )


def parse_calibration_profiles(raw_profiles: str) -> tuple[ArtifactProfileName, ...]:
    """Parses and validates calibration profile names from CLI input."""
    return runtime_calibration_helpers.normalize_calibration_profile_csv(raw_profiles)


def derive_runtime_recommendation(
    metrics: RuntimeCalibrationMetrics,
) -> tuple[RuntimeRecommendation, RecommendationConfidence, str]:
    """Derives runtime recommendation and confidence from calibration metrics."""
    return runtime_calibration_helpers.derive_runtime_recommendation_from_metrics(
        metrics
    )


def _calibrate_runtime_candidate(
    *,
    candidate: TranscriptionProfileCandidate,
    calibration_file: Path,
    language: str,
    iterations: int,
) -> RuntimeCalibrationMetrics:
    """Runs iterative runtime probes for one profile/model candidate."""
    active_profile = TranscriptionProfile(
        backend_id=candidate.backend_id,
        model_name=candidate.model_name,
        use_demucs=candidate.use_demucs,
        use_vad=candidate.use_vad,
    )
    stats = runtime_calibration_helpers.run_runtime_calibration_probes(
        backend_id=candidate.backend_id,
        active_profile=active_profile,
        calibration_file=calibration_file,
        language=language,
        iterations=iterations,
        load_model=load_whisper_model,
        transcribe=transcribe_with_model,
    )
    return RuntimeCalibrationMetrics(
        profile=candidate,
        iterations=iterations,
        successful_runs=stats.successful_runs,
        failed_runs=stats.failed_runs,
        mps_loaded_runs=stats.mps_loaded_runs,
        mps_completed_runs=stats.mps_completed_runs,
        mps_to_cpu_failover_runs=stats.mps_to_cpu_failover_runs,
        hard_mps_oom_runs=stats.hard_mps_oom_runs,
        mean_latency_seconds=stats.mean_latency_seconds,
        error_messages=stats.error_messages,
    )


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
    if iterations_per_profile <= 0:
        raise ValueError("iterations_per_profile must be greater than zero.")
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration audio file not found: {calibration_file}")

    active_settings = settings if settings is not None else get_settings()
    calibration_settings = (
        runtime_calibration_helpers.build_runtime_calibration_settings(active_settings)
    )
    recommendations: list[RuntimeCalibrationRecommendation] = []
    with settings_override(calibration_settings):
        for candidate in runtime_calibration_candidates(profile_names):
            metrics = _calibrate_runtime_candidate(
                candidate=candidate,
                calibration_file=calibration_file,
                language=language,
                iterations=iterations_per_profile,
            )
            recommendation, confidence, reason = derive_runtime_recommendation(metrics)
            recommendations.append(
                RuntimeCalibrationRecommendation(
                    profile=candidate,
                    recommendation=recommendation,
                    confidence=confidence,
                    reason=reason,
                    metrics=metrics,
                )
            )

    output_path = (
        runtime_calibration_helpers.runtime_calibration_report_path(active_settings)
        if report_path is None
        else report_path
    )
    payload = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "calibration_file": str(calibration_file),
        "iterations_per_profile": iterations_per_profile,
        "profiles": [
            {
                "profile": recommendation.profile.source_profile,
                "backend_id": recommendation.profile.backend_id,
                "model_name": recommendation.profile.model_name,
                "recommendation": recommendation.recommendation,
                "confidence": recommendation.confidence,
                "reason": recommendation.reason,
                "metrics": asdict(recommendation.metrics),
            }
            for recommendation in recommendations
        ],
    }
    persisted_path = profiling_reporting_helpers.persist_profile_report(
        output_path,
        payload,
    )
    return RuntimeCalibrationResult(
        recommendations=tuple(recommendations),
        report_path=persisted_path,
    )


def main() -> None:
    """Runs the internal transcription-default profiling workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal transcription default profiling utility"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=("benchmark", "runtime-calibration"),
    )
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--accuracy-floor", type=float, default=0.90)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--min-samples-for-recommendation", type=int, default=100)
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="stratified",
        choices=["stratified", "head"],
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--calibration-file", type=str, default=None)
    parser.add_argument("--calibration-iterations", type=int, default=2)
    parser.add_argument(
        "--calibration-profiles",
        type=str,
        default="accurate,medium,accurate-research,fast",
        help=(
            "Comma-separated profile list for runtime calibration "
            "(fast,medium,accurate,accurate-research)."
        ),
    )
    args = parser.parse_args()

    report_path = None if args.report_path is None else Path(args.report_path)
    if args.mode == "runtime-calibration":
        if args.calibration_file is None:
            raise ValueError(
                "--calibration-file is required for runtime-calibration mode."
            )
        calibration_result = run_transcription_runtime_calibration(
            calibration_file=Path(args.calibration_file),
            language=args.language,
            iterations_per_profile=args.calibration_iterations,
            profile_names=parse_calibration_profiles(args.calibration_profiles),
            report_path=report_path,
        )
        for line in profiling_reporting_helpers.runtime_calibration_summary_lines(
            calibration_result
        ):
            print(line)
        return

    result = run_default_profile_benchmark(
        language=args.language,
        sample_limit=args.sample_limit,
        absolute_accuracy_floor=args.accuracy_floor,
        maximum_accuracy_drop=args.max_accuracy_drop,
        minimum_required_samples_for_recommendation=(
            args.min_samples_for_recommendation
        ),
        sampling_strategy=args.sampling_strategy,
        random_seed=args.random_seed,
        report_path=report_path,
    )
    for line in profiling_reporting_helpers.profiling_summary_lines(result):
        print(line)


if __name__ == "__main__":
    main()
