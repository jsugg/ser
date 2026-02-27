"""Profiling utilities for Whisper transcription defaults on RAVDESS audio."""

from __future__ import annotations

import gc
import glob
import json
import logging
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

from ser.config import (
    AppConfig,
    ArtifactProfileName,
    apply_settings,
    get_settings,
    resolve_profile_transcription_config,
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
    start_time = time.perf_counter()
    if not files:
        return ProfileBenchmarkSummary(
            profile=candidate,
            evaluated_samples=0,
            failed_samples=0,
            exact_match_rate=0.0,
            mean_word_error_rate=1.0,
            median_word_error_rate=1.0,
            p90_word_error_rate=1.0,
            mean_accuracy=0.0,
            average_latency_seconds=0.0,
            total_runtime_seconds=0.0,
            error_message="No reference files provided.",
        )

    profile = TranscriptionProfile(
        backend_id=candidate.backend_id,
        model_name=candidate.model_name,
        use_demucs=candidate.use_demucs,
        use_vad=candidate.use_vad,
    )
    try:
        model = load_whisper_model(profile=profile)
    except Exception as err:
        logger.error(
            "Failed to load Whisper model for profile %s: %s",
            candidate.name,
            err,
        )
        return ProfileBenchmarkSummary(
            profile=candidate,
            evaluated_samples=0,
            failed_samples=len(files),
            exact_match_rate=0.0,
            mean_word_error_rate=1.0,
            median_word_error_rate=1.0,
            p90_word_error_rate=1.0,
            mean_accuracy=0.0,
            average_latency_seconds=0.0,
            total_runtime_seconds=time.perf_counter() - start_time,
            error_message=str(err),
        )

    word_error_rates: list[float] = []
    accuracies: list[float] = []
    latencies: list[float] = []
    exact_matches = 0
    failed_samples = 0
    for file_path in files:
        reference_text = ravdess_reference_text(file_path)
        if reference_text is None:
            failed_samples += 1
            continue

        sample_start = time.perf_counter()
        try:
            words = transcribe_with_model(
                model=model,
                file_path=str(file_path),
                language=language,
                profile=profile,
            )
        except Exception as err:
            failed_samples += 1
            logger.warning("Transcription failed for %s: %s", file_path, err)
            continue
        sample_latency = time.perf_counter() - sample_start

        hypothesis_text = transcript_words_to_text(words)
        sample_wer = word_error_rate(reference_text, hypothesis_text)
        sample_accuracy = max(0.0, 1.0 - sample_wer)

        word_error_rates.append(sample_wer)
        accuracies.append(sample_accuracy)
        latencies.append(sample_latency)
        if sample_wer == 0.0:
            exact_matches += 1

    del model

    evaluated_samples = len(word_error_rates)
    if evaluated_samples == 0:
        return ProfileBenchmarkSummary(
            profile=candidate,
            evaluated_samples=0,
            failed_samples=failed_samples,
            exact_match_rate=0.0,
            mean_word_error_rate=1.0,
            median_word_error_rate=1.0,
            p90_word_error_rate=1.0,
            mean_accuracy=0.0,
            average_latency_seconds=0.0,
            total_runtime_seconds=time.perf_counter() - start_time,
            error_message="No samples were transcribed successfully.",
        )

    return ProfileBenchmarkSummary(
        profile=candidate,
        evaluated_samples=evaluated_samples,
        failed_samples=failed_samples,
        exact_match_rate=exact_matches / float(evaluated_samples),
        mean_word_error_rate=statistics.fmean(word_error_rates),
        median_word_error_rate=statistics.median(word_error_rates),
        p90_word_error_rate=_percentile(word_error_rates, 0.90),
        mean_accuracy=statistics.fmean(accuracies),
        average_latency_seconds=statistics.fmean(latencies),
        total_runtime_seconds=time.perf_counter() - start_time,
        error_message=None,
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


def _default_report_path() -> Path:
    """Returns the default profiling report path."""
    settings = get_settings()
    return settings.models.folder / "transcription_profile_report.json"


def _persist_profile_report(path: Path, payload: dict[str, object]) -> Path:
    """Persists profile report payload to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


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

    output_path = _default_report_path() if report_path is None else report_path
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
    persisted_path = _persist_profile_report(output_path, payload)

    return ProfilingResult(
        reference_files=len(reference_files),
        gate=gate,
        summaries=summaries,
        recommendation=recommendation,
        report_path=persisted_path,
    )


def _runtime_calibration_report_path() -> Path:
    """Returns default output path for runtime calibration reports."""
    settings = get_settings()
    return settings.models.folder / "transcription_runtime_calibration_report.json"


def _normalize_profile_csv(raw_profiles: str) -> tuple[ArtifactProfileName, ...]:
    """Parses comma-separated profile names for calibration workflows."""
    parsed: list[ArtifactProfileName] = []
    for token in raw_profiles.split(","):
        normalized = token.strip().lower()
        if not normalized:
            continue
        if normalized not in {
            "fast",
            "medium",
            "accurate",
            "accurate-research",
        }:
            raise ValueError(f"Unsupported profile in calibration set: {token!r}.")
        parsed.append(cast(ArtifactProfileName, normalized))
    if not parsed:
        raise ValueError("At least one calibration profile must be provided.")
    return tuple(dict.fromkeys(parsed))


def parse_calibration_profiles(raw_profiles: str) -> tuple[ArtifactProfileName, ...]:
    """Parses and validates calibration profile names from CLI input."""
    return _normalize_profile_csv(raw_profiles)


def _resolve_runtime_device_for_loaded_model(
    *,
    model: object,
    backend_id: TranscriptionBackendId,
) -> str:
    """Resolves active runtime device for one loaded model object."""
    if backend_id == "stable_whisper":
        from ser.transcript.backends.stable_whisper_mps_compat import (
            get_stable_whisper_runtime_device,
        )

        return get_stable_whisper_runtime_device(
            model,
            default_device_type="cpu",
        )
    return "cpu"


def _is_hard_mps_oom(error: Exception) -> bool:
    """Returns whether one exception indicates a hard MPS OOM condition."""
    message = " ".join(str(error).split()).lower()
    if "out of memory" not in message or "mps" not in message:
        return False
    incompatibility_markers = (
        "sparsemps",
        "aten::empty.memory_format",
        "std_mean",
        "unsupported dtype",
        "cannot convert a mps tensor to float64 dtype",
        "not currently implemented",
    )
    return not any(marker in message for marker in incompatibility_markers)


def derive_runtime_recommendation(
    metrics: RuntimeCalibrationMetrics,
) -> tuple[RuntimeRecommendation, RecommendationConfidence, str]:
    """Derives runtime recommendation and confidence from calibration metrics."""
    if metrics.profile.backend_id != "stable_whisper":
        return (
            "prefer_cpu",
            "high",
            "backend does not support MPS runtime in this project policy.",
        )
    if metrics.iterations <= 0:
        return ("prefer_cpu", "low", "No calibration runs were executed.")
    if metrics.mps_loaded_runs == 0:
        confidence: RecommendationConfidence = (
            "high" if metrics.iterations >= 2 else "medium"
        )
        return (
            "prefer_cpu",
            confidence,
            "MPS runtime was never admitted at model load.",
        )

    mps_stability_ratio = metrics.mps_completed_runs / float(metrics.iterations)
    failover_ratio = metrics.mps_to_cpu_failover_runs / float(metrics.iterations)
    failure_ratio = metrics.failed_runs / float(metrics.iterations)

    if metrics.hard_mps_oom_runs > 0:
        confidence = "high" if metrics.hard_mps_oom_runs >= 2 else "medium"
        return (
            "prefer_cpu",
            confidence,
            "Hard MPS OOM observed during calibration.",
        )

    if mps_stability_ratio >= 0.90 and failure_ratio == 0.0:
        confidence = "high" if metrics.iterations >= 3 else "medium"
        return (
            "prefer_mps",
            confidence,
            "MPS runs remained stable across calibration.",
        )

    if mps_stability_ratio >= 0.40 and failover_ratio > 0.0:
        confidence = "medium" if metrics.iterations >= 2 else "low"
        return (
            "mps_with_failover",
            confidence,
            "MPS shows mixed stability; keep CPU failover enabled.",
        )

    confidence = "medium" if metrics.iterations >= 2 else "low"
    return (
        "prefer_cpu",
        confidence,
        "MPS stability was insufficient for reliable runtime selection.",
    )


def _calibration_settings(base_settings: AppConfig) -> AppConfig:
    """Builds settings snapshot used during runtime calibration probes."""
    return replace(
        base_settings,
        torch_runtime=replace(
            base_settings.torch_runtime,
            device="mps",
            dtype="auto",
        ),
        transcription=replace(
            base_settings.transcription,
            mps_admission_control_enabled=False,
        ),
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
    latencies: list[float] = []
    error_messages: list[str] = []
    successful_runs = 0
    failed_runs = 0
    mps_loaded_runs = 0
    mps_completed_runs = 0
    mps_to_cpu_failover_runs = 0
    hard_mps_oom_runs = 0

    for _ in range(iterations):
        model: object | None = None
        runtime_device_before = "cpu"
        run_started_at = time.perf_counter()
        try:
            model = load_whisper_model(profile=active_profile)
            runtime_device_before = _resolve_runtime_device_for_loaded_model(
                model=model,
                backend_id=candidate.backend_id,
            )
            if runtime_device_before == "mps":
                mps_loaded_runs += 1
            _ = transcribe_with_model(
                model=model,
                file_path=str(calibration_file),
                language=language,
                profile=active_profile,
            )
            successful_runs += 1
        except Exception as error:
            failed_runs += 1
            error_messages.append(str(error))
            if runtime_device_before == "mps" and _is_hard_mps_oom(error):
                hard_mps_oom_runs += 1
        else:
            runtime_device_after = (
                _resolve_runtime_device_for_loaded_model(
                    model=model,
                    backend_id=candidate.backend_id,
                )
                if model is not None
                else runtime_device_before
            )
            if runtime_device_before == "mps" and runtime_device_after == "mps":
                mps_completed_runs += 1
            if runtime_device_before == "mps" and runtime_device_after == "cpu":
                mps_to_cpu_failover_runs += 1
            latencies.append(time.perf_counter() - run_started_at)
        finally:
            if model is not None:
                del model
            gc.collect()

    mean_latency_seconds = statistics.fmean(latencies) if latencies else 0.0
    return RuntimeCalibrationMetrics(
        profile=candidate,
        iterations=iterations,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        mps_loaded_runs=mps_loaded_runs,
        mps_completed_runs=mps_completed_runs,
        mps_to_cpu_failover_runs=mps_to_cpu_failover_runs,
        hard_mps_oom_runs=hard_mps_oom_runs,
        mean_latency_seconds=mean_latency_seconds,
        error_messages=tuple(error_messages[:5]),
    )


def run_transcription_runtime_calibration(
    *,
    calibration_file: Path,
    language: str,
    iterations_per_profile: int = 2,
    profile_names: tuple[ArtifactProfileName, ...] = DEFAULT_CALIBRATION_PROFILES,
    report_path: Path | None = None,
) -> RuntimeCalibrationResult:
    """Runs runtime calibration probes and emits confidence-scored recommendations."""
    if iterations_per_profile <= 0:
        raise ValueError("iterations_per_profile must be greater than zero.")
    if not calibration_file.is_file():
        raise FileNotFoundError(f"Calibration audio file not found: {calibration_file}")

    original_settings = get_settings()
    apply_settings(_calibration_settings(original_settings))
    recommendations: list[RuntimeCalibrationRecommendation] = []
    try:
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
    finally:
        apply_settings(original_settings)

    output_path = (
        _runtime_calibration_report_path() if report_path is None else report_path
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
    persisted_path = _persist_profile_report(output_path, payload)
    return RuntimeCalibrationResult(
        recommendations=tuple(recommendations),
        report_path=persisted_path,
    )


def _print_profiling_summary(result: ProfilingResult) -> None:
    """Prints a concise summary for internal profiling runs."""
    print("\nTranscription default profiling summary")
    print(f"Reference files: {result.reference_files}")
    print(
        "Minimum mean accuracy gate: "
        f"{result.gate.minimum_mean_accuracy:.4f} "
        f"(baseline={result.gate.baseline_mean_accuracy:.4f}, "
        f"allowed_drop={result.gate.maximum_accuracy_drop:.4f}, "
        f"floor={result.gate.absolute_accuracy_floor:.4f})"
    )
    print(
        "profile | mean_accuracy | mean_wer | exact_match_rate | "
        "avg_latency_s | failed | error"
    )
    for summary in result.summaries:
        print(
            f"{summary.profile.name} | "
            f"{summary.mean_accuracy:.4f} | "
            f"{summary.mean_word_error_rate:.4f} | "
            f"{summary.exact_match_rate:.4f} | "
            f"{summary.average_latency_seconds:.4f} | "
            f"{summary.failed_samples} | "
            f"{summary.error_message or '-'}"
        )
    print("\nRecommendation")
    print(f"Change defaults: {result.recommendation.should_change_defaults}")
    print(f"Selected profile: {result.recommendation.selected_profile}")
    print(f"Reason: {result.recommendation.reason}")
    print(f"Report: {result.report_path}")


def _print_runtime_calibration_summary(result: RuntimeCalibrationResult) -> None:
    """Prints concise runtime-calibration recommendations."""
    print("\nTranscription runtime calibration summary")
    print(
        "profile | backend | model | recommendation | confidence | "
        "mps_loaded | mps_completed | failover | failed | mean_latency_s | reason"
    )
    for recommendation in result.recommendations:
        metrics = recommendation.metrics
        print(
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
    print(f"\nReport: {result.report_path}")


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
            profile_names=_normalize_profile_csv(args.calibration_profiles),
            report_path=report_path,
        )
        _print_runtime_calibration_summary(calibration_result)
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
    _print_profiling_summary(result)


if __name__ == "__main__":
    main()
