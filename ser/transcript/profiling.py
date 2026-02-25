"""Profiling utilities for Whisper transcription defaults on RAVDESS audio."""

from __future__ import annotations

import glob
import json
import logging
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from ser.config import (
    ArtifactProfileName,
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


def main() -> None:
    """Runs the internal transcription-default profiling workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Internal transcription default profiling utility"
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
    args = parser.parse_args()

    report_path = None if args.report_path is None else Path(args.report_path)
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
