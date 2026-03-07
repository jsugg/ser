"""Default-profile benchmark helpers for transcription profiling workflows."""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ser.domain import TranscriptWord
from ser.transcript.transcript_extractor import TranscriptionProfile

type LoadModelFn = Callable[[TranscriptionProfile | None], object]
type TranscribeFn = Callable[
    [object, str, str, TranscriptionProfile | None],
    list[TranscriptWord],
]
type ResolveReferenceTextFn = Callable[[Path], str | None]
type WordsToTextFn = Callable[[list[TranscriptWord]], str]
type WordErrorRateFn = Callable[[str, str], float]
type PercentileFn = Callable[[list[float], float], float]


@dataclass(frozen=True, slots=True)
class CandidateProfileBenchmarkStats:
    """Aggregated benchmark metrics for one candidate profiling run."""

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


def profile_candidate_transcriptions(
    *,
    candidate_name: str,
    profile: TranscriptionProfile,
    files: Sequence[Path],
    language: str,
    load_model: LoadModelFn,
    transcribe: TranscribeFn,
    resolve_reference_text: ResolveReferenceTextFn,
    words_to_text: WordsToTextFn,
    compute_word_error_rate: WordErrorRateFn,
    percentile: PercentileFn,
    logger: logging.Logger,
) -> CandidateProfileBenchmarkStats:
    """Profiles one candidate profile over a list of reference audio files."""
    start_time = time.perf_counter()
    if not files:
        return CandidateProfileBenchmarkStats(
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

    try:
        model = load_model(profile)
    except Exception as err:
        logger.error(
            "Failed to load Whisper model for profile %s: %s",
            candidate_name,
            err,
        )
        return CandidateProfileBenchmarkStats(
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
    try:
        for file_path in files:
            reference_text = resolve_reference_text(file_path)
            if reference_text is None:
                failed_samples += 1
                continue

            sample_start = time.perf_counter()
            try:
                words = transcribe(model, str(file_path), language, profile)
            except Exception as err:
                failed_samples += 1
                logger.warning("Transcription failed for %s: %s", file_path, err)
                continue
            sample_latency = time.perf_counter() - sample_start

            hypothesis_text = words_to_text(words)
            sample_wer = compute_word_error_rate(reference_text, hypothesis_text)
            sample_accuracy = max(0.0, 1.0 - sample_wer)

            word_error_rates.append(sample_wer)
            accuracies.append(sample_accuracy)
            latencies.append(sample_latency)
            if sample_wer == 0.0:
                exact_matches += 1
    finally:
        del model

    evaluated_samples = len(word_error_rates)
    if evaluated_samples == 0:
        return CandidateProfileBenchmarkStats(
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

    return CandidateProfileBenchmarkStats(
        evaluated_samples=evaluated_samples,
        failed_samples=failed_samples,
        exact_match_rate=exact_matches / float(evaluated_samples),
        mean_word_error_rate=statistics.fmean(word_error_rates),
        median_word_error_rate=statistics.median(word_error_rates),
        p90_word_error_rate=percentile(word_error_rates, 0.90),
        mean_accuracy=statistics.fmean(accuracies),
        average_latency_seconds=statistics.fmean(latencies),
        total_runtime_seconds=time.perf_counter() - start_time,
        error_message=None,
    )


__all__ = [
    "CandidateProfileBenchmarkStats",
    "profile_candidate_transcriptions",
]
