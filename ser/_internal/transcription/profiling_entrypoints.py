"""Internal entrypoints for public transcription profiling wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

from ser.profiles import TranscriptionBackendId

_TCandidate = TypeVar("_TCandidate", bound="_ProfileCandidate")
_TProfile = TypeVar("_TProfile")
_TSummary = TypeVar("_TSummary")
_TMetrics = TypeVar("_TMetrics")


class _ProfileCandidate(Protocol):
    """Minimal benchmark candidate contract for profiling entrypoints."""

    @property
    def name(self) -> str:
        """Returns candidate identifier used for reporting."""
        ...

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns transcription backend identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Returns backend model identifier."""
        ...

    @property
    def use_demucs(self) -> bool:
        """Returns whether Demucs preprocessing is enabled."""
        ...

    @property
    def use_vad(self) -> bool:
        """Returns whether VAD preprocessing is enabled."""
        ...


class _ProfileBenchmarkStats(Protocol):
    """Minimal benchmark stats contract returned by internal profiling helpers."""

    @property
    def evaluated_samples(self) -> int: ...

    @property
    def failed_samples(self) -> int: ...

    @property
    def exact_match_rate(self) -> float: ...

    @property
    def mean_word_error_rate(self) -> float: ...

    @property
    def median_word_error_rate(self) -> float: ...

    @property
    def p90_word_error_rate(self) -> float: ...

    @property
    def mean_accuracy(self) -> float: ...

    @property
    def average_latency_seconds(self) -> float: ...

    @property
    def total_runtime_seconds(self) -> float: ...

    @property
    def error_message(self) -> str | None: ...


class _RuntimeCalibrationProbeStats(Protocol):
    """Minimal calibration stats contract returned by internal probe helpers."""

    @property
    def successful_runs(self) -> int: ...

    @property
    def failed_runs(self) -> int: ...

    @property
    def mps_loaded_runs(self) -> int: ...

    @property
    def mps_completed_runs(self) -> int: ...

    @property
    def mps_to_cpu_failover_runs(self) -> int: ...

    @property
    def hard_mps_oom_runs(self) -> int: ...

    @property
    def mean_latency_seconds(self) -> float: ...

    @property
    def error_messages(self) -> tuple[str, ...]: ...


type _ProfileFactory[_TProfile] = Callable[..., _TProfile]
type _SummaryFactory[_TCandidate, _TSummary] = Callable[..., _TSummary]
type _MetricsFactory[_TCandidate, _TMetrics] = Callable[..., _TMetrics]


def profile_transcription_candidate(
    *,
    candidate: _TCandidate,
    files: Sequence[Path],
    language: str,
    profile_factory: _ProfileFactory[_TProfile],
    profile_candidate_transcriptions_fn: Callable[..., _ProfileBenchmarkStats],
    load_model: Callable[..., object],
    transcribe: Callable[..., object],
    resolve_reference_text: Callable[[Path], str | None],
    words_to_text: Callable[..., str],
    compute_word_error_rate: Callable[[str, str], float],
    percentile: Callable[[list[float], float], float],
    logger: logging.Logger,
    summary_factory: _SummaryFactory[_TCandidate, _TSummary],
) -> _TSummary:
    """Profiles one candidate and maps internal stats to the public summary type."""
    profile = profile_factory(
        backend_id=candidate.backend_id,
        model_name=candidate.model_name,
        use_demucs=candidate.use_demucs,
        use_vad=candidate.use_vad,
    )
    stats = profile_candidate_transcriptions_fn(
        candidate_name=candidate.name,
        profile=profile,
        files=files,
        language=language,
        load_model=load_model,
        transcribe=transcribe,
        resolve_reference_text=resolve_reference_text,
        words_to_text=words_to_text,
        compute_word_error_rate=compute_word_error_rate,
        percentile=percentile,
        logger=logger,
    )
    return summary_factory(
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


def calibrate_runtime_candidate(
    *,
    candidate: _TCandidate,
    calibration_file: Path,
    language: str,
    iterations: int,
    profile_factory: _ProfileFactory[_TProfile],
    run_runtime_calibration_probes_fn: Callable[..., _RuntimeCalibrationProbeStats],
    load_model: Callable[..., object],
    transcribe: Callable[..., object],
    metrics_factory: _MetricsFactory[_TCandidate, _TMetrics],
) -> _TMetrics:
    """Runs calibration probes for one candidate and maps stats to public metrics."""
    active_profile = profile_factory(
        backend_id=candidate.backend_id,
        model_name=candidate.model_name,
        use_demucs=candidate.use_demucs,
        use_vad=candidate.use_vad,
    )
    stats = run_runtime_calibration_probes_fn(
        backend_id=candidate.backend_id,
        active_profile=active_profile,
        calibration_file=calibration_file,
        language=language,
        iterations=iterations,
        load_model=load_model,
        transcribe=transcribe,
    )
    return metrics_factory(
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


__all__ = [
    "calibrate_runtime_candidate",
    "profile_transcription_candidate",
]
