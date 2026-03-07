"""Runtime-calibration helpers used by transcription profiling workflows."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

from ser.config import AppConfig, ArtifactProfileName
from ser.profiles import TranscriptionBackendId
from ser.transcript.transcript_extractor import TranscriptionProfile

type RecommendationConfidence = Literal["high", "medium", "low"]
type RuntimeRecommendation = Literal["prefer_cpu", "prefer_mps", "mps_with_failover"]


class _CalibrationMetricsProfile(Protocol):
    """Minimal profile contract consumed by runtime recommendation logic."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns transcription backend id for recommendation derivation."""
        ...


class _RuntimeCalibrationMetrics(Protocol):
    """Minimal metrics contract consumed by runtime recommendation logic."""

    @property
    def profile(self) -> _CalibrationMetricsProfile:
        """Returns profile metadata captured for this metrics sample."""
        ...

    @property
    def iterations(self) -> int:
        """Returns total calibration iterations executed."""
        ...

    @property
    def mps_loaded_runs(self) -> int:
        """Returns number of runs admitted to MPS at model load."""
        ...

    @property
    def mps_completed_runs(self) -> int:
        """Returns number of runs that stayed on MPS through completion."""
        ...

    @property
    def mps_to_cpu_failover_runs(self) -> int:
        """Returns number of runs that fell back from MPS to CPU."""
        ...

    @property
    def failed_runs(self) -> int:
        """Returns number of failed calibration runs."""
        ...

    @property
    def hard_mps_oom_runs(self) -> int:
        """Returns number of hard MPS out-of-memory failures."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeCalibrationProbeStats:
    """Aggregated calibration probe metrics for one candidate profile."""

    successful_runs: int
    failed_runs: int
    mps_loaded_runs: int
    mps_completed_runs: int
    mps_to_cpu_failover_runs: int
    hard_mps_oom_runs: int
    mean_latency_seconds: float
    error_messages: tuple[str, ...]


def runtime_calibration_report_path(settings: AppConfig) -> Path:
    """Returns default output path for runtime calibration reports."""
    return settings.models.folder / "transcription_runtime_calibration_report.json"


def normalize_calibration_profile_csv(
    raw_profiles: str,
) -> tuple[ArtifactProfileName, ...]:
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


def resolve_runtime_device_for_loaded_model(
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


def is_hard_mps_oom(error: Exception) -> bool:
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


def derive_runtime_recommendation_from_metrics(
    metrics: _RuntimeCalibrationMetrics,
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


def build_runtime_calibration_settings(base_settings: AppConfig) -> AppConfig:
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


def run_runtime_calibration_probes(
    *,
    backend_id: TranscriptionBackendId,
    active_profile: TranscriptionProfile,
    calibration_file: Path,
    language: str,
    iterations: int,
    load_model: Callable[[TranscriptionProfile | None], object],
    transcribe: Callable[[object, str, str, TranscriptionProfile | None], object],
) -> RuntimeCalibrationProbeStats:
    """Runs iterative runtime probes for one profile/model candidate."""
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
            model = load_model(active_profile)
            runtime_device_before = resolve_runtime_device_for_loaded_model(
                model=model,
                backend_id=backend_id,
            )
            if runtime_device_before == "mps":
                mps_loaded_runs += 1
            _ = transcribe(model, str(calibration_file), language, active_profile)
            successful_runs += 1
        except Exception as error:
            failed_runs += 1
            error_messages.append(str(error))
            if runtime_device_before == "mps" and is_hard_mps_oom(error):
                hard_mps_oom_runs += 1
        else:
            runtime_device_after = (
                resolve_runtime_device_for_loaded_model(
                    model=model,
                    backend_id=backend_id,
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
    return RuntimeCalibrationProbeStats(
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        mps_loaded_runs=mps_loaded_runs,
        mps_completed_runs=mps_completed_runs,
        mps_to_cpu_failover_runs=mps_to_cpu_failover_runs,
        hard_mps_oom_runs=hard_mps_oom_runs,
        mean_latency_seconds=mean_latency_seconds,
        error_messages=tuple(error_messages[:5]),
    )


__all__ = [
    "RuntimeCalibrationProbeStats",
    "build_runtime_calibration_settings",
    "derive_runtime_recommendation_from_metrics",
    "is_hard_mps_oom",
    "normalize_calibration_profile_csv",
    "resolve_runtime_device_for_loaded_model",
    "run_runtime_calibration_probes",
    "runtime_calibration_report_path",
]
