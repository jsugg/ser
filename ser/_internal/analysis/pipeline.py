"""One analysis call, run in the worker or inline: decode to validated scores."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser._internal.analysis.backend import AnalysisBackend, BackendScores
from ser.runtime.contracts import AnalysisConfig, LanguageHint
from ser.runtime.schema import UtteranceStatus


@dataclass(frozen=True)
class AnalysisJob:
    """One validated call: a path, or mono float32 samples and their rate."""

    path: str | None
    samples: NDArray[np.float32] | None
    sample_rate_hz: int | None
    language: LanguageHint | None


@dataclass(frozen=True)
class PipelineOutcome:
    """Everything a record holds besides identity and versions."""

    status: UtteranceStatus
    scores: BackendScores | None
    duration_s: float | None
    input_sample_rate_hz: int | None
    features_ms: int
    predict_ms: int


def analysis_deadline_s(config: AnalysisConfig, duration_s: float | None) -> float:
    """Seconds from ``analyze`` entry to the deadline.

    ``timeout_floor_s`` until the duration is known, then
    ``max(timeout_floor_s, timeout_per_audio_second * duration_s)``.
    """
    raise NotImplementedError("Deadlines are not implemented yet (brief A1-2).")


def run_pipeline(
    backend: AnalysisBackend,
    job: AnalysisJob,
    config: AnalysisConfig,
    *,
    deadline_monotonic_s: Callable[[], float],
    report_duration: Callable[[float], None],
) -> PipelineOutcome:
    """Decode, canonicalise, check limits, extract, score, validate.

    Silence: ``20 * log10(rms)`` of the canonical mono samples, full scale 1.0, before
    any feature transform; ``silent`` when strictly below ``silence_rms_dbfs``.
    ``features_ms`` covers decode, conversion and extraction, ``predict_ms`` covers
    scoring; a stage that never started reports 0.
    """
    raise NotImplementedError("The analysis pipeline is not implemented yet (brief A1-2).")


def scores_are_valid(scores: BackendScores, labels: tuple[str, ...], duration_s: float) -> bool:
    """Backend output the record may carry; anything else is ``model_error``.

    Utterance and frame scores have exactly ``labels`` as keys, are finite and
    non-negative, and sum to 1 within 1e-6. Frames are sorted, lie in
    ``[0, duration_s]``, and leave no gap from 0 to ``duration_s``.
    """
    raise NotImplementedError("Score validation is not implemented yet (brief A1-2).")


def top_label(labels: Mapping[str, float]) -> tuple[str, float]:
    """Most probable label; ties go to the alphabetically first."""
    raise NotImplementedError("Top label selection is not implemented yet (brief A1-2).")


__all__ = [
    "AnalysisJob",
    "PipelineOutcome",
    "analysis_deadline_s",
    "run_pipeline",
    "scores_are_valid",
    "top_label",
]
