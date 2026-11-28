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

    ``max(timeout_floor_s, timeout_per_audio_second * duration_s)`` once the source
    duration is known, ``timeout_floor_s`` before. Array input knows it at entry; a
    path learns it while decoding, and the deadline is then extended once.
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
    """One call, in this order; the first step that decides the status ends it.

    1. Argument checks raise ``AnalysisUsageError``; ``analyze`` does them first.
    2. Decode a path (failure is ``unreadable``). A source with zero samples, or a
       source duration outside ``[min_duration_s, max_duration_s]``, is
       ``too_short`` or ``too_long`` without conversion.
    3. Convert exactly once through ``audio_utils.to_canonical_audio``; a
       ``ValueError`` there (non-finite samples) is ``unreadable``.
    4. ``silent`` when ``20 * log10(rms)`` of the canonical mono samples, full scale
       1.0, before any feature transform, is strictly below ``silence_rms_dbfs``.
    5. Extract, score, and validate with ``validated_scores``.

    ``duration_s`` and ``input_sample_rate_hz`` are the source's whenever known,
    including for ``too_short`` and ``too_long``. ``report_duration`` is called
    once, as soon as the source duration is known. Timing uses
    ``time.monotonic_ns`` floored to whole milliseconds: ``features_ms`` covers
    decode, conversion and extraction, ``predict_ms`` covers scoring; a stage that
    never started reports 0.
    """
    raise NotImplementedError("The analysis pipeline is not implemented yet (brief A1-2).")


def validated_scores(
    scores: BackendScores, labels: tuple[str, ...], duration_s: float
) -> BackendScores | None:
    """Backend output as the record carries it, or ``None`` for ``model_error``.

    Valid: utterance and frame scores have exactly ``labels`` as keys, each is
    finite and in ``[0, 1]``, and each set sums to 1 within 1e-6; frames are sorted,
    lie in ``[0, duration_s]``, and leave no gap from 0 to ``duration_s``. Valid
    score sets are returned divided by their sum.
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
    "top_label",
    "validated_scores",
]
