"""One analysis call, run in the worker or inline: decode to validated scores."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser._internal.analysis import audio_input
from ser._internal.analysis.backend import AnalysisBackend, BackendFrame, BackendScores
from ser.runtime.contracts import AnalysisConfig, LanguageHint
from ser.runtime.schema import UtteranceStatus

_LOGGER = logging.getLogger(__name__)


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
    return max(config.timeout_floor_s, config.timeout_per_audio_second * (duration_s or 0.0))


def run_pipeline(
    backend: AnalysisBackend,
    job: AnalysisJob,
    config: AnalysisConfig,
    *,
    deadline_monotonic_s: Callable[[], float],
    report_duration: Callable[[float, int], None],
    labels: tuple[str, ...],
    report_features: Callable[[int, int], None] | None = None,
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
    started_ns = time.monotonic_ns()
    predict_ns: int | None = None
    features_ms = 0
    rate = job.sample_rate_hz
    duration = len(job.samples) / rate if job.samples is not None and rate is not None else None

    def finish(status: UtteranceStatus, scores: BackendScores | None = None) -> PipelineOutcome:
        end_ns = time.monotonic_ns()
        if time.monotonic() >= deadline_monotonic_s():
            status, scores = "timeout", None
        _LOGGER.debug("Analysis pipeline completed: %s", status)
        return PipelineOutcome(
            status,
            scores,
            duration,
            rate,
            (end_ns - started_ns) // 1_000_000 if predict_ns is None else features_ms,
            (end_ns - predict_ns) // 1_000_000 if predict_ns is not None else 0,
        )

    if time.monotonic() >= deadline_monotonic_s():
        return PipelineOutcome("timeout", None, duration, rate, 0, 0)
    if job.path is not None:
        decoded = audio_input.decode_file(job.path)
        if isinstance(decoded, audio_input.UnreadableAudio):
            return finish("unreadable")
        rate = decoded.sample_rate_hz
        duration = len(decoded.samples) / rate
        report_duration(duration, rate)
    else:
        assert job.samples is not None and rate is not None
        assert duration is not None
        decoded = audio_input.DecodedAudio(job.samples, rate)
    if time.monotonic() >= deadline_monotonic_s():
        return finish("timeout")
    if duration < config.min_duration_s:
        return finish("too_short")
    if duration > config.max_duration_s:
        return finish("too_long")
    try:
        canonical = audio_input.canonicalize(decoded)
    except ValueError:
        return finish("unreadable")
    if time.monotonic() >= deadline_monotonic_s():
        return finish("timeout")
    mean_square = float(np.mean(np.square(canonical.samples, dtype=np.float64)))
    if mean_square == 0.0 or 20 * math.log10(math.sqrt(mean_square)) < config.silence_rms_dbfs:
        return finish("silent")
    if time.monotonic() >= deadline_monotonic_s():
        return finish("timeout")
    try:
        features = backend.extract(canonical, deadline_monotonic_s=deadline_monotonic_s())
        if time.monotonic() >= deadline_monotonic_s():
            return finish("timeout")
        predict_ns = time.monotonic_ns()
        features_ms = (predict_ns - started_ns) // 1_000_000
        if report_features is not None:
            report_features(features_ms, predict_ns)
        raw_scores = backend.score(
            features, language=job.language, deadline_monotonic_s=deadline_monotonic_s()
        )
        scores = validated_scores(raw_scores, labels, duration)
    except MemoryError:
        raise
    except Exception:
        return finish("model_error")
    return finish("ok", scores) if scores is not None else finish("model_error")


def validated_scores(
    scores: BackendScores, labels: tuple[str, ...], duration_s: float
) -> BackendScores | None:
    """Backend output as the record carries it, or ``None`` for ``model_error``.

    Valid: utterance and frame scores have exactly ``labels`` as keys, each is
    finite and in ``[0, 1]``, and each set sums to 1 within 1e-6; frames are sorted,
    lie in ``[0, duration_s]``, and leave no gap from 0 to ``duration_s``. Valid
    score sets are returned divided by their sum.
    """
    label_set = set(labels)

    def normalize(values: Mapping[str, float]) -> dict[str, float] | None:
        if values.keys() != label_set or any(
            not (math.isfinite(value) and 0 <= value <= 1) for value in values.values()
        ):
            return None
        total = math.fsum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            return None
        return {key: value / total for key, value in values.items()}

    utterance = normalize(scores.labels)
    if utterance is None:
        return None
    frames: list[BackendFrame] = []
    covered_to = previous_start = 0.0
    for frame in scores.frames:
        if not previous_start <= frame.start_s <= covered_to:
            return None
        if not frame.start_s < frame.end_s <= duration_s:
            return None
        normalized = normalize(frame.labels)
        if normalized is None:
            return None
        frames.append(BackendFrame(float(frame.start_s), float(frame.end_s), normalized))
        previous_start = frame.start_s
        covered_to = max(covered_to, frame.end_s)
    if not frames or covered_to != duration_s:
        return None
    return BackendScores(utterance, tuple(frames))


def top_label(labels: Mapping[str, float]) -> tuple[str, float]:
    """Most probable label; ties go to the alphabetically first."""
    label = min(labels, key=lambda name: (-labels[name], name))
    return label, labels[label]


__all__ = [
    "AnalysisJob",
    "PipelineOutcome",
    "analysis_deadline_s",
    "run_pipeline",
    "top_label",
    "validated_scores",
]
