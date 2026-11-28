"""Backend seam: one model implementation per profile, loaded once per process."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ser._internal.utils.audio_utils import CanonicalAudio
from ser.config import FeatureFlags
from ser.profiles import ProfileName
from ser.runtime.contracts import LanguageHint, ModelFormat, ModelManifest


@dataclass(frozen=True)
class BackendFrame:
    """Scores for one model window, in seconds of canonical audio. Descriptive only."""

    start_s: float
    end_s: float
    labels: Mapping[str, float]


@dataclass(frozen=True)
class BackendScores:
    """Backend output. ``labels`` is the one authority for the utterance scores."""

    labels: Mapping[str, float]
    frames: tuple[BackendFrame, ...]


class AnalysisBackend(Protocol):
    """A loaded model; holds no per-call state."""

    def extract(self, audio: CanonicalAudio, *, deadline_monotonic_s: float) -> object:
        """Computes model features from canonical audio."""
        ...

    def score(
        self,
        features: object,
        *,
        language: LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> BackendScores:
        """Scores features; ``language`` is the caller's validated hint."""
        ...

    def close(self) -> None:
        """Releases the model."""
        ...


@dataclass(frozen=True)
class BackendLoadRequest:
    """Everything a loader receives; sent to the worker process on every start.

    ``max_threads`` is ``None`` in inline isolation: a loader then leaves
    process-wide thread settings alone. In the worker those are already capped.
    """

    artifact_format: ModelFormat
    payload: bytes
    manifest: ModelManifest
    max_threads: int | None
    device: Literal["cpu"] = "cpu"
    feature_flags: FeatureFlags = field(default_factory=FeatureFlags)
    enabled_features: tuple[str, ...] | None = None


# Loaders must be module-level callables: they pickle by reference into the worker.
type BackendLoader = Callable[[BackendLoadRequest], AnalysisBackend]


def backend_loaders() -> Mapping[ProfileName, BackendLoader]:
    """Returns the loader for each profile that analysis supports."""
    return {"fast": _load_fast}


def _load_fast(request: BackendLoadRequest) -> AnalysisBackend:
    from ser._internal.analysis.fast import load_fast

    return load_fast(request)


def frame_layout(
    duration_s: float, *, frame_size_s: float, stride_s: float
) -> tuple[tuple[float, float], ...]:
    """Windows of the ``fast`` backend.

    Frame ``k`` starts at ``k * stride_s`` and ends at ``min(start + frame_size_s,
    duration_s)``; frames are added until one reaches ``duration_s``. Audio no longer
    than one frame gives the single frame ``(0, duration_s)``.
    """
    spans: list[tuple[float, float]] = []
    while True:
        start = len(spans) * stride_s
        end = min(start + frame_size_s, duration_s)
        spans.append((start, end))
        if end == duration_s:
            return tuple(spans)


def coverage_weighted_mean(frames: tuple[BackendFrame, ...], duration_s: float) -> dict[str, float]:
    """The ``fast`` pooling rule: every instant of the utterance weighs the same.

    Frame ``f`` weighs ``w_f``, the integral over ``f`` of ``1 / c(t)``, where ``c(t)``
    counts the frames covering ``t``; the result is ``sum(w_f * p_f) / duration_s``.
    """
    edges = sorted({edge for frame in frames for edge in (frame.start_s, frame.end_s)})
    totals = dict.fromkeys(frames[0].labels, 0.0)
    for left, right in zip(edges, edges[1:], strict=False):
        covering = [frame for frame in frames if frame.start_s <= left and frame.end_s >= right]
        if not covering:
            continue
        weight = (right - left) / len(covering)
        for frame in covering:
            for label, probability in frame.labels.items():
                totals[label] += weight * probability
    return {label: total / duration_s for label, total in totals.items()}


__all__ = [
    "AnalysisBackend",
    "BackendFrame",
    "BackendLoadRequest",
    "BackendLoader",
    "BackendScores",
    "backend_loaders",
    "coverage_weighted_mean",
    "frame_layout",
]
