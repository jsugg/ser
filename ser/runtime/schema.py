"""Versioned runtime inference schema and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypedDict

from ser.domain import EmotionSegment

OUTPUT_SCHEMA_VERSION = "v1"
ARTIFACT_SCHEMA_VERSION = "v2"


@dataclass(frozen=True)
class FramePrediction:
    """One frame-level inference prediction."""

    start_seconds: float
    end_seconds: float
    emotion: str
    confidence: float
    probabilities: dict[str, float] | None


@dataclass(frozen=True)
class SegmentPrediction:
    """Merged segment-level inference prediction."""

    emotion: str
    start_seconds: float
    end_seconds: float
    confidence: float
    probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class InferenceResult:
    """Full inference payload with frame and segment predictions."""

    schema_version: str
    segments: list[SegmentPrediction]
    frames: list[FramePrediction]


def to_legacy_emotion_segments(result: InferenceResult) -> list[EmotionSegment]:
    """Converts detailed inference output to legacy emotion segments."""
    return [
        EmotionSegment(
            emotion=segment.emotion,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
        )
        for segment in result.segments
    ]


UTTERANCE_SCHEMA_VERSION: Final = "ser.utterance/v1"

type UtteranceStatus = Literal[
    "ok",
    "too_short",
    "too_long",
    "silent",
    "unreadable",
    "timeout",
    "model_error",
]


class UtteranceFrame(TypedDict):
    """Label probabilities for one model window inside an utterance."""

    start_s: float
    end_s: float
    labels: dict[str, float]


class UtteranceAudio(TypedDict):
    """Facts about the analysed input; ``None`` when the input could not be decoded."""

    duration_s: float | None
    input_sample_rate_hz: int | None


class UtteranceTiming(TypedDict):
    """Wall-clock milliseconds spent per stage; excluded from determinism."""

    features: int
    predict: int


class UtteranceRecord(TypedDict):
    """One ``ser.utterance/v1`` analysis record (see docs/adr/0001-utterance-analysis.md).

    Every status other than ``"ok"`` carries ``None`` in ``labels``, ``top_label``,
    ``confidence`` and ``frames``.
    """

    schema_version: Literal["ser.utterance/v1"]
    ser_version: str
    model_id: str
    model_sha256: str
    label_set_id: str
    labels: dict[str, float] | None
    top_label: str | None
    confidence: float | None
    frames: list[UtteranceFrame] | None
    dimensions: None
    audio: UtteranceAudio
    status: UtteranceStatus
    timing_ms: UtteranceTiming
