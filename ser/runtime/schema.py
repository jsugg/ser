"""Versioned runtime inference schema and compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass

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
