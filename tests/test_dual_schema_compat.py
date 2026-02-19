"""Compatibility tests between detailed schema and legacy segment API."""

import pytest

from ser.domain import EmotionSegment
from ser.models import emotion_model as em
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    InferenceResult,
    SegmentPrediction,
    to_legacy_emotion_segments,
)


def test_to_legacy_emotion_segments_preserves_segment_boundaries() -> None:
    """Schema adapter should preserve emotion and boundary values exactly."""
    result = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="happy",
                start_seconds=0.0,
                end_seconds=1.5,
                confidence=0.9,
                probabilities={"happy": 0.9, "sad": 0.1},
            ),
            SegmentPrediction(
                emotion="sad",
                start_seconds=1.5,
                end_seconds=3.0,
                confidence=0.7,
                probabilities={"happy": 0.2, "sad": 0.8},
            ),
        ],
        frames=[],
    )

    assert to_legacy_emotion_segments(result) == [
        EmotionSegment("happy", 0.0, 1.5),
        EmotionSegment("sad", 1.5, 3.0),
    ]


def test_predict_emotions_uses_detailed_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy entrypoint should delegate to detailed inference adapter."""
    payload = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="neutral",
                start_seconds=0.25,
                end_seconds=2.0,
                confidence=0.8,
                probabilities=None,
            )
        ],
        frames=[],
    )
    monkeypatch.setattr(em, "predict_emotions_detailed", lambda _file: payload)

    assert em.predict_emotions("sample.wav") == [EmotionSegment("neutral", 0.25, 2.0)]
