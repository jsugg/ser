"""Tests for deterministic medium segment postprocessing."""

from __future__ import annotations

import pytest

from ser.runtime.postprocessing import (
    SegmentPostprocessingConfig,
    postprocess_frame_predictions,
)
from ser.runtime.schema import FramePrediction


def _frame(
    *,
    emotion: str,
    start: float,
    end: float,
    confidence: float,
) -> FramePrediction:
    """Builds deterministic frame prediction fixtures."""
    return FramePrediction(
        start_seconds=start,
        end_seconds=end,
        emotion=emotion,
        confidence=confidence,
        probabilities={"happy": confidence, "sad": 1.0 - confidence},
    )


def test_postprocessing_smoothing_removes_isolated_label_flip() -> None:
    """Majority smoothing should collapse a one-frame label oscillation."""
    frames = [
        _frame(emotion="happy", start=0.0, end=1.0, confidence=0.9),
        _frame(emotion="sad", start=1.0, end=2.0, confidence=0.9),
        _frame(emotion="happy", start=2.0, end=3.0, confidence=0.9),
    ]

    segments = postprocess_frame_predictions(
        frames,
        config=SegmentPostprocessingConfig(
            smoothing_window_frames=3,
            hysteresis_enter_confidence=0.0,
            hysteresis_exit_confidence=0.0,
            min_segment_duration_seconds=0.0,
        ),
    )

    assert [(seg.emotion, seg.start_seconds, seg.end_seconds) for seg in segments] == [
        ("happy", 0.0, 3.0)
    ]


def test_postprocessing_hysteresis_blocks_low_confidence_transition() -> None:
    """Low-confidence label switches should be rejected by hysteresis."""
    frames = [
        _frame(emotion="happy", start=0.0, end=1.0, confidence=0.90),
        _frame(emotion="sad", start=1.0, end=2.0, confidence=0.50),
        _frame(emotion="sad", start=2.0, end=3.0, confidence=0.52),
    ]

    segments = postprocess_frame_predictions(
        frames,
        config=SegmentPostprocessingConfig(
            smoothing_window_frames=1,
            hysteresis_enter_confidence=0.60,
            hysteresis_exit_confidence=0.45,
            min_segment_duration_seconds=0.0,
        ),
    )

    assert [(seg.emotion, seg.start_seconds, seg.end_seconds) for seg in segments] == [
        ("happy", 0.0, 3.0)
    ]


def test_postprocessing_min_duration_merges_short_segments() -> None:
    """Segments shorter than minimum duration should merge into neighbors."""
    frames = [
        _frame(emotion="happy", start=0.0, end=1.0, confidence=0.90),
        _frame(emotion="sad", start=1.0, end=2.0, confidence=0.95),
        _frame(emotion="happy", start=2.0, end=3.0, confidence=0.88),
    ]

    segments = postprocess_frame_predictions(
        frames,
        config=SegmentPostprocessingConfig(
            smoothing_window_frames=1,
            hysteresis_enter_confidence=0.0,
            hysteresis_exit_confidence=0.0,
            min_segment_duration_seconds=1.5,
        ),
    )

    assert [(seg.emotion, seg.start_seconds, seg.end_seconds) for seg in segments] == [
        ("sad", 0.0, 3.0)
    ]


def test_postprocessing_rejects_invalid_hysteresis_thresholds() -> None:
    """Enter threshold must not be smaller than exit threshold."""
    with pytest.raises(ValueError, match="hysteresis_enter_confidence"):
        postprocess_frame_predictions(
            [_frame(emotion="happy", start=0.0, end=1.0, confidence=0.9)],
            config=SegmentPostprocessingConfig(
                smoothing_window_frames=1,
                hysteresis_enter_confidence=0.3,
                hysteresis_exit_confidence=0.4,
                min_segment_duration_seconds=0.0,
            ),
        )
