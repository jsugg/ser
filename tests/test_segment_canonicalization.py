"""Tests for deterministic temporal segment canonicalization."""

from __future__ import annotations

from dataclasses import dataclass

from ser.domain import EmotionSegment
from ser.runtime.schema import SegmentPrediction
from ser.utils.segment_canonicalization import CanonicalSegment, canonicalize_segments


def test_canonicalize_segments_merges_same_label_overlap_and_adjacency() -> None:
    """Same-label overlap/adjacency should collapse into one canonical segment."""
    canonical = canonicalize_segments(
        [
            EmotionSegment("calm", 0.0, 1.0),
            EmotionSegment("calm", 0.5, 1.5),
            EmotionSegment("calm", 1.5, 2.0),
        ]
    )

    assert canonical == [CanonicalSegment("calm", 0.0, 2.0)]


def test_canonicalize_segments_truncates_previous_label_on_overlap() -> None:
    """Overlapping different labels should switch at the newer start timestamp."""
    canonical = canonicalize_segments(
        [
            EmotionSegment("happy", 0.0, 1.0),
            EmotionSegment("sad", 0.8, 1.5),
        ]
    )

    assert canonical == [
        CanonicalSegment("happy", 0.0, 0.8),
        CanonicalSegment("sad", 0.8, 1.5),
    ]


def test_canonicalize_segments_same_start_prefers_higher_confidence() -> None:
    """Conflicting labels at same start should prefer higher confidence."""
    canonical = canonicalize_segments(
        [
            SegmentPrediction("sad", 0.0, 1.0, confidence=0.65),
            SegmentPrediction("happy", 0.0, 1.0, confidence=0.80),
        ]
    )

    assert canonical == [CanonicalSegment("happy", 0.0, 1.0)]


def test_canonicalize_segments_same_start_without_confidence_is_lexical() -> None:
    """Same-start conflicts without confidence use lexical label ordering."""
    canonical = canonicalize_segments(
        [
            EmotionSegment("sad", 0.0, 1.0),
            EmotionSegment("happy", 0.0, 1.0),
        ]
    )

    assert canonical == [CanonicalSegment("happy", 0.0, 1.0)]


@dataclass(frozen=True)
class _ConfidenceCarrier:
    """Minimal segment-like object for invalid confidence coverage."""

    emotion: str
    start_seconds: float
    end_seconds: float
    confidence: object


def test_canonicalize_segments_ignores_invalid_or_nonfinite_inputs() -> None:
    """Invalid or non-finite segments should be discarded during normalization."""
    canonical = canonicalize_segments(
        [
            EmotionSegment("", 0.0, 1.0),
            EmotionSegment("happy", 1.0, 1.0),
            EmotionSegment("sad", 2.0, 1.0),
            _ConfidenceCarrier("happy", 0.0, 1.0, confidence=float("nan")),
        ]
    )

    assert canonical == [CanonicalSegment("happy", 0.0, 1.0)]
