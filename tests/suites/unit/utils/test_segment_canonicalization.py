"""Tests for deterministic temporal segment canonicalization."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from ser.domain import EmotionSegment
from ser.runtime.schema import SegmentPrediction
from ser.utils.segment_canonicalization import CanonicalSegment, canonicalize_segments

_PROPERTY_TEST_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    database=None,
)
_SEGMENT_LABELS = st.sampled_from(("angry", "calm", "happy", "sad"))
_TIME_STEP = st.integers(min_value=-10, max_value=20).map(lambda value: value / 2.0)
_POSITIVE_DURATION = st.integers(min_value=1, max_value=12).map(lambda value: value / 2.0)


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


@dataclass(frozen=True)
class _GeneratedSegment:
    """Minimal valid segment-like object for property-based canonicalization tests."""

    emotion: str
    start_seconds: float
    end_seconds: float
    confidence: float | None


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


@st.composite
def _generated_segment_strategy(draw: st.DrawFn) -> _GeneratedSegment:
    """Build a valid segment with deterministic half-second timing granularity."""

    start_seconds = draw(_TIME_STEP)
    duration_seconds = draw(_POSITIVE_DURATION)
    confidence = draw(
        st.one_of(
            st.none(),
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return _GeneratedSegment(
        emotion=draw(_SEGMENT_LABELS),
        start_seconds=start_seconds,
        end_seconds=start_seconds + duration_seconds,
        confidence=confidence,
    )


@_PROPERTY_TEST_SETTINGS
@given(st.lists(_generated_segment_strategy(), max_size=12))
def test_canonicalize_segments_is_idempotent(
    segments: list[_GeneratedSegment],
) -> None:
    """Canonicalization should stabilize after one pass."""

    canonical = canonicalize_segments(segments)

    assert canonicalize_segments(canonical) == canonical


@_PROPERTY_TEST_SETTINGS
@given(st.lists(_generated_segment_strategy(), max_size=12))
def test_canonicalize_segments_preserves_sorted_non_overlapping_output(
    segments: list[_GeneratedSegment],
) -> None:
    """Canonical output should remain ordered, positive-duration, and gap-safe."""

    canonical = canonicalize_segments(segments)

    assert canonical == sorted(
        canonical,
        key=lambda segment: (segment.start_seconds, segment.end_seconds),
    )
    assert all(segment.end_seconds > segment.start_seconds for segment in canonical)
    for previous, current in zip(canonical, canonical[1:], strict=False):
        assert previous.end_seconds <= current.start_seconds
        assert not (
            previous.emotion == current.emotion and previous.end_seconds == current.start_seconds
        )
