"""Behavior tests for timeline merging utilities."""

from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.utils.timeline_utils import build_timeline


def test_build_timeline_aligns_words_to_emotion_intervals() -> None:
    """Words should inherit emotion based on time-interval membership."""
    text = [
        TranscriptWord("Hello", 0.3000000000004, 0.45),
        TranscriptWord("world", 0.75, 0.9),
    ]
    emotions = [EmotionSegment("happy", 0.0, 0.5), EmotionSegment("sad", 0.5, 1.0)]

    timeline = build_timeline(text, emotions)
    indexed = {
        round(row.timestamp_seconds, 3): (row.emotion, row.speech) for row in timeline
    }

    assert indexed[0.0] == ("happy", "")
    assert indexed[0.3] == ("happy", "Hello")
    assert indexed[0.5] == ("sad", "")
    assert indexed[0.75] == ("sad", "world")


def test_build_timeline_joins_words_with_same_timestamp() -> None:
    """Multiple words at the same timestamp should be combined predictably."""
    text = [TranscriptWord("hello", 0.1, 0.2), TranscriptWord("there", 0.1, 0.2)]
    emotions = [EmotionSegment("calm", 0.0, 1.0)]

    timeline = build_timeline(text, emotions)
    indexed = {
        round(row.timestamp_seconds, 3): (row.emotion, row.speech) for row in timeline
    }

    assert indexed[0.1] == ("calm", "hello there")


def test_build_timeline_skips_invalid_emotion_segments() -> None:
    """Emotion segments with inverted intervals should be ignored."""
    text = [TranscriptWord("word", 1.2, 1.3)]
    emotions = [EmotionSegment("angry", 2.0, 1.5)]

    timeline = build_timeline(text, emotions)

    assert timeline == [TimelineEntry(1.2, "", "word")]


def test_build_timeline_returns_empty_for_empty_inputs() -> None:
    """No transcript and no emotions should produce an empty timeline."""
    assert build_timeline([], []) == []


def test_build_timeline_keeps_last_emotion_on_terminal_boundary() -> None:
    """A word exactly on the final boundary should keep the final emotion."""
    text = [TranscriptWord("done", 1.0, 1.1)]
    emotions = [EmotionSegment("calm", 0.0, 1.0)]

    timeline = build_timeline(text, emotions)
    indexed = {
        round(row.timestamp_seconds, 3): (row.emotion, row.speech) for row in timeline
    }

    assert indexed[1.0] == ("calm", "done")


def test_build_timeline_canonicalizes_overlaps_before_render() -> None:
    """Overlapping labels should be canonicalized into one active label per time."""
    text: list[TranscriptWord] = []
    emotions = [
        EmotionSegment("happy", 0.0, 1.0),
        EmotionSegment("sad", 0.8, 1.5),
    ]

    timeline = build_timeline(text, emotions)
    indexed = {
        round(row.timestamp_seconds, 3): (row.emotion, row.speech) for row in timeline
    }

    assert indexed[0.0] == ("happy", "")
    assert indexed[0.8] == ("sad", "")
    assert indexed[1.5] == ("sad", "")


def test_build_timeline_same_start_conflict_uses_lexical_tie_break() -> None:
    """Conflicting labels at the same start use deterministic lexical tie-break."""
    timeline = build_timeline(
        text_with_timestamps=[],
        emotion_with_timestamps=[
            EmotionSegment("sad", 0.0, 1.0),
            EmotionSegment("happy", 0.0, 1.0),
        ],
    )

    assert timeline == [
        TimelineEntry(0.0, "happy", ""),
        TimelineEntry(1.0, "happy", ""),
    ]
