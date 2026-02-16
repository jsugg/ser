"""Domain data structures for transcript, emotion, and timeline entities."""

from typing import NamedTuple


class TranscriptWord(NamedTuple):
    """A transcript word with start/end timing in seconds."""

    word: str
    start_seconds: float
    end_seconds: float


class EmotionSegment(NamedTuple):
    """An emotion label active over a time interval."""

    emotion: str
    start_seconds: float
    end_seconds: float


class TimelineEntry(NamedTuple):
    """A merged timeline row containing time, emotion, and speech text."""

    timestamp_seconds: float
    emotion: str
    speech: str
