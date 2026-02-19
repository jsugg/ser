"""Runtime pipeline contracts for train/inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import ProfileName


@dataclass(frozen=True)
class InferenceRequest:
    """Input contract for one inference execution."""

    file_path: str
    language: str
    save_transcript: bool = False


@dataclass(frozen=True)
class InferenceExecution:
    """Output contract for one inference execution."""

    profile: ProfileName
    emotions: list[EmotionSegment]
    transcript: list[TranscriptWord]
    timeline: list[TimelineEntry]
    timeline_csv_path: str | None = None
