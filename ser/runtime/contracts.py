"""Runtime pipeline contracts for train/inference orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import ProfileName
from ser.runtime.schema import InferenceResult


@dataclass(frozen=True)
class InferenceRequest:
    """Input contract for one inference execution."""

    file_path: str
    language: str
    save_transcript: bool = False
    include_transcript: bool = True


@dataclass(frozen=True)
class InferenceExecution:
    """Output contract for one inference execution."""

    profile: ProfileName
    output_schema_version: str
    backend_id: str
    emotions: list[EmotionSegment]
    transcript: list[TranscriptWord]
    timeline: list[TimelineEntry]
    used_backend_path: bool = False
    timeline_csv_path: str | None = None
    detailed_result: InferenceResult | None = None
    phase_timings_seconds: dict[str, float] = field(default_factory=dict)


type BackendInferenceCallable = Callable[[InferenceRequest], InferenceResult]
