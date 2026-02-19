"""Runtime pipeline seam for profile-oriented orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ser.config import AppConfig, get_settings
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile, resolve_profile
from ser.runtime.contracts import InferenceExecution, InferenceRequest

type TrainModelCallable = Callable[[], None]
type PredictEmotionsCallable = Callable[[str], list[EmotionSegment]]
type ExtractTranscriptCallable = Callable[[str, str | None], list[TranscriptWord]]
type BuildTimelineCallable = Callable[
    [list[TranscriptWord], list[EmotionSegment]],
    list[TimelineEntry],
]
type PrintTimelineCallable = Callable[[list[TimelineEntry]], None]
type SaveTimelineCallable = Callable[[list[TimelineEntry], str], str]


@dataclass(frozen=True)
class RuntimePipeline:
    """Runtime pipeline wrapper around train and inference orchestration."""

    settings: AppConfig
    profile: RuntimeProfile
    train_model: TrainModelCallable
    predict_emotions: PredictEmotionsCallable
    extract_transcript: ExtractTranscriptCallable
    build_timeline: BuildTimelineCallable
    print_timeline: PrintTimelineCallable
    save_timeline_to_csv: SaveTimelineCallable

    def run_training(self) -> None:
        """Runs the model training workflow."""
        self.train_model()

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference and timeline generation for one audio file."""
        emotions = self.predict_emotions(request.file_path)
        transcript = self.extract_transcript(request.file_path, request.language)
        timeline = self.build_timeline(transcript, emotions)
        self.print_timeline(timeline)
        timeline_csv_path: str | None = None
        if request.save_transcript:
            timeline_csv_path = self.save_timeline_to_csv(timeline, request.file_path)
        return InferenceExecution(
            profile=self.profile.name,
            emotions=emotions,
            transcript=transcript,
            timeline=timeline,
            timeline_csv_path=timeline_csv_path,
        )


def create_runtime_pipeline(settings: AppConfig | None = None) -> RuntimePipeline:
    """Creates a runtime pipeline configured from application settings."""
    active_settings = settings if settings is not None else get_settings()
    from ser.models.emotion_model import predict_emotions, train_model
    from ser.transcript import extract_transcript
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    return RuntimePipeline(
        settings=active_settings,
        profile=resolve_profile(active_settings),
        train_model=train_model,
        predict_emotions=predict_emotions,
        extract_transcript=extract_transcript,
        build_timeline=build_timeline,
        print_timeline=print_timeline,
        save_timeline_to_csv=save_timeline_to_csv,
    )
