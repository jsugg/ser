"""Runtime pipeline seam for profile-oriented orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ser.config import AppConfig, get_settings
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile, resolve_profile
from ser.runtime.contracts import (
    BackendInferenceCallable,
    InferenceExecution,
    InferenceRequest,
)
from ser.runtime.registry import (
    RuntimeCapability,
    ensure_profile_supported,
    resolve_runtime_capability,
)
from ser.runtime.schema import InferenceResult, to_legacy_emotion_segments

type TrainModelCallable = Callable[[], None]
type PredictEmotionsCallable = Callable[[str], list[EmotionSegment]]
type PredictEmotionsDetailedCallable = Callable[[str], InferenceResult]
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
    capability: RuntimeCapability
    train_model: TrainModelCallable
    predict_emotions: PredictEmotionsCallable
    predict_emotions_detailed: PredictEmotionsDetailedCallable
    extract_transcript: ExtractTranscriptCallable
    build_timeline: BuildTimelineCallable
    print_timeline: PrintTimelineCallable
    save_timeline_to_csv: SaveTimelineCallable
    backend_inference: BackendInferenceCallable | None = None

    def run_training(self) -> None:
        """Runs the model training workflow."""
        ensure_profile_supported(self.capability)
        self.train_model()

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference and timeline generation for one audio file."""
        ensure_profile_supported(self.capability)
        detailed_result: InferenceResult | None = None
        backend_id = self.capability.backend_id
        used_backend_path = False
        output_schema_version = self.settings.schema.output_schema_version
        if self.capability.profile in {"medium", "accurate"}:
            if self.backend_inference is None:
                raise RuntimeError(
                    f"Backend hook missing for supported profile '{self.capability.profile}'."
                )
            detailed_result = self.backend_inference(request)
            output_schema_version = detailed_result.schema_version
            emotions = to_legacy_emotion_segments(detailed_result)
            used_backend_path = True
        elif self.settings.runtime_flags.new_output_schema:
            detailed_result = self.predict_emotions_detailed(request.file_path)
            output_schema_version = detailed_result.schema_version
            emotions = to_legacy_emotion_segments(detailed_result)
        else:
            emotions = self.predict_emotions(request.file_path)
        transcript = self.extract_transcript(request.file_path, request.language)
        timeline = self.build_timeline(transcript, emotions)
        self.print_timeline(timeline)
        timeline_csv_path: str | None = None
        if request.save_transcript:
            timeline_csv_path = self.save_timeline_to_csv(timeline, request.file_path)
        return InferenceExecution(
            profile=self.profile.name,
            output_schema_version=output_schema_version,
            backend_id=backend_id,
            emotions=emotions,
            transcript=transcript,
            timeline=timeline,
            used_backend_path=used_backend_path,
            timeline_csv_path=timeline_csv_path,
            detailed_result=detailed_result,
        )


def create_runtime_pipeline(settings: AppConfig | None = None) -> RuntimePipeline:
    """Creates a runtime pipeline configured from application settings."""
    active_settings = settings if settings is not None else get_settings()
    backend_hooks: dict[str, BackendInferenceCallable] = {}
    implemented_backends = frozenset({"handcrafted", *backend_hooks.keys()})
    profile = resolve_profile(active_settings)
    capability = resolve_runtime_capability(
        active_settings,
        available_backend_hooks=implemented_backends,
    )
    from ser.models.emotion_model import (
        predict_emotions,
        predict_emotions_detailed,
        train_model,
    )
    from ser.transcript import extract_transcript
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    return RuntimePipeline(
        settings=active_settings,
        profile=profile,
        capability=capability,
        train_model=train_model,
        predict_emotions=predict_emotions,
        predict_emotions_detailed=predict_emotions_detailed,
        backend_inference=backend_hooks.get(capability.backend_id),
        extract_transcript=extract_transcript,
        build_timeline=build_timeline,
        print_timeline=print_timeline,
        save_timeline_to_csv=save_timeline_to_csv,
    )
