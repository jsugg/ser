"""Runtime pipeline seam for profile-oriented orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ser.config import AppConfig, get_settings, resolve_profile_transcription_config
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile, resolve_profile
from ser.runtime.backend_hooks import build_backend_hooks
from ser.runtime.contracts import (
    BackendInferenceCallable,
    InferenceExecution,
    InferenceRequest,
)
from ser.runtime.phase_contract import (
    PHASE_TIMELINE_BUILD,
    PHASE_TIMELINE_OUTPUT,
)
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.registry import (
    RuntimeCapability,
    ensure_profile_supported,
    resolve_runtime_capability,
)
from ser.runtime.schema import InferenceResult, to_legacy_emotion_segments
from ser.utils.logger import get_logger

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

logger = get_logger(__name__)


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
        phase_timings: dict[str, float] = {}
        detailed_result: InferenceResult | None = None
        backend_id: str = self.capability.backend_id
        used_backend_path = False
        output_schema_version: str = self.settings.schema.output_schema_version
        emotions: list[EmotionSegment]
        if self.backend_inference is not None:
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

        transcript: list[TranscriptWord] = self.extract_transcript(
            request.file_path, request.language
        )

        timeline_build_started_at = log_phase_started(
            logger,
            phase_name=PHASE_TIMELINE_BUILD,
            profile=self.profile.name,
        )
        timeline: list[TimelineEntry]
        try:
            timeline = self.build_timeline(transcript, emotions)
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_TIMELINE_BUILD,
                started_at=timeline_build_started_at,
                profile=self.profile.name,
            )
            raise
        phase_timings[PHASE_TIMELINE_BUILD] = log_phase_completed(
            logger,
            phase_name=PHASE_TIMELINE_BUILD,
            started_at=timeline_build_started_at,
            profile=self.profile.name,
        )

        timeline_output_started_at = log_phase_started(
            logger,
            phase_name=PHASE_TIMELINE_OUTPUT,
            profile=self.profile.name,
        )
        try:
            self.print_timeline(timeline)
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_TIMELINE_OUTPUT,
                started_at=timeline_output_started_at,
                profile=self.profile.name,
            )
            raise
        timeline_csv_path: str | None = None
        if request.save_transcript:
            timeline_csv_path = self.save_timeline_to_csv(timeline, request.file_path)
        phase_timings[PHASE_TIMELINE_OUTPUT] = log_phase_completed(
            logger,
            phase_name=PHASE_TIMELINE_OUTPUT,
            started_at=timeline_output_started_at,
            profile=self.profile.name,
        )
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
            phase_timings_seconds=phase_timings,
        )


def create_runtime_pipeline(settings: AppConfig | None = None) -> RuntimePipeline:
    """Creates a runtime pipeline configured from application settings."""
    active_settings: AppConfig = settings if settings is not None else get_settings()
    backend_hooks: dict[str, Callable[[InferenceRequest], InferenceResult]] = (
        build_backend_hooks(active_settings)
    )
    implemented_backends: frozenset[str] = frozenset(
        {"handcrafted", *backend_hooks.keys()}
    )
    profile: RuntimeProfile = resolve_profile(active_settings)
    capability: RuntimeCapability = resolve_runtime_capability(
        active_settings,
        available_backend_hooks=implemented_backends,
    )
    from ser.models.emotion_model import (
        predict_emotions,
        predict_emotions_detailed,
        train_accurate_model,
        train_accurate_research_model,
        train_medium_model,
        train_model,
    )
    from ser.transcript import TranscriptionProfile, extract_transcript
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    if capability.profile == "medium":
        selected_train_model: Callable[[], None] = train_medium_model
    elif capability.profile == "accurate":
        selected_train_model = train_accurate_model
    elif capability.profile == "accurate-research":
        selected_train_model = train_accurate_research_model
    else:
        selected_train_model = train_model
    transcription_model_name, transcription_use_demucs, transcription_use_vad = (
        resolve_profile_transcription_config(capability.profile)
    )
    transcription_profile = TranscriptionProfile(
        model_name=transcription_model_name,
        use_demucs=transcription_use_demucs,
        use_vad=transcription_use_vad,
    )

    def extract_transcript_for_profile(
        file_path: str,
        language: str | None,
    ) -> list[TranscriptWord]:
        return extract_transcript(file_path, language, profile=transcription_profile)

    return RuntimePipeline(
        settings=active_settings,
        profile=profile,
        capability=capability,
        train_model=selected_train_model,
        predict_emotions=predict_emotions,
        predict_emotions_detailed=predict_emotions_detailed,
        backend_inference=backend_hooks.get(capability.backend_id),
        extract_transcript=extract_transcript_for_profile,
        build_timeline=build_timeline,
        print_timeline=print_timeline,
        save_timeline_to_csv=save_timeline_to_csv,
    )
