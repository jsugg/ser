"""Runtime pipeline seam for profile-oriented orchestration."""

from __future__ import annotations

import gc
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from ser._internal.config.bootstrap import resolve_profile_transcription_config
from ser._internal.runtime.environment_plan import build_runtime_environment_plan
from ser._internal.runtime.process_env import temporary_process_env
from ser.config import AppConfig, settings_override
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile, TranscriptionBackendId, resolve_profile
from ser.runtime.backend_hooks import build_backend_hooks
from ser.runtime.contracts import (
    BackendInferenceCallable,
    InferenceExecution,
    InferenceRequest,
    SubtitleFormat,
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
from ser.utils.subtitles import resolve_subtitle_export_request
from ser.utils.transcription_compat import (
    has_known_faster_whisper_openmp_runtime_conflict,
)

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
type SaveSubtitleCallable = Callable[
    [list[TimelineEntry], str, SubtitleFormat, str | None],
    str,
]

logger = get_logger(__name__)


def _invoke_training_entrypoint(
    train_fn: Callable[..., None],
    *,
    settings: AppConfig,
) -> None:
    """Calls one training entrypoint with explicit settings when supported."""

    if "settings" in inspect.signature(train_fn).parameters:
        train_fn(settings=settings)
        return
    train_fn()


def _resolve_transcription_profile_with_runtime_fallback(
    *,
    backend_id: TranscriptionBackendId,
    model_name: str,
    use_demucs: bool,
    use_vad: bool,
) -> tuple[TranscriptionBackendId, str, bool, bool]:
    """Applies deterministic runtime compatibility policy for transcription."""
    if backend_id == "faster_whisper" and has_known_faster_whisper_openmp_runtime_conflict():
        logger.info(
            "Transcription backend retained: faster_whisper "
            "(reason=openmp_runtime_conflict, mode=process_isolation)."
        )
    return backend_id, model_name, use_demucs, use_vad


def _release_torch_runtime_memory_before_transcription() -> None:
    """Releases best-effort torch accelerator cache before transcription starts."""
    gc.collect()
    torch_module = sys.modules.get("torch")
    if not isinstance(torch_module, ModuleType):
        return
    mps_module = getattr(torch_module, "mps", None)
    if isinstance(mps_module, ModuleType):
        is_available = getattr(mps_module, "is_available", None)
        empty_cache = getattr(mps_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch MPS cache before transcription.",
                exc_info=True,
            )
    cuda_module = getattr(torch_module, "cuda", None)
    if isinstance(cuda_module, ModuleType):
        is_available = getattr(cuda_module, "is_available", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch CUDA cache before transcription.",
                exc_info=True,
            )


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
    save_timeline_to_subtitles: SaveSubtitleCallable | None = None
    backend_inference: BackendInferenceCallable | None = None

    def run_training(self) -> None:
        """Runs the model training workflow."""
        ensure_profile_supported(self.capability)
        runtime_environment = build_runtime_environment_plan(self.settings)
        with (
            settings_override(self.settings),
            temporary_process_env(runtime_environment.torch_runtime),
        ):
            self.train_model()

    def run_inference(self, request: InferenceRequest) -> InferenceExecution:
        """Runs inference and timeline generation for one audio file."""
        ensure_profile_supported(self.capability)
        subtitle_export = resolve_subtitle_export_request(
            output_path=request.subtitle_output_path,
            subtitle_format=request.subtitle_format,
        )
        if subtitle_export is not None and not request.include_transcript:
            raise ValueError(
                "Subtitle export requires transcript extraction; disable subtitle export or "
                "remove the no-transcript option."
            )
        runtime_environment = build_runtime_environment_plan(self.settings)
        with (
            settings_override(self.settings),
            temporary_process_env(runtime_environment.torch_runtime),
        ):
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

            transcript: list[TranscriptWord]
            if request.include_transcript:
                _release_torch_runtime_memory_before_transcription()
                transcript = self.extract_transcript(request.file_path, request.language)
            else:
                transcript = []

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
            subtitle_path: str | None = None
            if request.save_transcript:
                timeline_csv_path = self.save_timeline_to_csv(timeline, request.file_path)
            if subtitle_export is not None:
                if self.save_timeline_to_subtitles is None:
                    raise RuntimeError("Subtitle export is unavailable for the active runtime.")
                resolved_subtitle_format, resolved_subtitle_output_path = subtitle_export
                subtitle_path = self.save_timeline_to_subtitles(
                    timeline,
                    request.file_path,
                    resolved_subtitle_format,
                    resolved_subtitle_output_path,
                )
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
                subtitle_path=subtitle_path,
                detailed_result=detailed_result,
                phase_timings_seconds=phase_timings,
            )


def create_runtime_pipeline(settings: AppConfig) -> RuntimePipeline:
    """Creates a runtime pipeline configured from application settings."""
    backend_hooks: dict[str, Callable[[InferenceRequest], InferenceResult]] = build_backend_hooks(
        settings
    )
    implemented_backends: frozenset[str] = frozenset({"handcrafted", *backend_hooks.keys()})
    profile: RuntimeProfile = resolve_profile(settings)
    capability: RuntimeCapability = resolve_runtime_capability(
        settings,
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
    from ser.utils.subtitles import save_timeline_to_subtitles
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    def selected_train_model() -> None:
        """Runs the profile-selected training entrypoint with explicit settings."""
        if capability.profile == "medium":
            _invoke_training_entrypoint(train_medium_model, settings=settings)
        elif capability.profile == "accurate":
            _invoke_training_entrypoint(train_accurate_model, settings=settings)
        elif capability.profile == "accurate-research":
            _invoke_training_entrypoint(train_accurate_research_model, settings=settings)
        else:
            _invoke_training_entrypoint(train_model, settings=settings)

    (
        transcription_backend_id,
        transcription_model_name,
        transcription_use_demucs,
        transcription_use_vad,
    ) = resolve_profile_transcription_config(capability.profile)
    (
        transcription_backend_id,
        transcription_model_name,
        transcription_use_demucs,
        transcription_use_vad,
    ) = _resolve_transcription_profile_with_runtime_fallback(
        backend_id=transcription_backend_id,
        model_name=transcription_model_name,
        use_demucs=transcription_use_demucs,
        use_vad=transcription_use_vad,
    )
    transcription_profile = TranscriptionProfile(
        backend_id=transcription_backend_id,
        model_name=transcription_model_name,
        use_demucs=transcription_use_demucs,
        use_vad=transcription_use_vad,
    )

    def extract_transcript_for_profile(
        file_path: str,
        language: str | None,
    ) -> list[TranscriptWord]:
        return extract_transcript(
            file_path,
            language,
            profile=transcription_profile,
            settings=settings,
        )

    def predict_emotions_for_settings(file_path: str) -> list[EmotionSegment]:
        return predict_emotions(file_path, settings=settings)

    def predict_emotions_detailed_for_settings(file_path: str) -> InferenceResult:
        return predict_emotions_detailed(file_path, settings=settings)

    def save_timeline_to_subtitles_for_settings(
        timeline: list[TimelineEntry],
        file_path: str,
        subtitle_format: SubtitleFormat,
        output_path: str | None,
    ) -> str:
        return save_timeline_to_subtitles(
            timeline,
            file_path,
            subtitle_format=subtitle_format,
            output_path=output_path,
        )

    return RuntimePipeline(
        settings=settings,
        profile=profile,
        capability=capability,
        train_model=selected_train_model,
        predict_emotions=predict_emotions_for_settings,
        predict_emotions_detailed=predict_emotions_detailed_for_settings,
        backend_inference=backend_hooks.get(capability.backend_id),
        extract_transcript=extract_transcript_for_profile,
        build_timeline=build_timeline,
        print_timeline=print_timeline,
        save_timeline_to_csv=save_timeline_to_csv,
        save_timeline_to_subtitles=save_timeline_to_subtitles_for_settings,
    )
