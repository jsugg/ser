"""Tests for runtime pipeline orchestration seam."""

import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

import ser.config as config
import ser.runtime.pipeline as runtime_pipeline_module
from ser.config import AppConfig, TimelineConfig
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile
from ser.runtime.contracts import InferenceRequest
from ser.runtime.pipeline import RuntimePipeline, create_runtime_pipeline
from ser.runtime.registry import RuntimeCapability, UnsupportedProfileError
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
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
type SaveSubtitleCallable = Callable[[list[TimelineEntry], str, str, str | None], str]

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reset_ambient_settings")]


def _build_test_pipeline(
    *,
    train_model: TrainModelCallable,
    predict_emotions: PredictEmotionsCallable,
    predict_emotions_detailed: PredictEmotionsDetailedCallable,
    extract_transcript: ExtractTranscriptCallable,
    build_timeline: BuildTimelineCallable,
    print_timeline: PrintTimelineCallable,
    save_timeline_to_csv: SaveTimelineCallable,
    save_timeline_to_subtitles: SaveSubtitleCallable | None = None,
    backend_inference: Callable[[InferenceRequest], InferenceResult] | None = None,
    settings: AppConfig | None = None,
) -> RuntimePipeline:
    """Creates a runtime pipeline with injected deterministic dependencies."""
    pipeline_settings = config.reload_settings() if settings is None else settings
    return RuntimePipeline(
        settings=pipeline_settings,
        profile=RuntimeProfile(
            name="fast",
            description="Test profile",
        ),
        capability=RuntimeCapability(
            profile="fast",
            backend_id="handcrafted",
            available=True,
        ),
        train_model=train_model,
        predict_emotions=predict_emotions,
        predict_emotions_detailed=predict_emotions_detailed,
        backend_inference=backend_inference,
        extract_transcript=extract_transcript,
        build_timeline=build_timeline,
        print_timeline=print_timeline,
        save_timeline_to_csv=save_timeline_to_csv,
        save_timeline_to_subtitles=save_timeline_to_subtitles,
    )


def test_run_training_invokes_training_dependency() -> None:
    """Pipeline training should delegate to the injected train callable."""
    called = {"train": False}

    def fake_train_model() -> None:
        called["train"] = True

    pipeline = _build_test_pipeline(
        train_model=fake_train_model,
        predict_emotions=lambda _file_path: [],
        predict_emotions_detailed=lambda _file_path: InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        ),
        extract_transcript=lambda _file_path, _language: [],
        build_timeline=lambda _transcript, _emotions: [],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    pipeline.run_training()
    assert called["train"] is True


def test_run_training_scopes_pipeline_settings_for_dependencies() -> None:
    """Training dependencies should observe the pipeline's explicit settings snapshot."""
    ambient_settings = config.reload_settings()
    scoped_settings = replace(ambient_settings, default_language="pt-BR")
    captured: dict[str, object] = {}

    def fake_train_model() -> None:
        captured["active_settings"] = config.get_settings()

    pipeline = _build_test_pipeline(
        train_model=fake_train_model,
        predict_emotions=lambda _file_path: [],
        predict_emotions_detailed=lambda _file_path: InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        ),
        extract_transcript=lambda _file_path, _language: [],
        build_timeline=lambda _transcript, _emotions: [],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        settings=scoped_settings,
    )

    pipeline.run_training()

    assert captured["active_settings"] is scoped_settings
    restored_settings = config.get_settings()
    assert restored_settings is not scoped_settings
    assert restored_settings == ambient_settings


def test_run_inference_with_save_transcript_enabled() -> None:
    """Pipeline inference should call save path when requested."""
    calls: dict[str, object] = {}
    emotions = [EmotionSegment("happy", 0.0, 1.0)]
    transcript = [TranscriptWord("ola", 0.0, 0.5)]
    timeline = [TimelineEntry(0.0, "happy", "ola")]

    def fake_predict(file_path: str) -> list[EmotionSegment]:
        calls["predict"] = file_path
        return emotions

    def fake_extract(file_path: str, language: str | None) -> list[TranscriptWord]:
        calls["extract"] = (file_path, language)
        return transcript

    def fake_build(
        text_rows: list[TranscriptWord], emotion_rows: list[EmotionSegment]
    ) -> list[TimelineEntry]:
        calls["build"] = (text_rows, emotion_rows)
        return timeline

    def fake_print(timeline_rows: list[TimelineEntry]) -> None:
        calls["print"] = timeline_rows

    def fake_save(timeline_rows: list[TimelineEntry], file_path: str) -> str:
        calls["save"] = (timeline_rows, file_path)
        return "timeline.csv"

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=fake_predict,
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=fake_extract,
        build_timeline=fake_build,
        print_timeline=fake_print,
        save_timeline_to_csv=fake_save,
    )

    execution = pipeline.run_inference(
        InferenceRequest(
            file_path="sample.wav",
            language="pt",
            save_transcript=True,
        )
    )

    assert execution.profile == "fast"
    assert execution.output_schema_version == "v1"
    assert execution.backend_id == "handcrafted"
    assert execution.used_backend_path is False
    assert execution.emotions == emotions
    assert execution.transcript == transcript
    assert execution.timeline == timeline
    assert execution.timeline_csv_path == "timeline.csv"
    assert execution.detailed_result is None
    assert calls["predict"] == "sample.wav"
    assert calls["extract"] == ("sample.wav", "pt")
    assert calls["build"] == (transcript, emotions)
    assert calls["print"] == timeline
    assert calls["save"] == (timeline, "sample.wav")


def test_run_inference_with_subtitle_export_enabled() -> None:
    """Pipeline inference should export subtitles when requested."""
    calls: dict[str, object] = {}
    emotions = [EmotionSegment("happy", 0.0, 1.0)]
    transcript = [TranscriptWord("ola", 0.0, 0.5)]
    timeline = [TimelineEntry(0.0, "happy", "ola")]

    def fake_save_subtitles(
        timeline_rows: list[TimelineEntry],
        file_path: str,
        subtitle_format: str,
        output_path: str | None,
    ) -> str:
        calls["save_subtitles"] = (timeline_rows, file_path, subtitle_format, output_path)
        return "timeline.vtt"

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: emotions,
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=lambda _file_path, _language: transcript,
        build_timeline=lambda _transcript, _emotions: timeline,
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        save_timeline_to_subtitles=fake_save_subtitles,
    )

    execution = pipeline.run_inference(
        InferenceRequest(
            file_path="sample.wav",
            language="pt",
            subtitle_output_path="exports/sample.vtt",
        )
    )

    assert execution.subtitle_path == "timeline.vtt"
    assert execution.timeline_csv_path is None
    assert calls["save_subtitles"] == (
        timeline,
        "sample.wav",
        "vtt",
        "exports/sample.vtt",
    )


def test_run_inference_scopes_pipeline_settings_for_dependencies() -> None:
    """Inference dependencies should observe the pipeline's explicit settings snapshot."""
    ambient_settings = config.reload_settings()
    scoped_settings = replace(ambient_settings, default_language="pt-BR")
    captured: dict[str, object] = {}

    def fake_predict(_file_path: str) -> list[EmotionSegment]:
        captured["predict_settings"] = config.get_settings()
        return [EmotionSegment("happy", 0.0, 1.0)]

    def fake_extract(_file_path: str, _language: str | None) -> list[TranscriptWord]:
        captured["extract_settings"] = config.get_settings()
        return [TranscriptWord("ola", 0.0, 0.5)]

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=fake_predict,
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=fake_extract,
        build_timeline=lambda _transcript, _emotions: [],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        settings=scoped_settings,
    )

    pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    assert captured["predict_settings"] is scoped_settings
    assert captured["extract_settings"] is scoped_settings
    restored_settings = config.get_settings()
    assert restored_settings is not scoped_settings
    assert restored_settings == ambient_settings


def test_run_inference_skips_save_when_flag_is_disabled() -> None:
    """Pipeline inference should skip CSV save when save flag is false."""
    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: [EmotionSegment("calm", 0.0, 1.0)],
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=lambda _file_path, _language: [TranscriptWord("oi", 0.0, 0.4)],
        build_timeline=lambda _transcript, _emotions: [TimelineEntry(0.0, "calm", "oi")],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: (_ for _ in ()).throw(
            AssertionError("save_timeline_to_csv should not be called")
        ),
    )

    execution = pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    assert execution.output_schema_version == "v1"
    assert execution.backend_id == "handcrafted"
    assert execution.used_backend_path is False
    assert execution.timeline_csv_path is None
    assert execution.subtitle_path is None


def test_run_inference_rejects_subtitle_export_without_transcript() -> None:
    """Subtitle export requires transcript content and should fail when disabled."""
    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: [EmotionSegment("calm", 0.0, 1.0)],
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=lambda _file_path, _language: [TranscriptWord("oi", 0.0, 0.4)],
        build_timeline=lambda _transcript, _emotions: [TimelineEntry(0.0, "calm", "oi")],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    with pytest.raises(
        ValueError,
        match="Subtitle export requires transcript extraction",
    ):
        pipeline.run_inference(
            InferenceRequest(
                file_path="sample.wav",
                language="en",
                include_transcript=False,
                subtitle_format="srt",
            )
        )


def test_run_inference_skips_transcript_extraction_when_disabled() -> None:
    """Pipeline should build an emotion-only timeline when include_transcript is false."""
    calls: dict[str, object] = {}
    emotions = [EmotionSegment("happy", 0.0, 1.0)]

    def _fake_build(
        text_rows: list[TranscriptWord], emotion_rows: list[EmotionSegment]
    ) -> list[TimelineEntry]:
        calls["build"] = (text_rows, emotion_rows)
        return [TimelineEntry(0.0, "happy", "")]

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: emotions,
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=lambda _file_path, _language: (_ for _ in ()).throw(
            AssertionError("extract_transcript should not be called")
        ),
        build_timeline=_fake_build,
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    execution = pipeline.run_inference(
        InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
            include_transcript=False,
        )
    )

    assert execution.transcript == []
    assert calls["build"] == ([], emotions)


def test_run_inference_releases_torch_memory_before_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference should release accelerator cache before transcript extraction."""
    call_order: list[str] = []
    emotions = [EmotionSegment("calm", 0.0, 1.0)]
    transcript = [TranscriptWord("oi", 0.0, 0.4)]

    monkeypatch.setattr(
        runtime_pipeline_module,
        "_release_torch_runtime_memory_before_transcription",
        lambda: call_order.append("release"),
    )

    def _fake_extract(_file_path: str, _language: str | None) -> list[TranscriptWord]:
        call_order.append("extract")
        return transcript

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: emotions,
        predict_emotions_detailed=lambda _file_path: InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        ),
        extract_transcript=_fake_extract,
        build_timeline=lambda _transcript, _emotions: [],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    _ = pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    assert call_order == ["release", "extract"]


def test_release_torch_runtime_memory_before_transcription_empties_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch runtime cache release should be best-effort and availability-gated."""
    calls: list[str] = []
    fake_torch = ModuleType("torch")
    fake_mps = ModuleType("mps")
    fake_cuda = ModuleType("cuda")
    cast(Any, fake_mps).is_available = lambda: True
    cast(Any, fake_mps).empty_cache = lambda: calls.append("mps")
    cast(Any, fake_cuda).is_available = lambda: True
    cast(Any, fake_cuda).empty_cache = lambda: calls.append("cuda")
    cast(Any, fake_torch).mps = fake_mps
    cast(Any, fake_torch).cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runtime_pipeline_module.gc, "collect", lambda: calls.append("gc"))

    runtime_pipeline_module._release_torch_runtime_memory_before_transcription()

    assert calls == ["gc", "mps", "cuda"]


def test_run_inference_uses_backend_hook_for_fast_when_available() -> None:
    """Pipeline fast profile should route inference through backend hook when present."""
    backend_result = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="happy",
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=1.0,
                probabilities=None,
            )
        ],
        frames=[
            FramePrediction(
                start_seconds=0.0,
                end_seconds=1.0,
                emotion="happy",
                confidence=1.0,
                probabilities=None,
            )
        ],
    )
    calls: dict[str, object] = {}
    transcript = [TranscriptWord("hi", 0.0, 0.5)]
    timeline = [TimelineEntry(0.0, "happy", "hi")]

    def fake_backend_hook(request: InferenceRequest) -> InferenceResult:
        calls["request"] = request
        return backend_result

    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Legacy predict path should not run when backend hook exists.")
        ),
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed legacy path should not run when backend hook exists.")
        ),
        extract_transcript=lambda _file_path, _language: transcript,
        build_timeline=lambda _transcript, _emotions: timeline,
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        backend_inference=fake_backend_hook,
    )

    execution = pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    assert calls["request"] == InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    assert execution.used_backend_path is True
    assert execution.detailed_result == backend_result


def test_create_runtime_pipeline_uses_resolved_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should resolve medium profile when medium flag is enabled."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        "ser.runtime.backend_hooks._missing_optional_modules",
        lambda _required_modules: ("transformers",),
    )
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    assert pipeline.profile.name == "medium"
    assert pipeline.capability.profile == "medium"
    assert pipeline.capability.available is False


def test_create_runtime_pipeline_marks_medium_available_when_hook_registry_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should treat medium as available when hook registry is populated."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    def fake_medium_hook(_request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {"hf_xlsr": fake_medium_hook},
    )
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    assert pipeline.profile.name == "medium"
    assert pipeline.capability.profile == "medium"
    assert pipeline.capability.available is True
    assert pipeline.backend_inference is fake_medium_hook


def test_create_runtime_pipeline_marks_accurate_available_when_hook_registry_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should treat accurate as available when hook registry is populated."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    def fake_accurate_hook(_request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {"hf_whisper": fake_accurate_hook},
    )
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    assert pipeline.profile.name == "accurate"
    assert pipeline.capability.profile == "accurate"
    assert pipeline.capability.available is True
    assert pipeline.backend_inference is fake_accurate_hook


def test_create_runtime_pipeline_marks_accurate_research_available_when_hook_registry_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should treat accurate-research as available when hook is ready."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    monkeypatch.setenv("SER_ENABLE_RESTRICTED_BACKENDS", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    def fake_accurate_research_hook(_request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {"emotion2vec": fake_accurate_research_hook},
    )
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    assert pipeline.profile.name == "accurate-research"
    assert pipeline.capability.profile == "accurate-research"
    assert pipeline.capability.available is True
    assert pipeline.backend_inference is fake_accurate_research_hook


@pytest.mark.parametrize(
    (
        "env",
        "backend_id",
        "expected_transcription_backend_id",
        "expected_model_name",
        "expected_use_demucs",
    ),
    [
        ({}, "handcrafted", "faster_whisper", "distil-large-v3", False),
        (
            {"SER_ENABLE_MEDIUM_PROFILE": "true"},
            "hf_xlsr",
            "stable_whisper",
            "turbo",
            True,
        ),
        (
            {"SER_ENABLE_ACCURATE_PROFILE": "true"},
            "hf_whisper",
            "stable_whisper",
            "large",
            True,
        ),
        (
            {
                "SER_ENABLE_ACCURATE_PROFILE": "true",
                "SER_ENABLE_ACCURATE_RESEARCH_PROFILE": "true",
            },
            "emotion2vec",
            "stable_whisper",
            "large",
            True,
        ),
    ],
)
def test_create_runtime_pipeline_uses_profile_specific_transcription_profile(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    backend_id: str,
    expected_transcription_backend_id: str,
    expected_model_name: str,
    expected_use_demucs: bool,
) -> None:
    """Factory should bind transcript extraction to selected profile defaults."""
    monkeypatch.setattr(
        "ser.runtime.pipeline.has_known_faster_whisper_openmp_runtime_conflict",
        lambda: False,
    )
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_DEMUCS", raising=False)
    monkeypatch.delenv("WHISPER_VAD", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )
    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {
            backend_id: lambda _request: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        },
    )
    captured: dict[str, object] = {}

    def fake_extract(
        _file_path: str,
        _language: str | None,
        profile: object | None = None,
        *,
        settings: object | None = None,
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
        captured["settings"] = settings
        return []

    monkeypatch.setattr("ser.transcript.extract_transcript", fake_extract)
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    from ser.transcript import TranscriptionProfile

    profile = captured["profile"]
    assert isinstance(profile, TranscriptionProfile)
    assert profile.backend_id == expected_transcription_backend_id
    assert profile.model_name == expected_model_name
    assert profile.use_demucs is expected_use_demucs
    assert profile.use_vad is True
    assert captured["settings"] is settings


def test_create_runtime_pipeline_uses_settings_timeline_folder_for_default_subtitle_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Factory-backed subtitle export should forward the configured timeline folder."""
    monkeypatch.setattr("ser.runtime.pipeline.build_backend_hooks", lambda _settings: {})
    monkeypatch.setattr(
        "ser.models.emotion_model.predict_emotions",
        lambda _file_path, *, settings=None: [EmotionSegment("happy", 0.0, 1.0)],
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.predict_emotions_detailed",
        lambda _file_path, *, settings=None: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
    )
    monkeypatch.setattr(
        "ser.transcript.extract_transcript",
        lambda _file_path, _language, profile=None, *, settings=None: [
            TranscriptWord("hello", 0.0, 0.5)
        ],
    )
    monkeypatch.setattr(
        "ser.utils.timeline_utils.build_timeline",
        lambda _transcript, _emotions: [TimelineEntry(0.0, "happy", "hello")],
    )
    monkeypatch.setattr("ser.utils.timeline_utils.print_timeline", lambda _timeline: None)
    captured: dict[str, object] = {}

    def fake_save_timeline_to_subtitles(
        timeline: list[TimelineEntry],
        file_name: str,
        *,
        subtitle_format: str,
        output_path: str | None = None,
        timeline_config: TimelineConfig | None = None,
    ) -> str:
        captured["call"] = (timeline, file_name, subtitle_format, output_path, timeline_config)
        return "custom/sample.vtt"

    monkeypatch.setattr(
        "ser.utils.subtitles.save_timeline_to_subtitles",
        fake_save_timeline_to_subtitles,
    )
    settings = replace(
        config.reload_settings(),
        timeline=TimelineConfig(folder=tmp_path / "exports"),
    )
    pipeline = create_runtime_pipeline(settings)

    execution = pipeline.run_inference(
        InferenceRequest(
            file_path="sample.wav",
            language="en",
            subtitle_format="vtt",
        )
    )

    assert execution.subtitle_path == "custom/sample.vtt"
    assert captured["call"] == (
        [TimelineEntry(0.0, "happy", "hello")],
        "sample.wav",
        "vtt",
        None,
        settings.timeline,
    )


def test_create_runtime_pipeline_retains_faster_whisper_on_openmp_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast profile should keep faster-whisper and rely on process isolation."""
    monkeypatch.setattr(
        "ser.runtime.pipeline.has_known_faster_whisper_openmp_runtime_conflict",
        lambda: True,
    )
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )
    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {
            "handcrafted": lambda _request: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        },
    )
    captured: dict[str, object] = {}

    def fake_extract(
        _file_path: str,
        _language: str | None,
        profile: object | None = None,
        *,
        settings: object | None = None,
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
        captured["settings"] = settings
        return []

    monkeypatch.setattr("ser.transcript.extract_transcript", fake_extract)
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    from ser.transcript import TranscriptionProfile

    profile = captured["profile"]
    assert isinstance(profile, TranscriptionProfile)
    assert profile.backend_id == "faster_whisper"
    assert profile.model_name == "distil-large-v3"
    assert profile.use_demucs is False
    assert profile.use_vad is True
    assert captured["settings"] is settings


def test_create_runtime_pipeline_respects_explicit_faster_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit faster backend override should remain stable under conflict risk."""
    monkeypatch.setenv("WHISPER_BACKEND", "faster_whisper")
    monkeypatch.setenv("WHISPER_MODEL", "distil-large-v3")
    monkeypatch.setattr(
        "ser.runtime.pipeline.has_known_faster_whisper_openmp_runtime_conflict",
        lambda: True,
    )
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )
    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {
            "handcrafted": lambda _request: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        },
    )
    captured: dict[str, object] = {}

    def fake_extract(
        _file_path: str,
        _language: str | None,
        profile: object | None = None,
        *,
        settings: object | None = None,
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
        captured["settings"] = settings
        return []

    monkeypatch.setattr("ser.transcript.extract_transcript", fake_extract)
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)

    pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    from ser.transcript import TranscriptionProfile

    profile = captured["profile"]
    assert isinstance(profile, TranscriptionProfile)
    assert profile.backend_id == "faster_whisper"
    assert profile.model_name == "distil-large-v3"
    assert captured["settings"] is settings


def test_create_runtime_pipeline_uses_medium_training_callable_when_medium_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium profile should route training through medium training entrypoint."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    def fake_medium_hook(_request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    called = {"fast": False, "medium": False}

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {"hf_xlsr": fake_medium_hook},
    )
    captured_settings: dict[str, object] = {}

    def _fake_train_model(*, settings: object | None = None) -> None:
        captured_settings["fast"] = settings
        called["fast"] = True

    def _fake_train_medium_model(*, settings: object | None = None) -> None:
        captured_settings["medium"] = settings
        called["medium"] = True

    monkeypatch.setattr("ser.models.emotion_model.train_model", _fake_train_model)
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        _fake_train_medium_model,
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert pipeline.profile.name == "medium"
    assert called["medium"] is True
    assert called["fast"] is False
    assert captured_settings["medium"] is settings


def test_create_runtime_pipeline_uses_accurate_training_callable_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate profile should route training through accurate training entrypoint."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    called = {"fast": False, "medium": False, "accurate": False}

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {
            "hf_whisper": lambda _request: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        },
    )
    captured_settings: dict[str, object] = {}

    def _fake_train_model(*, settings: object | None = None) -> None:
        captured_settings["fast"] = settings
        called["fast"] = True

    def _fake_train_medium_model(*, settings: object | None = None) -> None:
        captured_settings["medium"] = settings
        called["medium"] = True

    def _fake_train_accurate_model(*, settings: object | None = None) -> None:
        captured_settings["accurate"] = settings
        called["accurate"] = True

    monkeypatch.setattr("ser.models.emotion_model.train_model", _fake_train_model)
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        _fake_train_medium_model,
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_model",
        _fake_train_accurate_model,
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert called["fast"] is False
    assert called["medium"] is False
    assert called["accurate"] is True
    assert captured_settings["accurate"] is settings


def test_create_runtime_pipeline_uses_accurate_research_training_callable_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate-research profile should route to dedicated training entrypoint."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    monkeypatch.setenv("SER_ENABLE_RESTRICTED_BACKENDS", "true")
    monkeypatch.setattr(
        "ser.runtime.registry._missing_optional_modules",
        lambda _required_modules: (),
    )

    called = {
        "fast": False,
        "medium": False,
        "accurate": False,
        "accurate_research": False,
    }

    monkeypatch.setattr(
        "ser.runtime.pipeline.build_backend_hooks",
        lambda _settings: {
            "emotion2vec": lambda _request: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        },
    )
    captured_settings: dict[str, object] = {}

    def _fake_train_model(*, settings: object | None = None) -> None:
        captured_settings["fast"] = settings
        called["fast"] = True

    def _fake_train_medium_model(*, settings: object | None = None) -> None:
        captured_settings["medium"] = settings
        called["medium"] = True

    def _fake_train_accurate_model(*, settings: object | None = None) -> None:
        captured_settings["accurate"] = settings
        called["accurate"] = True

    def _fake_train_accurate_research_model(
        *,
        settings: object | None = None,
    ) -> None:
        captured_settings["accurate_research"] = settings
        called["accurate_research"] = True

    monkeypatch.setattr("ser.models.emotion_model.train_model", _fake_train_model)
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        _fake_train_medium_model,
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_model",
        _fake_train_accurate_model,
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_research_model",
        _fake_train_accurate_research_model,
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert called["fast"] is False
    assert called["medium"] is False
    assert called["accurate"] is False
    assert called["accurate_research"] is True
    assert captured_settings["accurate_research"] is settings


def test_run_inference_uses_detailed_schema_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detailed inference path should be used when new schema flag is enabled."""
    monkeypatch.setenv("SER_ENABLE_NEW_OUTPUT_SCHEMA", "true")
    settings = config.reload_settings()
    detailed = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="angry",
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.8,
                probabilities={"angry": 0.8, "neutral": 0.2},
            )
        ],
        frames=[],
    )
    calls: dict[str, object] = {}

    def fake_predict_emotions_detailed(file_path: str) -> InferenceResult:
        calls["detailed"] = file_path
        return detailed

    pipeline = RuntimePipeline(
        settings=settings,
        profile=RuntimeProfile(name="fast", description="Test profile"),
        capability=RuntimeCapability(
            profile="fast",
            backend_id="handcrafted",
            available=True,
        ),
        train_model=lambda: None,
        predict_emotions=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Legacy path should not run when detailed schema is enabled.")
        ),
        predict_emotions_detailed=fake_predict_emotions_detailed,
        extract_transcript=lambda _file_path, _language: [TranscriptWord("oi", 0.0, 0.5)],
        build_timeline=lambda _transcript, emotions: [
            TimelineEntry(0.0, emotions[0].emotion, "oi")
        ],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    execution = pipeline.run_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )

    assert calls["detailed"] == "sample.wav"
    assert execution.output_schema_version == OUTPUT_SCHEMA_VERSION
    assert execution.backend_id == "handcrafted"
    assert execution.used_backend_path is False
    assert execution.detailed_result == detailed
    assert execution.emotions == [EmotionSegment("angry", 0.0, 1.0)]


def test_run_inference_uses_backend_hook_for_supported_medium_profile() -> None:
    """Medium profile should route through backend hook when capability is ready."""
    request = InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    calls: dict[str, object] = {}
    detailed = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="happy",
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.9,
                probabilities={"happy": 0.9, "neutral": 0.1},
            )
        ],
        frames=[],
    )

    def fake_backend_inference(inference_request: InferenceRequest) -> InferenceResult:
        calls["request"] = inference_request
        return detailed

    pipeline = RuntimePipeline(
        settings=config.reload_settings(),
        profile=RuntimeProfile(name="medium", description="Medium test profile"),
        capability=RuntimeCapability(
            profile="medium",
            backend_id="hf_xlsr",
            available=True,
        ),
        train_model=lambda: None,
        predict_emotions=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Legacy predict path should not run for backend hook flow.")
        ),
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed legacy path should not run for backend hook flow.")
        ),
        extract_transcript=lambda _file_path, _language: [TranscriptWord("oi", 0.0, 0.5)],
        build_timeline=lambda _transcript, emotions: [
            TimelineEntry(0.0, emotions[0].emotion, "oi")
        ],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        backend_inference=fake_backend_inference,
    )

    execution = pipeline.run_inference(request)

    assert calls["request"] == request
    assert execution.profile == "medium"
    assert execution.output_schema_version == OUTPUT_SCHEMA_VERSION
    assert execution.backend_id == "hf_xlsr"
    assert execution.used_backend_path is True
    assert execution.detailed_result == detailed
    assert execution.emotions == [EmotionSegment("happy", 0.0, 1.0)]


def test_run_inference_uses_backend_hook_for_supported_accurate_profile() -> None:
    """Accurate profile should route through backend hook when capability is ready."""
    request = InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    calls: dict[str, object] = {}
    detailed = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[
            SegmentPrediction(
                emotion="sad",
                start_seconds=0.0,
                end_seconds=1.0,
                confidence=0.92,
                probabilities={"sad": 0.92, "neutral": 0.08},
            )
        ],
        frames=[],
    )

    def fake_backend_inference(inference_request: InferenceRequest) -> InferenceResult:
        calls["request"] = inference_request
        return detailed

    pipeline = RuntimePipeline(
        settings=config.reload_settings(),
        profile=RuntimeProfile(name="accurate", description="Accurate test profile"),
        capability=RuntimeCapability(
            profile="accurate",
            backend_id="hf_whisper",
            available=True,
        ),
        train_model=lambda: None,
        predict_emotions=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Legacy predict path should not run for backend hook flow.")
        ),
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed legacy path should not run for backend hook flow.")
        ),
        extract_transcript=lambda _file_path, _language: [TranscriptWord("oi", 0.0, 0.5)],
        build_timeline=lambda _transcript, emotions: [
            TimelineEntry(0.0, emotions[0].emotion, "oi")
        ],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
        backend_inference=fake_backend_inference,
    )

    execution = pipeline.run_inference(request)

    assert calls["request"] == request
    assert execution.profile == "accurate"
    assert execution.output_schema_version == OUTPUT_SCHEMA_VERSION
    assert execution.backend_id == "hf_whisper"
    assert execution.used_backend_path is True
    assert execution.detailed_result == detailed
    assert execution.emotions == [EmotionSegment("sad", 0.0, 1.0)]


def test_run_inference_raises_for_unsupported_profile_capability() -> None:
    """Pipeline should fail fast for unresolved profile/backend capability."""
    pipeline = RuntimePipeline(
        settings=config.reload_settings(),
        profile=RuntimeProfile(name="medium", description="Medium test profile"),
        capability=RuntimeCapability(
            profile="medium",
            backend_id="hf_xlsr",
            available=False,
            message="medium pending",
        ),
        train_model=lambda: None,
        predict_emotions=lambda _file_path: [],
        predict_emotions_detailed=lambda _file_path: InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        ),
        extract_transcript=lambda _file_path, _language: [],
        build_timeline=lambda _transcript, _emotions: [],
        print_timeline=lambda _timeline: None,
        save_timeline_to_csv=lambda _timeline, _file_path: "unused.csv",
    )

    with pytest.raises(UnsupportedProfileError, match="medium pending"):
        pipeline.run_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
        )
