"""Tests for runtime pipeline orchestration seam."""

import sys
from collections.abc import Callable, Generator
from types import ModuleType
from typing import Any, cast

import pytest

import ser.config as config
import ser.runtime.pipeline as runtime_pipeline_module
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


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _build_test_pipeline(
    *,
    train_model: TrainModelCallable,
    predict_emotions: PredictEmotionsCallable,
    predict_emotions_detailed: PredictEmotionsDetailedCallable,
    extract_transcript: ExtractTranscriptCallable,
    build_timeline: BuildTimelineCallable,
    print_timeline: PrintTimelineCallable,
    save_timeline_to_csv: SaveTimelineCallable,
    backend_inference: Callable[[InferenceRequest], InferenceResult] | None = None,
) -> RuntimePipeline:
    """Creates a runtime pipeline with injected deterministic dependencies."""
    return RuntimePipeline(
        settings=config.reload_settings(),
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


def test_run_inference_skips_save_when_flag_is_disabled() -> None:
    """Pipeline inference should skip CSV save when save flag is false."""
    pipeline = _build_test_pipeline(
        train_model=lambda: None,
        predict_emotions=lambda _file_path: [EmotionSegment("calm", 0.0, 1.0)],
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError("Detailed path should not run when schema flag is disabled.")
        ),
        extract_transcript=lambda _file_path, _language: [
            TranscriptWord("oi", 0.0, 0.4)
        ],
        build_timeline=lambda _transcript, _emotions: [
            TimelineEntry(0.0, "calm", "oi")
        ],
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
    monkeypatch.setattr(
        runtime_pipeline_module.gc, "collect", lambda: calls.append("gc")
    )

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
            AssertionError(
                "Legacy predict path should not run when backend hook exists."
            )
        ),
        predict_emotions_detailed=lambda _file_path: (_ for _ in ()).throw(
            AssertionError(
                "Detailed legacy path should not run when backend hook exists."
            )
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
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
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
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
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
    ) -> list[TranscriptWord]:
        captured["profile"] = profile
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
    monkeypatch.setattr(
        "ser.models.emotion_model.train_model",
        lambda: called.__setitem__("fast", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        lambda: called.__setitem__("medium", True),
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert pipeline.profile.name == "medium"
    assert called["medium"] is True
    assert called["fast"] is False


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
    monkeypatch.setattr(
        "ser.models.emotion_model.train_model",
        lambda: called.__setitem__("fast", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        lambda: called.__setitem__("medium", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_model",
        lambda: called.__setitem__("accurate", True),
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert called["fast"] is False
    assert called["medium"] is False
    assert called["accurate"] is True


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
    monkeypatch.setattr(
        "ser.models.emotion_model.train_model",
        lambda: called.__setitem__("fast", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_medium_model",
        lambda: called.__setitem__("medium", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_model",
        lambda: called.__setitem__("accurate", True),
    )
    monkeypatch.setattr(
        "ser.models.emotion_model.train_accurate_research_model",
        lambda: called.__setitem__("accurate_research", True),
    )

    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    pipeline.run_training()

    assert called["fast"] is False
    assert called["medium"] is False
    assert called["accurate"] is False
    assert called["accurate_research"] is True


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
            AssertionError(
                "Legacy path should not run when detailed schema is enabled."
            )
        ),
        predict_emotions_detailed=fake_predict_emotions_detailed,
        extract_transcript=lambda _file_path, _language: [
            TranscriptWord("oi", 0.0, 0.5)
        ],
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
    request = InferenceRequest(
        file_path="sample.wav", language="en", save_transcript=False
    )
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
        extract_transcript=lambda _file_path, _language: [
            TranscriptWord("oi", 0.0, 0.5)
        ],
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
    request = InferenceRequest(
        file_path="sample.wav", language="en", save_transcript=False
    )
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
        extract_transcript=lambda _file_path, _language: [
            TranscriptWord("oi", 0.0, 0.5)
        ],
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
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            )
        )
