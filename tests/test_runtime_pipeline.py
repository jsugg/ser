"""Tests for runtime pipeline orchestration seam."""

from collections.abc import Callable, Generator

import pytest

import ser.config as config
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile
from ser.runtime.contracts import InferenceRequest
from ser.runtime.pipeline import RuntimePipeline, create_runtime_pipeline
from ser.runtime.registry import RuntimeCapability, UnsupportedProfileError
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
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
    assert execution.timeline_csv_path is None


def test_create_runtime_pipeline_uses_resolved_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should resolve medium profile when medium flag is enabled."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    assert pipeline.profile.name == "medium"
    assert pipeline.capability.profile == "medium"
    assert pipeline.capability.available is False


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
    assert execution.detailed_result == detailed
    assert execution.emotions == [EmotionSegment("angry", 0.0, 1.0)]


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
