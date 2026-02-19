"""Tests for runtime pipeline orchestration seam."""

from collections.abc import Callable, Generator

import pytest

import ser.config as config
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.profiles import RuntimeProfile
from ser.runtime.contracts import InferenceRequest
from ser.runtime.pipeline import RuntimePipeline, create_runtime_pipeline

type TrainModelCallable = Callable[[], None]
type PredictEmotionsCallable = Callable[[str], list[EmotionSegment]]
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
        train_model=train_model,
        predict_emotions=predict_emotions,
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
    assert execution.emotions == emotions
    assert execution.transcript == transcript
    assert execution.timeline == timeline
    assert execution.timeline_csv_path == "timeline.csv"
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

    assert execution.timeline_csv_path is None


def test_create_runtime_pipeline_uses_resolved_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory should resolve medium profile when medium flag is enabled."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    settings = config.reload_settings()
    pipeline = create_runtime_pipeline(settings)
    assert pipeline.profile.name == "medium"
