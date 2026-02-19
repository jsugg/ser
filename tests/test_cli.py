"""Behavior tests for CLI argument dispatch and exit semantics."""

from types import SimpleNamespace
from typing import cast

import pytest

import ser.__main__ as cli
import ser.models.emotion_model as emotion_model
import ser.transcript as transcript_module
import ser.utils.timeline_utils as timeline_utils
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.runtime import InferenceRequest


def _patch_common_cli_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches shared CLI dependencies to keep tests deterministic."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(default_language="en"),
    )


def test_cli_exits_with_error_when_file_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should return exit code 1 when no prediction file is provided."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1


def test_cli_train_option_invokes_training_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--train` should dispatch to training workflow and exit successfully."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train"])

    called = {"train": False}

    def fake_train_model() -> None:
        called["train"] = True

    monkeypatch.setattr(emotion_model, "train_model", fake_train_model)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert called["train"] is True


def test_cli_prediction_passes_language_and_saves_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should pass language through and save CSV when requested."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "es", "--save_transcript"],
    )

    calls: dict[str, object] = {}
    emotions = [EmotionSegment("happy", 0.0, 1.0)]
    transcript = [TranscriptWord("hola", 0.0, 0.5)]
    timeline = [TimelineEntry(0.0, "happy", "hola")]

    monkeypatch.setattr(emotion_model, "predict_emotions", lambda file: emotions)

    def fake_extract_transcript(
        file_path: str, language: str | None
    ) -> list[TranscriptWord]:
        calls["extract"] = (file_path, language)
        return transcript

    monkeypatch.setattr(
        transcript_module, "extract_transcript", fake_extract_transcript
    )
    monkeypatch.setattr(
        timeline_utils,
        "build_timeline",
        lambda text, emo: timeline if text == transcript and emo == emotions else [],
    )
    monkeypatch.setattr(
        timeline_utils,
        "print_timeline",
        lambda built_timeline: calls.setdefault("printed", built_timeline),
    )

    def fake_save_timeline_to_csv(
        built_timeline: list[TimelineEntry], file_name: str
    ) -> str:
        calls["saved"] = (built_timeline, file_name)
        return "out.csv"

    monkeypatch.setattr(
        timeline_utils, "save_timeline_to_csv", fake_save_timeline_to_csv
    )

    cli.main()

    assert calls["extract"] == ("sample.wav", "es")
    assert calls["printed"] == timeline
    assert calls["saved"] == (timeline, "sample.wav")


def test_cli_prediction_does_not_save_when_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction path should skip CSV export unless `--save_transcript` is set."""
    _patch_common_cli_dependencies(monkeypatch)
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--file", "sample.wav"])
    monkeypatch.setattr(
        emotion_model,
        "predict_emotions",
        lambda _file: [EmotionSegment("calm", 0.0, 1.0)],
    )
    monkeypatch.setattr(
        transcript_module,
        "extract_transcript",
        lambda _file, _language: [TranscriptWord("hello", 0.0, 0.5)],
    )
    monkeypatch.setattr(
        timeline_utils,
        "build_timeline",
        lambda _text, _emo: [TimelineEntry(0.0, "calm", "hello")],
    )
    monkeypatch.setattr(timeline_utils, "print_timeline", lambda _timeline: None)

    save_called = {"value": False}

    def fake_save(_timeline: list[TimelineEntry], _file_name: str) -> str:
        save_called["value"] = True
        return "out.csv"

    monkeypatch.setattr(timeline_utils, "save_timeline_to_csv", fake_save)

    cli.main()

    assert save_called["value"] is False


def test_cli_train_option_uses_runtime_pipeline_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--train` should dispatch through runtime pipeline when flag is enabled."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(cli.sys, "argv", ["ser", "--train"])

    called = {"train": False}

    class FakePipeline:
        def run_training(self) -> None:
            called["train"] = True

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert called["train"] is True


def test_cli_prediction_uses_runtime_pipeline_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction should route through runtime pipeline when flag is enabled."""
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        cli,
        "reload_settings",
        lambda: SimpleNamespace(
            default_language="en",
            runtime_flags=SimpleNamespace(profile_pipeline=True),
        ),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["ser", "--file", "sample.wav", "--language", "pt", "--save_transcript"],
    )

    calls: dict[str, object] = {}

    class FakePipeline:
        def run_training(self) -> None:
            raise AssertionError("Training path should not run for prediction command.")

        def run_inference(self, request: object) -> object:
            calls["request"] = request
            return SimpleNamespace(timeline_csv_path="timeline.csv")

    monkeypatch.setattr(
        cli, "_build_runtime_pipeline", lambda _settings: FakePipeline()
    )

    cli.main()

    request = cast(InferenceRequest, calls["request"])
    assert request.file_path == "sample.wav"
    assert request.language == "pt"
    assert request.save_transcript is True
