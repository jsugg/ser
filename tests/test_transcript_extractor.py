"""Behavior tests for transcript extraction error handling."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Never, cast

import pytest

from ser.domain import TranscriptWord
from ser.transcript import transcript_extractor as te

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult


class FailingModel:
    """Fake model that always fails during transcription."""

    def transcribe(self, **_kwargs: object) -> Never:
        raise RuntimeError("transcribe failure")


class FakeResult:
    """Whisper-like result object with configurable word payload."""

    def __init__(self, words: list[SimpleNamespace]) -> None:
        self._words = words

    def all_words(self) -> list[SimpleNamespace]:
        return self._words


def test_extract_transcript_raises_transcription_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational failures should propagate as TranscriptionError."""
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: FailingModel())

    with pytest.raises(te.TranscriptionError, match="Failed to transcribe audio"):
        te.extract_transcript("does-not-matter.wav", "en")


def test_extract_transcript_returns_empty_list_for_successful_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call with no words should return an empty transcript."""
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "__transcribe_file",
        lambda _model, _language, _file: FakeResult([]),
    )

    assert te.extract_transcript("empty.wav", "en") == []


def test_extract_transcript_formats_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Word-level timestamps should be preserved in formatted output."""
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "__transcribe_file",
        lambda _model, _language, _file: FakeResult(
            [SimpleNamespace(word="hello", start=0.1, end=0.3)]
        ),
    )

    assert te.extract_transcript("sample.wav", "en") == [
        TranscriptWord("hello", 0.1, 0.3)
    ]


def test_format_transcript_raises_for_invalid_result() -> None:
    """Invalid result objects should raise a domain-level error."""
    with pytest.raises(te.TranscriptionError, match="Invalid Whisper result object"):
        invalid_result = cast("WhisperResult", object())
        te.format_transcript(invalid_result)


def test_load_whisper_model_routes_downloads_to_model_cache_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Whisper and torch-hub assets should route to SER model cache roots."""
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = SimpleNamespace(
        models=SimpleNamespace(
            whisper_download_root=download_root,
            torch_cache_root=torch_cache_root,
        )
    )
    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_load_model(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(te, "get_settings", lambda: settings)
    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.delenv("TORCH_HOME", raising=False)

    loaded = te.load_whisper_model(
        profile=te.TranscriptionProfile(
            model_name="tiny",
            use_demucs=False,
            use_vad=False,
        )
    )

    assert loaded is fake_model
    assert captured["download_root"] == str(download_root)
    assert te.os.environ["TORCH_HOME"] == str(torch_cache_root)
    assert download_root.is_dir()
    assert torch_cache_root.is_dir()
