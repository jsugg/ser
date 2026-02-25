"""Behavior tests for transcript extraction error handling."""

import logging
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
        lambda _model, _language, _file: [],
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
        lambda _model, _language, _file: [TranscriptWord("hello", 0.1, 0.3)],
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


def test_load_whisper_model_supports_faster_whisper_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Faster-whisper backend should be loadable through the same facade."""
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = SimpleNamespace(
        models=SimpleNamespace(
            whisper_download_root=download_root,
            torch_cache_root=torch_cache_root,
        )
    )
    captured: dict[str, object] = {}

    class _FakeWhisperModel:
        def __init__(
            self,
            model_size_or_path: str,
            *,
            device: str,
            compute_type: str,
            download_root: str,
        ) -> None:
            captured["model_size_or_path"] = model_size_or_path
            captured["device"] = device
            captured["compute_type"] = compute_type
            captured["download_root"] = download_root

    monkeypatch.setattr(te, "get_settings", lambda: settings)
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=_FakeWhisperModel),
    )

    loaded = te.load_whisper_model(
        profile=te.TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
        )
    )

    assert isinstance(loaded, _FakeWhisperModel)
    assert captured["model_size_or_path"] == "distil-large-v3"
    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "int8"
    assert captured["download_root"] == str(download_root)


def test_transcribe_with_model_supports_faster_whisper_word_segments() -> None:
    """Faster-whisper segment word payloads should map to TranscriptWord rows."""
    words = [
        SimpleNamespace(word="hello", start=0.0, end=0.2),
        SimpleNamespace(word="world", start=0.2, end=0.5),
    ]

    class _FakeFasterModel:
        def transcribe(
            self, *_args: object, **_kwargs: object
        ) -> tuple[object, object]:
            return iter([SimpleNamespace(words=words)]), object()

    transcript = te.transcribe_with_model(
        model=_FakeFasterModel(),
        file_path="sample.wav",
        language="en",
        profile=te.TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
        ),
    )

    assert transcript == [
        TranscriptWord("hello", 0.0, 0.2),
        TranscriptWord("world", 0.2, 0.5),
    ]


def test_faster_whisper_info_logs_are_demoted_to_debug_during_transcription() -> None:
    """faster-whisper INFO entries should be demoted to DEBUG in transcription scope."""
    words = [SimpleNamespace(word="hello", start=0.0, end=0.2)]
    captured: list[logging.LogRecord] = []
    root_logger = logging.getLogger()
    original_level = root_logger.level

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _CaptureHandler(level=logging.DEBUG)
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    try:

        class _FakeFasterModel:
            def transcribe(
                self, *_args: object, **_kwargs: object
            ) -> tuple[object, object]:
                logging.getLogger("faster_whisper").info("Processing audio sample")
                return iter([SimpleNamespace(words=words)]), object()

        transcript = te.transcribe_with_model(
            model=_FakeFasterModel(),
            file_path="sample.wav",
            language="en",
            profile=te.TranscriptionProfile(
                backend_id="faster_whisper",
                model_name="distil-large-v3",
                use_demucs=False,
                use_vad=True,
            ),
        )
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)

    assert transcript == [TranscriptWord("hello", 0.0, 0.2)]
    faster_records = [
        record for record in captured if record.name.startswith("faster_whisper")
    ]
    assert faster_records, "Expected faster_whisper logs to be captured."
    assert all(record.levelno == logging.DEBUG for record in faster_records)


def test_extract_transcript_logs_setup_before_model_load_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transcription setup phase should precede model load when download is needed."""
    phase_events: list[tuple[str, str]] = []
    settings = SimpleNamespace(default_language="en")

    def _fake_phase_started(_logger: object, *, phase_name: str) -> float:
        phase_events.append(("start", phase_name))
        return 1.0

    def _fake_phase_completed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        phase_events.append(("completed", phase_name))
        assert started_at == 1.0

    monkeypatch.setattr(te, "get_settings", lambda: settings)
    monkeypatch.setattr(te, "log_phase_started", _fake_phase_started)
    monkeypatch.setattr(te, "log_phase_completed", _fake_phase_completed)
    monkeypatch.setattr(te, "log_phase_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(te, "_transcription_setup_required", lambda **_k: True)
    monkeypatch.setattr(te, "_prepare_transcription_assets", lambda **_k: None)
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file_path, _profile: [],
    )

    _ = te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
    )

    assert phase_events == [
        ("start", te.PHASE_TRANSCRIPTION_SETUP),
        ("completed", te.PHASE_TRANSCRIPTION_SETUP),
        ("start", te.PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("completed", te.PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("start", te.PHASE_TRANSCRIPTION),
        ("completed", te.PHASE_TRANSCRIPTION),
    ]


def test_extract_transcript_skips_setup_phase_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transcription setup phase should be omitted when assets are already present."""
    phase_events: list[tuple[str, str]] = []
    settings = SimpleNamespace(default_language="en")

    def _fake_phase_started(_logger: object, *, phase_name: str) -> float:
        phase_events.append(("start", phase_name))
        return 1.0

    def _fake_phase_completed(
        _logger: object,
        *,
        phase_name: str,
        started_at: float,
    ) -> None:
        phase_events.append(("completed", phase_name))
        assert started_at == 1.0

    def _fail_prepare(**_kwargs: object) -> None:
        raise AssertionError("setup should not run")

    monkeypatch.setattr(te, "get_settings", lambda: settings)
    monkeypatch.setattr(te, "log_phase_started", _fake_phase_started)
    monkeypatch.setattr(te, "log_phase_completed", _fake_phase_completed)
    monkeypatch.setattr(te, "log_phase_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(te, "_transcription_setup_required", lambda **_k: False)
    monkeypatch.setattr(te, "_prepare_transcription_assets", _fail_prepare)
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file_path, _profile: [],
    )

    _ = te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
    )

    assert phase_events == [
        ("start", te.PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("completed", te.PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("start", te.PHASE_TRANSCRIPTION),
        ("completed", te.PHASE_TRANSCRIPTION),
    ]


def test_faster_whisper_setup_required_when_cache_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """faster-whisper setup should run when local cache lookup misses."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(whisper_download_root=tmp_path / "model-cache")
        ),
    )
    captured: dict[str, object] = {}

    def _fake_download_model(
        model_name: str,
        *,
        local_files_only: bool,
        cache_dir: str,
    ) -> str:
        captured["model_name"] = model_name
        captured["local_files_only"] = local_files_only
        captured["cache_dir"] = cache_dir
        raise RuntimeError("cache miss")

    monkeypatch.setattr(
        te.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(download_model=_fake_download_model)
            if name == "faster_whisper.utils"
            else __import__(name)
        ),
    )

    required = te._transcription_setup_required(
        active_profile=te.TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="distil-large-v3",
        ),
        settings=settings,
    )

    assert required is True
    assert captured["model_name"] == "distil-large-v3"
    assert captured["local_files_only"] is True
    assert captured["cache_dir"] == str(settings.models.whisper_download_root)


def test_faster_whisper_prepare_transcription_assets_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """faster-whisper setup should trigger download in non-local-files mode."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(whisper_download_root=tmp_path / "model-cache")
        ),
    )
    captured: dict[str, object] = {}

    def _fake_download_model(
        model_name: str,
        *,
        local_files_only: bool,
        cache_dir: str,
    ) -> str:
        captured["model_name"] = model_name
        captured["local_files_only"] = local_files_only
        captured["cache_dir"] = cache_dir
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(
        te.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(download_model=_fake_download_model)
            if name == "faster_whisper.utils"
            else __import__(name)
        ),
    )

    te._prepare_transcription_assets(
        active_profile=te.TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="distil-large-v3",
        ),
        settings=settings,
    )

    assert captured["model_name"] == "distil-large-v3"
    assert captured["local_files_only"] is False
    assert captured["cache_dir"] == str(settings.models.whisper_download_root)
