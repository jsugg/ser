"""Behavior tests for transcript extraction error handling."""

import logging
import os
import sys
from multiprocessing.connection import Connection
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Never, cast

import pytest

from ser.domain import TranscriptWord
from ser.transcript import transcript_extractor as te
from ser.transcript.backends import faster_whisper as faster_whisper_adapter

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult


@pytest.fixture(autouse=True)
def _disable_openmp_conflict_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps transcript extractor backend tests deterministic."""
    monkeypatch.setattr(
        faster_whisper_adapter,
        "has_known_faster_whisper_openmp_runtime_conflict",
        lambda: False,
    )


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
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file, _profile: (_ for _ in ()).throw(
            te.TranscriptionError("Failed to transcribe audio.")
        ),
    )

    with pytest.raises(te.TranscriptionError, match="Failed to transcribe audio"):
        te._extract_transcript(
            "does-not-matter.wav",
            "en",
            te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        )


def test_extract_transcript_returns_empty_list_for_successful_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call with no words should return an empty transcript."""
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file, _profile: [],
    )

    assert (
        te._extract_transcript(
            "empty.wav",
            "en",
            te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        )
        == []
    )


def test_extract_transcript_formats_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Word-level timestamps should be preserved in formatted output."""
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: object())
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file, _profile: [TranscriptWord("hello", 0.1, 0.3)],
    )

    assert te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
    ) == [TranscriptWord("hello", 0.1, 0.3)]


def test_extract_transcript_releases_runtime_memory_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript extraction should release runtime memory on success."""
    loaded_model = object()
    released_models: list[object] = []
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: loaded_model)
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file, _profile: [],
    )
    monkeypatch.setattr(
        te,
        "_release_transcription_runtime_memory",
        lambda *, model: released_models.append(model),
    )

    result = te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
    )

    assert result == []
    assert released_models == [loaded_model]


def test_extract_transcript_releases_runtime_memory_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript extraction should release runtime memory on failures."""
    loaded_model = object()
    released_models: list[object] = []
    monkeypatch.setattr(te, "load_whisper_model", lambda _profile=None: loaded_model)
    monkeypatch.setattr(
        te,
        "_transcribe_file_with_profile",
        lambda _model, _language, _file, _profile: (_ for _ in ()).throw(
            te.TranscriptionError("Failed to transcribe audio.")
        ),
    )
    monkeypatch.setattr(
        te,
        "_release_transcription_runtime_memory",
        lambda *, model: released_models.append(model),
    )

    with pytest.raises(te.TranscriptionError, match="Failed to transcribe audio"):
        te._extract_transcript(
            "sample.wav",
            "en",
            te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        )

    assert released_models == [loaded_model]


def test_release_transcription_runtime_memory_empties_available_torch_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch cache cleanup should be best-effort and gated by availability checks."""
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
    monkeypatch.setattr(te.gc, "collect", lambda: calls.append("gc"))

    te._release_transcription_runtime_memory(model=object())

    assert calls == ["gc", "mps", "cuda"]


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
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        lambda model: model,
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
    assert os.environ["TORCH_HOME"] == str(torch_cache_root)
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


def test_load_whisper_model_uses_runtime_policy_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stable-whisper should stage MPS loads through CPU per compatibility flow."""
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    settings = SimpleNamespace(
        models=SimpleNamespace(
            whisper_download_root=download_root,
            torch_cache_root=torch_cache_root,
        ),
        torch_runtime=SimpleNamespace(device="auto", dtype="auto"),
    )
    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_load_model(**kwargs: object) -> object:
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(te, "get_settings", lambda: settings)
    monkeypatch.setattr(
        te,
        "resolve_transcription_runtime_policy",
        lambda **_kwargs: SimpleNamespace(
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
            memory_tier="low",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper."
        "enable_stable_whisper_mps_compatibility",
        lambda model: model,
    )
    monkeypatch.delenv("TORCH_HOME", raising=False)

    loaded = te.load_whisper_model(
        profile=te.TranscriptionProfile(
            backend_id="stable_whisper",
            model_name="large-v3",
            use_demucs=False,
            use_vad=True,
        )
    )

    assert loaded is fake_model
    assert captured["device"] == "cpu"
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


def test_extract_transcript_uses_process_isolation_for_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """faster-whisper profiles should route to process-isolated execution path."""
    captured: dict[str, object] = {}
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )

    def _fake_isolated_runner(
        *,
        file_path: str,
        language: str,
        profile: te.TranscriptionProfile,
    ) -> list[TranscriptWord]:
        captured["file_path"] = file_path
        captured["language"] = language
        captured["profile"] = profile
        return expected

    def _fail_in_process(**_kwargs: object) -> list[TranscriptWord]:
        raise AssertionError("in-process path should not be used for faster-whisper")

    monkeypatch.setattr(
        te, "_run_faster_whisper_process_isolated", _fake_isolated_runner
    )
    monkeypatch.setattr(te, "_extract_transcript_in_process", _fail_in_process)

    transcript = te._extract_transcript("sample.wav", "en", profile)

    assert transcript == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile


class _FakeIsolatedParentConnection:
    """Parent pipe endpoint with deterministic message queue."""

    def __init__(self, messages: list[tuple[object, ...]]) -> None:
        self._messages = messages
        self.closed = False

    def recv(self) -> tuple[object, ...]:
        if not self._messages:
            raise EOFError
        return self._messages.pop(0)

    def close(self) -> None:
        self.closed = True


class _FakeIsolatedChildConnection:
    """Child pipe endpoint for process-isolated transcript tests."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeIsolatedProcess:
    """Fake process supporting join-first and terminate-fallback scenarios."""

    def __init__(self, *, exit_on_join: bool) -> None:
        self._alive = False
        self.closed = False
        self.join_timeouts: list[float | None] = []
        self._exit_on_join = exit_on_join

    def start(self) -> None:
        self._alive = True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if self._exit_on_join:
            self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def close(self) -> None:
        self.closed = True


class _FakeIsolatedContext:
    """Fake spawn context for deterministic process-isolated cleanup tests."""

    def __init__(
        self,
        *,
        parent_conn: _FakeIsolatedParentConnection,
        child_conn: _FakeIsolatedChildConnection,
        process: _FakeIsolatedProcess,
    ) -> None:
        self.parent_conn = parent_conn
        self.child_conn = child_conn
        self.process = process

    def Pipe(
        self, duplex: bool = False
    ) -> tuple[_FakeIsolatedParentConnection, _FakeIsolatedChildConnection]:
        assert duplex is False
        return self.parent_conn, self.child_conn

    def Process(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        daemon: bool,
    ) -> _FakeIsolatedProcess:
        del target, args
        assert daemon is False
        return self.process


def _build_fake_isolated_context(
    *,
    messages: list[tuple[object, ...]],
    exit_on_join: bool,
) -> _FakeIsolatedContext:
    """Builds deterministic fake multiprocessing context for isolated runs."""
    return _FakeIsolatedContext(
        parent_conn=_FakeIsolatedParentConnection(messages),
        child_conn=_FakeIsolatedChildConnection(),
        process=_FakeIsolatedProcess(exit_on_join=exit_on_join),
    )


def test_faster_whisper_isolated_run_joins_worker_before_terminate_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful isolated runs should allow worker shutdown before terminate fallback."""
    messages: list[tuple[object, ...]] = [
        ("phase", "setup_complete"),
        ("phase", "model_loaded"),
        ("ok", [("hello", 0.0, 0.5)]),
    ]
    terminate_calls: list[object] = []

    context = _build_fake_isolated_context(messages=messages, exit_on_join=True)
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )
    monkeypatch.setattr(te.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(te, "_terminate_worker_process", terminate_calls.append)

    result = te._run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
    )

    assert result == [TranscriptWord("hello", 0.0, 0.5)]
    assert context.process.join_timeouts == [te._TERMINATE_GRACE_SECONDS]
    assert terminate_calls == []
    assert context.parent_conn.closed is True
    assert context.child_conn.closed is True
    assert context.process.closed is True


def test_faster_whisper_isolated_run_terminates_worker_after_join_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolated runs should still terminate workers that remain alive after join timeout."""
    messages: list[tuple[object, ...]] = [
        ("phase", "setup_complete"),
        ("phase", "model_loaded"),
        ("ok", [("hello", 0.0, 0.5)]),
    ]
    terminate_calls: list[object] = []

    def _fake_terminate_worker_process(process: object) -> None:
        terminate_calls.append(process)
        cast(_FakeIsolatedProcess, process)._alive = False

    context = _build_fake_isolated_context(messages=messages, exit_on_join=False)
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )
    monkeypatch.setattr(te.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(te, "_terminate_worker_process", _fake_terminate_worker_process)

    result = te._run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
    )

    assert result == [TranscriptWord("hello", 0.0, 0.5)]
    assert context.process.join_timeouts == [te._TERMINATE_GRACE_SECONDS]
    assert terminate_calls == [context.process]
    assert context.parent_conn.closed is True
    assert context.child_conn.closed is True
    assert context.process.closed is True


def test_runtime_request_for_isolated_faster_whisper_defaults_to_cpu() -> None:
    """Process-isolated faster runtime request should avoid torch dependency on CPU."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="auto", dtype="auto")),
    )
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )

    runtime_request = te._runtime_request_for_isolated_faster_whisper(
        profile=profile,
        settings=settings,
    )

    assert runtime_request.device_spec == "cpu"
    assert runtime_request.device_type == "cpu"
    assert runtime_request.precision_candidates == ("float32",)


def test_runtime_request_for_isolated_faster_whisper_honors_cuda_request() -> None:
    """Process-isolated faster runtime request should preserve explicit CUDA selectors."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(
            torch_runtime=SimpleNamespace(device="cuda:0", dtype="float16")
        ),
    )
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )

    runtime_request = te._runtime_request_for_isolated_faster_whisper(
        profile=profile,
        settings=settings,
    )

    assert runtime_request.device_spec == "cuda:0"
    assert runtime_request.device_type == "cuda"
    assert runtime_request.precision_candidates == ("float16",)


def test_transcription_worker_entry_blocks_torch_for_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Worker should disable torch import path before faster-whisper adapter operations."""
    observed: dict[str, object] = {}
    messages: list[tuple[object, ...]] = []

    class _FakeConnection:
        def send(self, message: tuple[object, ...]) -> None:
            messages.append(message)

        def close(self) -> None:
            observed["closed"] = True

    class _FakeAdapter:
        def setup_required(self, *, runtime_request: object, settings: object) -> bool:
            del runtime_request, settings
            observed["torch_none_setup"] = sys.modules.get("torch") is None
            return False

        def prepare_assets(self, *, runtime_request: object, settings: object) -> None:
            del runtime_request, settings
            raise AssertionError(
                "prepare_assets should not run when setup is not required"
            )

        def load_model(self, *, runtime_request: object, settings: object) -> object:
            del runtime_request, settings
            observed["torch_none_load"] = sys.modules.get("torch") is None
            return object()

        def transcribe(
            self,
            *,
            model: object,
            runtime_request: object,
            file_path: str,
            language: str,
            settings: object,
        ) -> list[TranscriptWord]:
            del model, runtime_request, settings
            observed["torch_none_transcribe"] = sys.modules.get("torch") is None
            observed["file_path"] = file_path
            observed["language"] = language
            return [TranscriptWord("hello", 0.0, 0.5)]

    monkeypatch.setattr(
        te, "get_settings", lambda: cast(te.AppConfig, SimpleNamespace())
    )
    monkeypatch.setattr(
        te,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: cast(object, _FakeAdapter()),
    )
    payload = te._TranscriptionProcessPayload(
        file_path="sample.wav",
        language="en",
        profile=te.TranscriptionProfile(
            backend_id="faster_whisper",
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
        ),
    )
    original_torch = sys.modules.pop("torch", None)
    try:
        te._transcription_worker_entry(
            payload,
            cast(Connection, _FakeConnection()),
        )
    finally:
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        elif "torch" in sys.modules:
            del sys.modules["torch"]

    assert observed["torch_none_setup"] is True
    assert observed["torch_none_load"] is True
    assert observed["torch_none_transcribe"] is True
    assert observed["file_path"] == "sample.wav"
    assert observed["language"] == "en"
    assert observed["closed"] is True
    assert messages[0] == ("phase", "setup_complete")
    assert messages[1] == ("phase", "model_loaded")
    assert messages[2] == ("ok", [("hello", 0.0, 0.5)])


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
        faster_whisper_adapter.importlib,
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
        faster_whisper_adapter.importlib,
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
