"""Behavior tests for transcript extraction error handling."""

import logging
import os
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Never, cast

import pytest

from ser.domain import TranscriptWord
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
    PHASE_TRANSCRIPTION_SETUP,
)
from ser.transcript import transcript_extractor as te
from ser.transcript.backends import faster_whisper as faster_whisper_adapter
from ser.transcript.backends.base import CompatibilityIssue

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
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        te._boundary_support, "transcription_setup_required", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            te.TranscriptionError("Failed to transcribe audio.")
        ),
    )

    with pytest.raises(te.TranscriptionError, match="Failed to transcribe audio"):
        te._extract_transcript(
            "does-not-matter.wav",
            "en",
            te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
            settings=settings,
        )


def test_extract_transcript_returns_empty_list_for_successful_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful call with no words should return an empty transcript."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        te._boundary_support, "transcription_setup_required", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: [],
    )

    assert (
        te._extract_transcript(
            "empty.wav",
            "en",
            te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
            settings=settings,
        )
        == []
    )


def test_extract_transcript_formats_word_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Word-level timestamps should be preserved in formatted output."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        te._boundary_support, "transcription_setup_required", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: [TranscriptWord("hello", 0.1, 0.3)],
    )

    assert te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        settings=settings,
    ) == [TranscriptWord("hello", 0.1, 0.3)]


def test_extract_transcript_releases_runtime_memory_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript extraction should release runtime memory on success."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    loaded_model = object()
    released_models: list[object] = []
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: loaded_model,
    )
    monkeypatch.setattr(
        te._boundary_support, "transcription_setup_required", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: [],
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
        settings=settings,
    )

    assert result == []
    assert released_models == [loaded_model]


def test_extract_transcript_releases_runtime_memory_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript extraction should release runtime memory on failures."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    loaded_model = object()
    released_models: list[object] = []
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: loaded_model,
    )
    monkeypatch.setattr(
        te._boundary_support, "transcription_setup_required", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
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
            settings=settings,
        )

    assert released_models == [loaded_model]


def test_release_transcription_runtime_memory_empties_available_torch_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch cache cleanup should be best-effort and gated by availability checks."""
    from ser._internal.transcription import process_worker as process_worker_helpers

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
        process_worker_helpers.gc,
        "collect",
        lambda: calls.append("gc"),
    )

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
    huggingface_cache_root = tmp_path / "model-cache" / "huggingface"
    modelscope_cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    settings = SimpleNamespace(
        models=SimpleNamespace(
            whisper_download_root=download_root,
            torch_cache_root=torch_cache_root,
            huggingface_cache_root=huggingface_cache_root,
            modelscope_cache_root=modelscope_cache_root,
        ),
        torch_runtime=SimpleNamespace(enable_mps_fallback=False),
    )
    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_load_model(**kwargs: object) -> object:
        captured["torch_home"] = os.getenv("TORCH_HOME")
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(te, "reload_settings", lambda: settings)
    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper." "enable_stable_whisper_mps_compatibility",
        lambda model: model,
    )
    monkeypatch.delenv("TORCH_HOME", raising=False)

    loaded = te.load_whisper_model(
        profile=te.TranscriptionProfile(
            backend_id="stable_whisper",
            model_name="tiny",
            use_demucs=False,
            use_vad=False,
        )
    )

    assert loaded is fake_model
    assert captured["download_root"] == str(download_root)
    assert captured["torch_home"] == str(torch_cache_root)
    assert "TORCH_HOME" not in os.environ
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

    monkeypatch.setattr(te, "reload_settings", lambda: settings)
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
    huggingface_cache_root = tmp_path / "model-cache" / "huggingface"
    modelscope_cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    settings = SimpleNamespace(
        models=SimpleNamespace(
            whisper_download_root=download_root,
            torch_cache_root=torch_cache_root,
            huggingface_cache_root=huggingface_cache_root,
            modelscope_cache_root=modelscope_cache_root,
        ),
        torch_runtime=SimpleNamespace(
            device="auto",
            dtype="auto",
            enable_mps_fallback=False,
        ),
    )
    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_load_model(**kwargs: object) -> object:
        captured["torch_home"] = os.getenv("TORCH_HOME")
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(te, "reload_settings", lambda: settings)
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
        "ser.transcript.backends.stable_whisper." "enable_stable_whisper_mps_compatibility",
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


def test_load_whisper_model_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit settings should fully drive model loading without ambient fallbacks."""
    download_root = tmp_path / "model-cache" / "OpenAI" / "whisper"
    torch_cache_root = tmp_path / "model-cache" / "torch"
    huggingface_cache_root = tmp_path / "model-cache" / "huggingface"
    modelscope_cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    settings = cast(
        te.AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(
                whisper_download_root=download_root,
                torch_cache_root=torch_cache_root,
                huggingface_cache_root=huggingface_cache_root,
                modelscope_cache_root=modelscope_cache_root,
                whisper_model=SimpleNamespace(name="large-v3"),
            ),
            transcription=SimpleNamespace(
                backend_id="stable_whisper",
                use_demucs=False,
                use_vad=True,
            ),
            torch_runtime=SimpleNamespace(
                device="auto",
                dtype="auto",
                enable_mps_fallback=False,
            ),
        ),
    )
    captured: dict[str, object] = {}
    fake_model = object()

    def _fake_load_model(**kwargs: object) -> object:
        captured["torch_home"] = os.getenv("TORCH_HOME")
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(
            AssertionError("explicit settings must bypass ambient resolution")
        ),
    )
    monkeypatch.setattr(
        te,
        "resolve_transcription_runtime_policy",
        lambda **_kwargs: SimpleNamespace(
            device_spec="cpu",
            device_type="cpu",
            precision_candidates=("float32",),
            memory_tier="low",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_whisper",
        SimpleNamespace(load_model=_fake_load_model),
    )
    monkeypatch.setattr(
        "ser.transcript.backends.stable_whisper." "enable_stable_whisper_mps_compatibility",
        lambda model: model,
    )
    monkeypatch.delenv("TORCH_HOME", raising=False)

    loaded = te.load_whisper_model(settings=settings)

    assert loaded is fake_model
    assert captured["name"] == "large-v3"
    assert captured["download_root"] == str(download_root)
    assert captured["torch_home"] == str(torch_cache_root)
    assert "TORCH_HOME" not in os.environ


def test_transcribe_with_model_supports_faster_whisper_word_segments() -> None:
    """Faster-whisper segment word payloads should map to TranscriptWord rows."""
    words = [
        SimpleNamespace(word="hello", start=0.0, end=0.2),
        SimpleNamespace(word="world", start=0.2, end=0.5),
    ]

    class _FakeFasterModel:
        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
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
            def transcribe(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
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
    faster_records = [record for record in captured if record.name.startswith("faster_whisper")]
    assert faster_records, "Expected faster_whisper logs to be captured."
    assert all(record.levelno == logging.DEBUG for record in faster_records)


def test_check_adapter_compatibility_logs_non_blocking_issues_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Repeated compatibility checks should emit non-blocking issues once."""
    compatibility_report = te.CompatibilityReport(
        backend_id="stable_whisper",
        operational_issues=(
            CompatibilityIssue(
                code="torio_ffmpeg_abi_mismatch",
                message="torchaudio FFmpeg extension ABI mismatch",
            ),
        ),
        noise_issues=(
            CompatibilityIssue(
                code="stable_whisper_invalid_escape_sequence",
                message="stable-whisper import warning noise",
            ),
        ),
    )

    class _FakeAdapter:
        def check_compatibility(
            self,
            *,
            runtime_request: te.BackendRuntimeRequest,
            settings: object,
        ) -> te.CompatibilityReport:
            del runtime_request
            del settings
            return compatibility_report

    monkeypatch.setattr(
        te,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: _FakeAdapter(),
    )
    monkeypatch.setattr(te, "_EMITTED_COMPATIBILITY_ISSUE_KEYS", set())
    runtime_request = te.BackendRuntimeRequest(
        model_name="large-v2",
        use_demucs=False,
        use_vad=True,
    )
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    with caplog.at_level(logging.DEBUG):
        _ = te._check_adapter_compatibility(
            active_profile=profile,
            settings=settings,
            runtime_request=runtime_request,
        )
        _ = te._check_adapter_compatibility(
            active_profile=profile,
            settings=settings,
            runtime_request=runtime_request,
        )

    noise_records = [
        record
        for record in caplog.records
        if "noise issue [stable_whisper_invalid_escape_sequence]" in record.getMessage()
    ]
    operational_records = [
        record
        for record in caplog.records
        if "operational issue [torio_ffmpeg_abi_mismatch]" in record.getMessage()
    ]
    assert len(noise_records) == 1
    assert len(operational_records) == 1


def test_check_adapter_compatibility_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor compatibility wrapper should delegate with injected dependencies."""
    captured: dict[str, object] = {}
    emitted_issue_keys: set[tuple[str, str, str]] = set()
    report = te.CompatibilityReport(backend_id="stable_whisper")

    def _fake_impl(**kwargs: object) -> te.CompatibilityReport:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(te, "_EMITTED_COMPATIBILITY_ISSUE_KEYS", emitted_issue_keys)
    monkeypatch.setattr(te._boundary_support, "_check_adapter_compatibility_impl", _fake_impl)
    runtime_request = te.BackendRuntimeRequest(
        model_name="large-v2",
        use_demucs=False,
        use_vad=True,
    )
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    resolved = te._check_adapter_compatibility(
        active_profile=profile,
        settings=settings,
        runtime_request=runtime_request,
    )

    assert resolved is report
    assert captured["active_profile"] == profile
    assert captured["settings"] is settings
    assert captured["runtime_request"] == runtime_request
    assert (
        captured["runtime_request_resolver"] is te._boundary_support._runtime_request_from_profile
    )
    assert (
        captured["adapter_resolver"] is te._boundary_support.resolve_transcription_backend_adapter
    )
    assert captured["error_factory"] is te.TranscriptionError
    assert captured["emitted_issue_keys"] is emitted_issue_keys
    assert captured["logger"] is te.logger


def test_check_adapter_compatibility_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility wrapper should delegate boundary assembly to the internal owner."""
    captured: dict[str, object] = {}
    report = te.CompatibilityReport(backend_id="stable_whisper")
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())
    emitted_issue_keys: set[tuple[str, str, str]] = set()

    def _fake_boundary_impl(**kwargs: object) -> te.CompatibilityReport:
        captured.update(kwargs)
        return report

    monkeypatch.setattr(te, "_EMITTED_COMPATIBILITY_ISSUE_KEYS", emitted_issue_keys)
    monkeypatch.setattr(
        te._boundary_support,
        "_check_adapter_compatibility_boundary_impl",
        _fake_boundary_impl,
    )

    resolved = te._check_adapter_compatibility(
        active_profile=profile,
        settings=settings,
    )

    assert resolved is report
    assert captured["active_profile"] == profile
    assert captured["settings"] is settings
    assert captured["runtime_request"] is None
    assert (
        captured["check_adapter_compatibility_impl"]
        is te._boundary_support._check_adapter_compatibility_impl
    )
    assert (
        captured["runtime_request_resolver"] is te._boundary_support._runtime_request_from_profile
    )
    assert (
        captured["adapter_resolver"] is te._boundary_support.resolve_transcription_backend_adapter
    )
    assert captured["emitted_issue_keys"] is emitted_issue_keys
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError


def test_runtime_request_from_profile_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime-request wrapper should delegate with policy/default injections."""
    captured: dict[str, object] = {}
    expected = te.BackendRuntimeRequest(
        model_name="large-v2",
        use_demucs=True,
        use_vad=True,
        device_spec="cpu",
        device_type="cpu",
        precision_candidates=("float32",),
        memory_tier="unknown",
    )

    def _fake_impl(**kwargs: object) -> te.BackendRuntimeRequest:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(te, "_runtime_request_from_profile_impl", _fake_impl)
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    resolved = te._runtime_request_from_profile(profile, settings)

    assert resolved is expected
    assert captured["active_profile"] == profile
    assert captured["settings"] is settings
    assert captured["runtime_policy_resolver"] is te.resolve_transcription_runtime_policy
    assert captured["default_mps_low_memory_threshold_gb"] == te.DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB


def test_run_faster_whisper_process_isolated_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-isolated wrapper should delegate with injected dependencies."""
    captured: dict[str, object] = {}
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    settings = cast(
        te.AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto")),
    )

    def _fake_impl(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        te._boundary_support,
        "_run_faster_whisper_process_isolated_impl",
        _fake_impl,
    )
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )

    resolved = te._run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings=settings,
    )

    assert resolved == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    settings_resolver = cast(Callable[[], te.AppConfig], captured["settings_resolver"])
    assert settings_resolver() is settings
    runtime_request_resolver = cast(
        Callable[[te.TranscriptionProfile, te.AppConfig], te.BackendRuntimeRequest],
        captured["runtime_request_resolver"],
    )
    runtime_request = runtime_request_resolver(
        profile,
        cast(
            te.AppConfig,
            SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto")),
        ),
    )
    assert runtime_request.device_type == "cpu"
    assert runtime_request.precision_candidates == ("float32",)
    assert captured["payload_factory"] is te._boundary_support._build_transcription_process_payload
    payload_factory = cast(Callable[..., object], captured["payload_factory"])
    payload = cast(
        te._TranscriptionProcessPayload,
        payload_factory(
            file_path="sample.wav",
            language="en",
            profile=profile,
            runtime_request=te.BackendRuntimeRequest(
                model_name="distil-large-v3",
                use_demucs=False,
                use_vad=True,
                device_spec="cpu",
                device_type="cpu",
                precision_candidates=("float32",),
                memory_tier="not_applicable",
            ),
            settings=cast(
                te.AppConfig,
                SimpleNamespace(
                    models=SimpleNamespace(whisper_download_root=Path("/tmp/whisper-cache"))
                ),
            ),
        ),
    )
    assert payload.settings.models.whisper_download_root == Path("/tmp/whisper-cache")
    assert captured["get_spawn_context"] is te._spawn_context
    get_spawn_context = cast(Callable[[], object], captured["get_spawn_context"])
    spawn_context = get_spawn_context()
    assert hasattr(spawn_context, "Pipe")
    assert hasattr(spawn_context, "Process")
    worker_entry = cast(Callable[[object, object], None], captured["worker_entry"])
    assert worker_entry is te._transcription_worker_entry
    assert ForkingPickler.dumps(worker_entry)
    terminate_worker_process = cast(
        Callable[[object], None],
        captured["terminate_worker_process_fn"],
    )
    assert terminate_worker_process is te._terminate_worker_process
    assert callable(captured["recv_worker_message_fn"])
    assert captured["raise_worker_error_fn"] is te._raise_worker_error
    assert captured["transcript_word_factory"] is te.TranscriptWord
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError
    assert captured["terminate_grace_seconds"] == te._TERMINATE_GRACE_SECONDS


def test_run_faster_whisper_process_isolated_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public isolated-run wrapper should delegate assembly to the internal boundary owner."""
    captured: dict[str, object] = {}
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    def _fake_boundary_impl(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        te._boundary_support,
        "_run_faster_whisper_process_isolated_boundary_impl",
        _fake_boundary_impl,
    )

    resolved = te._run_faster_whisper_process_isolated(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings=settings,
    )

    assert resolved == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    assert captured["settings"] is settings
    assert (
        captured["run_faster_whisper_process_isolated_impl"]
        is te._boundary_support._run_faster_whisper_process_isolated_impl
    )
    assert callable(captured["runtime_request_resolver"])
    assert captured["payload_factory"] is te._boundary_support._build_transcription_process_payload
    assert captured["spawn_context_resolver"] is te._spawn_context
    assert captured["worker_entry"] is te._transcription_worker_entry
    assert callable(captured["recv_worker_message_fn"])
    assert captured["raise_worker_error_fn"] is te._raise_worker_error
    assert captured["terminate_worker_process_fn"] is te._terminate_worker_process
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError
    assert captured["terminate_grace_seconds"] == te._TERMINATE_GRACE_SECONDS


def test_extract_transcript_in_process_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript wrapper should delegate with injected dependencies."""
    captured: dict[str, object] = {}
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    settings = cast(te.AppConfig, SimpleNamespace())

    def _fake_impl(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(te._boundary_support, "_extract_transcript_in_process_impl", _fake_impl)
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
        use_demucs=True,
        use_vad=True,
    )

    resolved = te._extract_transcript_in_process(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings=settings,
    )

    assert resolved == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    settings_resolver = cast(Callable[[], te.AppConfig], captured["settings_resolver"])
    assert settings_resolver() is settings
    assert callable(captured["setup_required_checker"])
    assert callable(captured["prepare_assets_runner"])
    assert callable(captured["load_model_fn"])
    assert callable(captured["transcribe_with_profile_fn"])
    assert captured["release_memory_fn"] is te._release_transcription_runtime_memory
    assert captured["logger"] is te.logger


def test_extract_transcript_in_process_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process transcript wrapper should delegate boundary assembly to the internal owner."""
    captured: dict[str, object] = {}
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    settings = cast(te.AppConfig, SimpleNamespace())
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
        use_demucs=True,
        use_vad=True,
    )

    def _fake_boundary_impl(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        te._boundary_support,
        "_extract_transcript_in_process_boundary_impl",
        _fake_boundary_impl,
    )

    resolved = te._extract_transcript_in_process(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings=settings,
    )

    assert resolved == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    assert captured["settings"] is settings
    assert (
        captured["extract_transcript_in_process_impl"]
        is te._boundary_support._extract_transcript_in_process_impl
    )
    assert callable(captured["setup_required_checker"])
    assert callable(captured["prepare_assets_runner"])
    assert callable(captured["load_whisper_model_fn"])
    assert callable(captured["transcribe_with_profile_fn"])
    assert captured["release_memory_fn"] is te._release_transcription_runtime_memory
    assert captured["phase_started_fn"] is te.log_phase_started
    assert captured["phase_completed_fn"] is te.log_phase_completed
    assert captured["phase_failed_fn"] is te.log_phase_failed
    assert captured["logger"] is te.logger


def test_transcription_setup_required_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setup-required wrapper should delegate boundary assembly to the runtime owner."""
    captured: dict[str, object] = {}
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    def _fake_boundary_impl(**kwargs: object) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        te._boundary_support,
        "_transcription_setup_required_boundary_impl",
        _fake_boundary_impl,
    )

    required = te._transcription_setup_required(
        active_profile=profile,
        settings=settings,
    )

    assert required is True
    assert captured["active_profile"] == profile
    assert captured["settings"] is settings
    assert (
        captured["transcription_setup_required_impl"]
        is te._boundary_support._transcription_setup_required_impl
    )
    assert (
        captured["runtime_request_resolver"] is te._boundary_support._runtime_request_from_profile
    )
    assert callable(captured["compatibility_checker"])
    assert (
        captured["adapter_resolver"] is te._boundary_support.resolve_transcription_backend_adapter
    )


def test_prepare_transcription_assets_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asset-preparation wrapper should delegate boundary assembly to the runtime owner."""
    captured: dict[str, object] = {}
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    settings = cast(te.AppConfig, SimpleNamespace())

    def _fake_boundary_impl(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        te._boundary_support,
        "_prepare_transcription_assets_boundary_impl",
        _fake_boundary_impl,
    )

    te._prepare_transcription_assets(
        active_profile=profile,
        settings=settings,
    )

    assert captured["active_profile"] == profile
    assert captured["settings"] is settings
    assert (
        captured["prepare_transcription_assets_impl"]
        is te._boundary_support._prepare_transcription_assets_impl
    )
    assert (
        captured["runtime_request_resolver"] is te._boundary_support._runtime_request_from_profile
    )
    assert callable(captured["compatibility_checker"])
    assert (
        captured["adapter_resolver"] is te._boundary_support.resolve_transcription_backend_adapter
    )


def test_mark_compatibility_issues_as_emitted_suppresses_duplicate_operational_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pre-emitted compatibility issues should not be logged again."""
    monkeypatch.setattr(te, "_EMITTED_COMPATIBILITY_ISSUE_KEYS", set())
    te.mark_compatibility_issues_as_emitted(
        backend_id="stable_whisper",
        issue_kind="operational",
        issue_codes=("torio_ffmpeg_abi_mismatch",),
    )

    class _Adapter:
        def check_compatibility(
            self,
            *,
            runtime_request: te.BackendRuntimeRequest,
            settings: te.AppConfig,
        ) -> te.CompatibilityReport:
            del runtime_request, settings
            return te.CompatibilityReport(
                backend_id="stable_whisper",
                operational_issues=(
                    CompatibilityIssue(
                        code="torio_ffmpeg_abi_mismatch",
                        message="torchaudio FFmpeg extension ABI mismatch",
                        impact="degraded",
                    ),
                ),
            )

    monkeypatch.setattr(
        te,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: cast(object, _Adapter()),
    )
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    runtime_request = te.BackendRuntimeRequest(
        model_name="large-v2",
        use_demucs=False,
        use_vad=True,
    )

    with caplog.at_level(logging.WARNING, logger=te.logger.name):
        te._check_adapter_compatibility(
            active_profile=profile,
            settings=cast(te.AppConfig, SimpleNamespace()),
            runtime_request=runtime_request,
        )

    records = [
        record
        for record in caplog.records
        if "operational issue [torio_ffmpeg_abi_mismatch]" in record.getMessage()
    ]
    assert records == []


def test_extract_transcript_logs_setup_before_model_load_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transcription setup phase should precede model load when download is needed."""
    phase_events: list[tuple[str, str]] = []
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))

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

    monkeypatch.setattr(te, "log_phase_started", _fake_phase_started)
    monkeypatch.setattr(te, "log_phase_completed", _fake_phase_completed)
    monkeypatch.setattr(te, "log_phase_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(te._boundary_support, "transcription_setup_required", lambda **_k: True)
    monkeypatch.setattr(te._boundary_support, "prepare_transcription_assets", lambda **_k: None)
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: [],
    )

    _ = te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        settings=settings,
    )

    assert phase_events == [
        ("start", PHASE_TRANSCRIPTION_SETUP),
        ("completed", PHASE_TRANSCRIPTION_SETUP),
        ("start", PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("completed", PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("start", PHASE_TRANSCRIPTION),
        ("completed", PHASE_TRANSCRIPTION),
    ]


def test_extract_transcript_skips_setup_phase_when_not_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transcription setup phase should be omitted when assets are already present."""
    phase_events: list[tuple[str, str]] = []
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))

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

    monkeypatch.setattr(te, "log_phase_started", _fake_phase_started)
    monkeypatch.setattr(te, "log_phase_completed", _fake_phase_completed)
    monkeypatch.setattr(te, "log_phase_failed", lambda *_a, **_k: None)
    monkeypatch.setattr(te._boundary_support, "transcription_setup_required", lambda **_k: False)
    monkeypatch.setattr(te._boundary_support, "prepare_transcription_assets", _fail_prepare)
    monkeypatch.setattr(
        te._boundary_support,
        "load_whisper_model_for_settings",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        te._boundary_support,
        "transcribe_with_profile",
        lambda *_args, **_kwargs: [],
    )

    _ = te._extract_transcript(
        "sample.wav",
        "en",
        te.TranscriptionProfile(backend_id="stable_whisper", model_name="large-v2"),
        settings=settings,
    )

    assert phase_events == [
        ("start", PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("completed", PHASE_TRANSCRIPTION_MODEL_LOAD),
        ("start", PHASE_TRANSCRIPTION),
        ("completed", PHASE_TRANSCRIPTION),
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

    def _fake_isolated_runner(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    def _fail_in_process(**_kwargs: object) -> list[TranscriptWord]:
        raise AssertionError("in-process path should not be used for faster-whisper")

    monkeypatch.setattr(
        te._boundary_support, "run_faster_whisper_process_isolated", _fake_isolated_runner
    )
    monkeypatch.setattr(te._boundary_support, "extract_transcript_in_process", _fail_in_process)
    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(AssertionError("private helper must use explicit settings")),
    )

    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    transcript = te._extract_transcript("sample.wav", "en", profile, settings=settings)

    assert transcript == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    assert captured["settings"] is settings


def test_extract_transcript_routes_non_faster_profiles_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non faster-whisper profiles should route to in-process execution."""
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
        use_demucs=True,
        use_vad=True,
    )
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    captured: dict[str, object] = {}

    def _fail_isolated_runner(**_kwargs: object) -> list[TranscriptWord]:
        raise AssertionError("process-isolated path should not be used")

    def _fake_in_process_runner(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        te._boundary_support, "run_faster_whisper_process_isolated", _fail_isolated_runner
    )
    monkeypatch.setattr(
        te._boundary_support, "extract_transcript_in_process", _fake_in_process_runner
    )
    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(AssertionError("private helper must use explicit settings")),
    )

    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    transcript = te._extract_transcript("sample.wav", "en", profile, settings=settings)

    assert transcript == expected
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] == profile
    assert captured["settings"] is settings


def test_transcribe_with_profile_uses_explicit_settings_without_ambient_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private transcription helper should honor caller-provided settings only."""
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
        use_demucs=False,
        use_vad=True,
    )
    runtime_request = cast(object, SimpleNamespace())
    captured: dict[str, object] = {}

    def _fake_transcribe(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return [TranscriptWord("hello", 0.0, 0.5)]

    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(AssertionError("private helper must use explicit settings")),
    )
    monkeypatch.setattr(
        te._boundary_support, "_runtime_request_from_profile", lambda *_a: runtime_request
    )
    monkeypatch.setattr(te._boundary_support, "check_adapter_compatibility", lambda **_kwargs: None)
    monkeypatch.setattr(
        te._boundary_support,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: SimpleNamespace(transcribe=_fake_transcribe),
    )

    transcript = te._transcribe_file_with_profile(
        object(),
        "en",
        "sample.wav",
        profile,
        settings=settings,
    )

    assert transcript == [TranscriptWord("hello", 0.0, 0.5)]
    assert captured["runtime_request"] is runtime_request
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["settings"] is settings


def test_transcribe_with_profile_resolves_defaults_from_explicit_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default transcription profile should resolve from explicit settings only."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(
            default_language="en",
            models=SimpleNamespace(whisper_model=SimpleNamespace(name="large-v3")),
            transcription=SimpleNamespace(
                backend_id="stable_whisper",
                use_demucs=False,
                use_vad=True,
            ),
            torch_runtime=SimpleNamespace(device="cpu", dtype="auto"),
        ),
    )
    runtime_request = cast(object, SimpleNamespace())
    captured: dict[str, object] = {}

    def _fake_runtime_request(
        active_profile: te.TranscriptionProfile,
        active_settings: te.AppConfig,
    ) -> object:
        captured["active_profile"] = active_profile
        captured["runtime_settings"] = active_settings
        return runtime_request

    def _fake_transcribe(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        return [TranscriptWord("hello", 0.0, 0.5)]

    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(AssertionError("private helper must use explicit settings")),
    )
    monkeypatch.setattr(
        te._boundary_support, "_runtime_request_from_profile", _fake_runtime_request
    )
    monkeypatch.setattr(te._boundary_support, "check_adapter_compatibility", lambda **_kwargs: None)
    monkeypatch.setattr(
        te._boundary_support,
        "resolve_transcription_backend_adapter",
        lambda _backend_id: SimpleNamespace(transcribe=_fake_transcribe),
    )

    transcript = te._transcribe_file_with_profile(
        object(),
        "en",
        "sample.wav",
        None,
        settings=settings,
    )

    assert transcript == [TranscriptWord("hello", 0.0, 0.5)]
    active_profile = cast(te.TranscriptionProfile, captured["active_profile"])
    assert active_profile.backend_id == "stable_whisper"
    assert active_profile.model_name == "large-v3"
    assert active_profile.use_demucs is False
    assert active_profile.use_vad is True
    assert captured["runtime_settings"] is settings
    assert captured["runtime_request"] is runtime_request
    assert captured["settings"] is settings


def test_transcribe_with_profile_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private transcription helper should delegate boundary assembly to the internal owner."""
    captured: dict[str, object] = {}
    settings = cast(te.AppConfig, SimpleNamespace(default_language="en"))
    expected = [TranscriptWord("hello", 0.0, 0.5)]
    model = object()
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
        use_demucs=False,
        use_vad=True,
    )

    def _fake_boundary_impl(*args: object, **kwargs: object) -> list[TranscriptWord]:
        captured["args"] = args
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        te._boundary_support,
        "_transcribe_with_profile_boundary_impl",
        _fake_boundary_impl,
    )

    transcript = te._transcribe_file_with_profile(
        model,
        "en",
        "sample.wav",
        profile,
        settings=settings,
    )

    assert transcript == expected
    assert captured["args"] == (model, "en", "sample.wav", profile)
    assert captured["settings"] is settings
    assert (
        captured["transcribe_with_profile_entrypoint"]
        is te._boundary_support._transcribe_with_profile_entrypoint
    )
    assert callable(captured["resolve_profile_for_settings"])
    assert (
        captured["runtime_request_resolver"] is te._boundary_support._runtime_request_from_profile
    )
    assert callable(captured["compatibility_checker"])
    assert (
        captured["adapter_resolver"] is te._boundary_support.resolve_transcription_backend_adapter
    )
    assert captured["passthrough_error_cls"] is te.TranscriptionError
    assert captured["logger"] is te.logger
    assert captured["error_factory"] is te.TranscriptionError


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
        settings=cast(
            te.AppConfig,
            SimpleNamespace(
                torch_runtime=SimpleNamespace(device="cpu", dtype="auto"),
                models=SimpleNamespace(whisper_download_root=Path("/tmp/whisper-cache")),
            ),
        ),
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
        settings=cast(
            te.AppConfig,
            SimpleNamespace(
                torch_runtime=SimpleNamespace(device="cpu", dtype="auto"),
                models=SimpleNamespace(whisper_download_root=Path("/tmp/whisper-cache")),
            ),
        ),
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
        SimpleNamespace(torch_runtime=SimpleNamespace(device="cuda:0", dtype="float16")),
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


def test_runtime_request_for_isolated_faster_whisper_logs_non_cuda_fallback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unsupported isolated-device selectors should fall back to CPU with one info log."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="mps", dtype="float16")),
    )
    profile = te.TranscriptionProfile(
        backend_id="faster_whisper",
        model_name="distil-large-v3",
        use_demucs=False,
        use_vad=True,
    )
    caplog.set_level(logging.INFO, logger=te.logger.name)

    runtime_request = te._runtime_request_for_isolated_faster_whisper(
        profile=profile,
        settings=settings,
    )

    assert runtime_request.device_spec == "cpu"
    assert runtime_request.device_type == "cpu"
    assert runtime_request.precision_candidates == ("float32",)
    assert any(
        "requested device 'mps' is unsupported; using cpu/float32" in record.getMessage()
        for record in caplog.records
    )


def test_runtime_request_for_isolated_faster_whisper_rejects_non_faster_backend() -> None:
    """Isolated runtime request helper should fail fast for non-faster backends."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto")),
    )
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )

    with pytest.raises(
        te.TranscriptionError,
        match="only supports faster-whisper backend",
    ):
        te._runtime_request_for_isolated_faster_whisper(
            profile=profile,
            settings=settings,
        )


def test_run_faster_whisper_process_isolated_rejects_non_faster_backend_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-isolated entrypoint should reject unsupported backend before spawning."""
    profile = te.TranscriptionProfile(
        backend_id="stable_whisper",
        model_name="large-v2",
    )
    monkeypatch.setattr(
        te.mp,
        "get_context",
        lambda _method: (_ for _ in ()).throw(AssertionError("must not spawn")),
    )

    with pytest.raises(
        te.TranscriptionError,
        match="only supports faster-whisper backend",
    ):
        te._run_faster_whisper_process_isolated(
            file_path="sample.wav",
            language="en",
            profile=profile,
            settings=cast(
                te.AppConfig,
                SimpleNamespace(torch_runtime=SimpleNamespace(device="cpu", dtype="auto")),
            ),
        )


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
            del runtime_request
            observed["torch_none_setup"] = sys.modules.get("torch") is None
            assert isinstance(settings, te._TranscriptionWorkerSettings)
            observed["whisper_download_root"] = settings.models.whisper_download_root
            return False

        def prepare_assets(self, *, runtime_request: object, settings: object) -> None:
            del runtime_request, settings
            raise AssertionError("prepare_assets should not run when setup is not required")

        def load_model(self, *, runtime_request: object, settings: object) -> object:
            del runtime_request
            observed["torch_none_load"] = sys.modules.get("torch") is None
            assert isinstance(settings, te._TranscriptionWorkerSettings)
            assert settings.models.whisper_download_root == Path("/tmp/whisper-cache")
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
            del model, runtime_request
            observed["torch_none_transcribe"] = sys.modules.get("torch") is None
            observed["file_path"] = file_path
            observed["language"] = language
            assert isinstance(settings, te._TranscriptionWorkerSettings)
            assert settings.models.whisper_download_root == Path("/tmp/whisper-cache")
            return [TranscriptWord("hello", 0.0, 0.5)]

    monkeypatch.setattr(
        te,
        "reload_settings",
        lambda: (_ for _ in ()).throw(AssertionError("worker must not use ambient settings")),
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
        runtime_request=te.BackendRuntimeRequest(
            model_name="distil-large-v3",
            use_demucs=False,
            use_vad=True,
            device_spec="cpu",
            device_type="cpu",
            precision_candidates=("float32",),
            memory_tier="not_applicable",
        ),
        settings=te._TranscriptionWorkerSettings(
            models=te._TranscriptionWorkerModelsConfig(
                whisper_download_root=Path("/tmp/whisper-cache")
            )
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
    assert observed["whisper_download_root"] == Path("/tmp/whisper-cache")
    assert observed["file_path"] == "sample.wav"
    assert observed["language"] == "en"
    assert observed["closed"] is True
    assert messages[0] == ("phase", "setup_complete")
    assert messages[1] == ("phase", "model_loaded")
    assert messages[2] == ("ok", [("hello", 0.0, 0.5)])


def test_transcription_worker_entry_delegates_to_boundary_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public worker-entry wrapper should delegate casting/setup to the internal owner."""
    captured: dict[str, object] = {}
    payload = object()
    connection = object()

    def _fake_boundary_impl(*args: object, **kwargs: object) -> None:
        captured["args"] = args
        captured.update(kwargs)

    monkeypatch.setattr(
        te,
        "_transcription_worker_entry_boundary_impl",
        _fake_boundary_impl,
    )

    te._transcription_worker_entry(payload, connection)

    assert captured["args"] == (payload, connection)
    assert captured["transcription_worker_entry_impl"] is te._transcription_worker_entry_impl
    assert captured["adapter_resolver"] is te._resolve_transcription_adapter


def test_faster_whisper_setup_required_when_cache_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """faster-whisper setup should run when local cache lookup misses."""
    settings = cast(
        te.AppConfig,
        SimpleNamespace(models=SimpleNamespace(whisper_download_root=tmp_path / "model-cache")),
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
        SimpleNamespace(models=SimpleNamespace(whisper_download_root=tmp_path / "model-cache")),
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
