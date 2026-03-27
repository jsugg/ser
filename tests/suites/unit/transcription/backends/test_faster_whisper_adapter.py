"""Focused unit coverage for faster-whisper adapter runtime branches."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.faster_whisper import FasterWhisperAdapter

pytestmark = pytest.mark.unit


def _runtime_request(
    *,
    model_name: str = "distil-large-v3",
    device_spec: str = "cpu",
    device_type: str = "cpu",
    precision_candidates: tuple[str, ...] = ("float32",),
    use_demucs: bool = False,
    use_vad: bool = True,
) -> BackendRuntimeRequest:
    """Build one runtime request for adapter-focused tests."""
    return BackendRuntimeRequest(
        model_name=model_name,
        use_demucs=use_demucs,
        use_vad=use_vad,
        device_spec=device_spec,
        device_type=device_type,
        precision_candidates=precision_candidates,
    )


def _settings(download_root: Path) -> AppConfig:
    """Build a minimal settings stub for download-root access."""
    return cast(
        AppConfig,
        SimpleNamespace(models=SimpleNamespace(whisper_download_root=download_root)),
    )


@dataclass(slots=True)
class _FspathValue:
    """Simple os.PathLike stub for cache-probe responses."""

    path: str

    def __fspath__(self) -> str:
        return self.path


@dataclass(slots=True)
class _WordStub:
    """Transcript word stub mirroring faster-whisper word attributes."""

    word: str
    start: float | None
    end: float | None


@dataclass(slots=True)
class _SegmentStub:
    """Transcript segment stub with `words` payload."""

    words: object


def test_setup_required_returns_false_for_empty_model_name(tmp_path: Path) -> None:
    """Blank model names should skip remote cache probing entirely."""
    adapter = FasterWhisperAdapter()

    assert (
        adapter.setup_required(
            runtime_request=_runtime_request(model_name="  "),
            settings=_settings(tmp_path),
        )
        is False
    )


def test_setup_required_returns_false_for_existing_directory(tmp_path: Path) -> None:
    """Local model directories should not request additional setup."""
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()

    adapter = FasterWhisperAdapter()

    assert (
        adapter.setup_required(
            runtime_request=_runtime_request(model_name=str(model_dir)),
            settings=_settings(tmp_path),
        )
        is False
    )


def test_setup_required_returns_true_when_local_only_probe_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failed local-only cache probe should request asset preparation."""

    def _download_model(_model_name: str, *, local_files_only: bool, cache_dir: str) -> str:
        assert local_files_only is True
        assert cache_dir == str(tmp_path)
        raise RuntimeError("cache miss")

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(download_model=_download_model)
            if name == "faster_whisper.utils"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )

    adapter = FasterWhisperAdapter()

    assert adapter.setup_required(
        runtime_request=_runtime_request(),
        settings=_settings(tmp_path),
    )


def test_setup_required_accepts_string_and_pathlike_directory_probes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing cache directories reported as string or PathLike should be accepted."""
    cached_dir = tmp_path / "cached-model"
    cached_dir.mkdir()
    adapter = FasterWhisperAdapter()

    def _download_from_string(
        _model_name: str,
        *,
        local_files_only: bool,
        cache_dir: str,
    ) -> str:
        assert local_files_only is True
        assert cache_dir == str(tmp_path)
        return str(cached_dir)

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(download_model=_download_from_string)
            if name == "faster_whisper.utils"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )
    assert (
        adapter.setup_required(
            runtime_request=_runtime_request(),
            settings=_settings(tmp_path),
        )
        is False
    )

    def _download_from_pathlike(
        _model_name: str,
        *,
        local_files_only: bool,
        cache_dir: str,
    ) -> _FspathValue:
        assert local_files_only is True
        assert cache_dir == str(tmp_path)
        return _FspathValue(str(cached_dir))

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(download_model=_download_from_pathlike)
            if name == "faster_whisper.utils"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )
    assert (
        adapter.setup_required(
            runtime_request=_runtime_request(),
            settings=_settings(tmp_path),
        )
        is False
    )


def test_prepare_assets_downloads_missing_model_into_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Asset preparation should download into the configured whisper cache root."""
    captured: dict[str, object] = {}

    def _download_model(model_name: str, *, local_files_only: bool, cache_dir: str) -> str:
        captured["model_name"] = model_name
        captured["local_files_only"] = local_files_only
        captured["cache_dir"] = cache_dir
        return cache_dir

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(download_model=_download_model)
            if name == "faster_whisper.utils"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )

    adapter = FasterWhisperAdapter()
    adapter.prepare_assets(
        runtime_request=_runtime_request(),
        settings=_settings(tmp_path),
    )

    assert captured == {
        "model_name": "distil-large-v3",
        "local_files_only": False,
        "cache_dir": str(tmp_path),
    }
    assert tmp_path.is_dir()


def test_prepare_assets_ignores_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing faster-whisper utilities should become a no-op."""
    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )

    adapter = FasterWhisperAdapter()
    adapter.prepare_assets(
        runtime_request=_runtime_request(),
        settings=_settings(tmp_path),
    )

    assert not any(tmp_path.iterdir())


def test_load_model_raises_clear_error_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Model loading should surface actionable dependency remediation."""
    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )

    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="Missing faster-whisper dependencies"):
        adapter.load_model(
            runtime_request=_runtime_request(),
            settings=_settings(tmp_path),
        )


def test_load_model_requires_whisper_model_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The adapter should reject packages that do not expose WhisperModel."""
    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace()
            if name == "faster_whisper"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )

    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="does not expose WhisperModel"):
        adapter.load_model(
            runtime_request=_runtime_request(),
            settings=_settings(tmp_path),
        )


def test_load_model_uses_cpu_int8_for_mps_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MPS requests should load faster-whisper on CPU with int8 compute."""
    captured: dict[str, object] = {}

    class _FakeWhisperModel:
        def __init__(
            self,
            model_name: str,
            *,
            device: str,
            compute_type: str,
            download_root: str,
        ) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["compute_type"] = compute_type
            captured["download_root"] = download_root

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(WhisperModel=_FakeWhisperModel)
            if name == "faster_whisper"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )

    adapter = FasterWhisperAdapter()
    model = adapter.load_model(
        runtime_request=_runtime_request(
            device_spec="mps",
            device_type="mps",
            precision_candidates=("float16", "float32"),
        ),
        settings=_settings(tmp_path),
    )

    assert isinstance(model, _FakeWhisperModel)
    assert captured == {
        "model_name": "distil-large-v3",
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(tmp_path),
    }


def test_load_model_uses_cuda_float16_when_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CUDA requests should preserve the resolved device and first precision candidate."""
    captured: dict[str, object] = {}

    class _FakeWhisperModel:
        def __init__(
            self,
            model_name: str,
            *,
            device: str,
            compute_type: str,
            download_root: str,
        ) -> None:
            captured["model_name"] = model_name
            captured["device"] = device
            captured["compute_type"] = compute_type
            captured["download_root"] = download_root

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.import_module",
        lambda name: (
            SimpleNamespace(WhisperModel=_FakeWhisperModel)
            if name == "faster_whisper"
            else (_ for _ in ()).throw(ModuleNotFoundError(name))
        ),
    )

    adapter = FasterWhisperAdapter()
    adapter.load_model(
        runtime_request=_runtime_request(
            device_spec="cuda:0",
            device_type="cuda",
            precision_candidates=("float16", "float32"),
        ),
        settings=_settings(tmp_path),
    )

    assert captured["device"] == "cuda:0"
    assert captured["compute_type"] == "float16"


def test_transcribe_requires_callable_model_entrypoint(tmp_path: Path) -> None:
    """Loaded model objects must expose a callable transcribe method."""
    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="does not expose a callable transcribe"):
        adapter.transcribe(
            model=object(),
            runtime_request=_runtime_request(),
            file_path="sample.wav",
            language="en",
            settings=_settings(tmp_path),
        )


def test_transcribe_wraps_backend_errors(
    tmp_path: Path,
) -> None:
    """Adapter should normalize backend transcription failures."""

    class _FailingModel:
        def transcribe(self, **_kwargs: object) -> tuple[object, object]:
            raise ValueError("boom")

    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="Failed to transcribe audio"):
        adapter.transcribe(
            model=_FailingModel(),
            runtime_request=_runtime_request(),
            file_path="sample.wav",
            language="en",
            settings=_settings(tmp_path),
        )


def test_transcribe_rejects_invalid_result_envelope(tmp_path: Path) -> None:
    """Unexpected backend result shapes should fail fast."""

    class _InvalidModel:
        def transcribe(self, **_kwargs: object) -> object:
            return ["not", "a", "tuple"]

    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="Unexpected result envelope"):
        adapter.transcribe(
            model=_InvalidModel(),
            runtime_request=_runtime_request(),
            file_path="sample.wav",
            language="en",
            settings=_settings(tmp_path),
        )


def test_transcribe_rejects_non_iterable_segments(tmp_path: Path) -> None:
    """Segments payloads must be iterable for transcript formatting."""

    class _InvalidModel:
        def transcribe(self, **_kwargs: object) -> tuple[object, object]:
            return (1, {"language": "en"})

    adapter = FasterWhisperAdapter()

    with pytest.raises(RuntimeError, match="Unexpected segment stream type"):
        adapter.transcribe(
            model=_InvalidModel(),
            runtime_request=_runtime_request(),
            file_path="sample.wav",
            language="en",
            settings=_settings(tmp_path),
        )


def test_transcribe_formats_word_timestamps_and_logs_demucs_warning(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Transcription should format valid words and skip incomplete segment entries."""

    class _FakeModel:
        def transcribe(self, **kwargs: object) -> tuple[object, object]:
            assert kwargs == {
                "audio": "sample.wav",
                "language": "en",
                "word_timestamps": True,
                "vad_filter": True,
                "beam_size": 5,
            }
            return (
                [
                    _SegmentStub(
                        words=[
                            _WordStub(word="hello", start=0.1, end=0.4),
                            _WordStub(word="skip-missing-start", start=None, end=0.5),
                            _WordStub(word="skip-missing-end", start=0.5, end=None),
                        ]
                    ),
                    _SegmentStub(words="not-a-sequence"),
                    _SegmentStub(words=(_WordStub(word="world", start=0.5, end=0.8),)),
                ],
                {"language": "en"},
            )

    caplog.set_level("WARNING")
    adapter = FasterWhisperAdapter()

    transcript = adapter.transcribe(
        model=_FakeModel(),
        runtime_request=_runtime_request(use_demucs=True),
        file_path="sample.wav",
        language="en",
        settings=_settings(tmp_path),
    )

    assert transcript == [
        TranscriptWord(word="hello", start_seconds=0.1, end_seconds=0.4),
        TranscriptWord(word="world", start_seconds=0.5, end_seconds=0.8),
    ]
    assert "demucs flag is ignored" in caplog.text


def test_is_module_available_handles_sys_modules_and_find_spec_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability probing should tolerate loaded modules and invalid specs."""
    adapter = FasterWhisperAdapter()
    module_name = "synthetic_faster_whisper_dependency"
    original = sys.modules.get(module_name)
    sys.modules[module_name] = ModuleType(module_name)
    try:
        assert adapter._is_module_available(module_name) is True
    finally:
        if original is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original

    monkeypatch.setattr(
        "ser.transcript.backends.faster_whisper.importlib.util.find_spec",
        lambda name: (_ for _ in ()).throw(ValueError(name)),
    )
    assert adapter._is_module_available("missing_dependency") is False
