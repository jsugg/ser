"""Unit tests for process-worker payload and cleanup helpers."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from ser._internal.transcription import process_worker
from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest
else:
    BackendRuntimeRequest = object

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Profile:
    """Minimal process-isolated transcription profile stub."""

    backend_id: TranscriptionBackendId = "faster_whisper"
    model_name: str = "tiny"
    use_demucs: bool = False
    use_vad: bool = True


def _runtime_request() -> BackendRuntimeRequest:
    return cast(
        BackendRuntimeRequest,
        SimpleNamespace(
            model_name="tiny",
            use_demucs=False,
            use_vad=True,
        ),
    )


def test_build_transcription_worker_settings_projects_whisper_root() -> None:
    """Worker settings should keep only the serializable model subset."""
    settings = cast(
        AppConfig,
        SimpleNamespace(models=SimpleNamespace(whisper_download_root=Path("/tmp/whisper-cache"))),
    )

    resolved = process_worker.build_transcription_worker_settings(settings)

    assert resolved.models.whisper_download_root == Path("/tmp/whisper-cache")


def test_build_transcription_process_payload_embeds_worker_settings() -> None:
    """Process payload should include the stripped worker settings snapshot."""
    settings = cast(
        AppConfig,
        SimpleNamespace(models=SimpleNamespace(whisper_download_root=Path("/tmp/whisper-cache"))),
    )
    profile = _Profile()
    runtime_request = _runtime_request()

    resolved = process_worker.build_transcription_process_payload(
        file_path="sample.wav",
        language="en",
        profile=profile,
        runtime_request=runtime_request,
        settings=settings,
    )

    assert resolved.file_path == "sample.wav"
    assert resolved.language == "en"
    assert resolved.profile is profile
    assert resolved.runtime_request == runtime_request
    assert resolved.settings.models.whisper_download_root == Path("/tmp/whisper-cache")


def test_release_transcription_runtime_memory_clears_available_accelerator_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup helper should empty available MPS and CUDA caches."""
    cache_events: list[str] = []
    torch_module = ModuleType("torch")
    mps_module = ModuleType("torch.mps")
    cuda_module = ModuleType("torch.cuda")
    mps_module.is_available = lambda: True  # type: ignore[attr-defined]
    mps_module.empty_cache = lambda: cache_events.append("mps")  # type: ignore[attr-defined]
    cuda_module.is_available = lambda: True  # type: ignore[attr-defined]
    cuda_module.empty_cache = lambda: cache_events.append("cuda")  # type: ignore[attr-defined]
    torch_module.mps = mps_module  # type: ignore[attr-defined]
    torch_module.cuda = cuda_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    process_worker.release_transcription_runtime_memory(
        model=object(),
        logger=logging.getLogger("ser.tests.process_worker"),
    )

    assert cache_events == ["mps", "cuda"]


def test_release_transcription_runtime_memory_logs_and_ignores_cache_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cleanup helper should swallow accelerator cache errors after debug logging."""
    torch_module = ModuleType("torch")
    mps_module = ModuleType("torch.mps")
    cuda_module = ModuleType("torch.cuda")
    mps_module.is_available = lambda: True  # type: ignore[attr-defined]
    cuda_module.is_available = lambda: True  # type: ignore[attr-defined]

    def _raise_cache_error() -> None:
        raise RuntimeError("cache boom")

    mps_module.empty_cache = _raise_cache_error  # type: ignore[attr-defined]
    cuda_module.empty_cache = _raise_cache_error  # type: ignore[attr-defined]
    torch_module.mps = mps_module  # type: ignore[attr-defined]
    torch_module.cuda = cuda_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    with caplog.at_level(logging.DEBUG):
        process_worker.release_transcription_runtime_memory(
            model=object(),
            logger=logging.getLogger("ser.tests.process_worker"),
        )

    assert "Ignored failure while emptying torch MPS cache after transcription." in caplog.text
    assert "Ignored failure while emptying torch CUDA cache after transcription." in caplog.text
