"""Unit tests for public-boundary process-isolation glue."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Never, cast

import pytest

from ser._internal.transcription import public_boundary_process as boundary_process
from ser._internal.transcription.process_worker import (
    TranscriptionProcessPayload,
    TranscriptionWorkerModelsConfig,
    TranscriptionWorkerSettings,
)
from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest
else:
    BackendRuntimeRequest = object

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Profile:
    """Minimal profile satisfying the process-isolated public boundary."""

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
            use_vad=False,
        ),
    )


def test_run_faster_whisper_process_isolated_from_public_boundary_injects_settings_snapshot() -> (
    None
):
    """Boundary runner should pass explicit settings and TranscriptWord factory downstream."""
    settings = cast(AppConfig, SimpleNamespace(name="settings"))
    profile = _Profile()
    captured: dict[str, object] = {}

    def _run_impl(**kwargs: object) -> list[TranscriptWord]:
        captured.update(kwargs)
        settings_resolver = kwargs["settings_resolver"]
        assert callable(settings_resolver)
        assert settings_resolver() is settings
        transcript_word_factory = cast(type[TranscriptWord], kwargs["transcript_word_factory"])
        assert transcript_word_factory("word", 0.0, 0.5) == TranscriptWord("word", 0.0, 0.5)
        return [TranscriptWord("word", 0.0, 0.5)]

    resolved = boundary_process.run_faster_whisper_process_isolated_from_public_boundary(
        file_path="sample.wav",
        language="en",
        profile=profile,
        settings=settings,
        run_faster_whisper_process_isolated_impl=_run_impl,
        runtime_request_resolver=lambda _profile, _settings: _runtime_request(),
        payload_factory=lambda **kwargs: kwargs,
        spawn_context_resolver=lambda: "spawn-context",
        worker_entry=lambda _payload, _connection: None,
        recv_worker_message_fn=lambda _connection, *, stage: ("phase", "setup_complete"),
        raise_worker_error_fn=lambda _message: (_ for _ in ()).throw(RuntimeError("boom")),
        terminate_worker_process_fn=lambda _process: None,
        logger=logging.getLogger("ser.tests.public_boundary_process"),
        error_factory=RuntimeError,
        terminate_grace_seconds=0.1,
    )

    assert resolved == [TranscriptWord("word", 0.0, 0.5)]
    assert captured["file_path"] == "sample.wav"
    assert captured["language"] == "en"
    assert captured["profile"] is profile


def test_recv_worker_message_from_public_boundary_delegates_to_impl() -> None:
    """Boundary receiver should pass through the stage and error factory."""
    connection = object()
    captured: dict[str, object] = {}

    resolved = boundary_process.recv_worker_message_from_public_boundary(
        connection,
        stage="setup",
        recv_worker_message_impl=lambda active_connection, *, stage, error_factory: captured.update(
            {
                "connection": active_connection,
                "stage": stage,
                "error_factory": error_factory,
            }
        )
        or ("phase", "setup_complete"),
        error_factory=RuntimeError,
    )

    assert resolved == ("phase", "setup_complete")
    assert captured == {
        "connection": connection,
        "stage": "setup",
        "error_factory": RuntimeError,
    }


def test_raise_worker_error_from_public_boundary_delegates_to_impl() -> None:
    """Boundary error raiser should forward the worker payload unchanged."""
    captured: dict[str, object] = {}

    def _raise_impl(message: object, *, error_factory: type[Exception]) -> Never:
        captured.update({"message": message, "error_factory": error_factory})
        raise error_factory("worker boom")

    with pytest.raises(RuntimeError, match="worker boom"):
        boundary_process.raise_worker_error_from_public_boundary(
            ("err", "setup", "RuntimeError", "worker boom"),
            raise_worker_error_impl=_raise_impl,
            error_factory=RuntimeError,
        )

    assert captured["message"] == ("err", "setup", "RuntimeError", "worker boom")
    assert captured["error_factory"] is RuntimeError


def test_terminate_worker_process_from_public_boundary_delegates_to_impl() -> None:
    """Boundary terminator should pass grace-period policy to the implementation."""
    process = object()
    captured: dict[str, object] = {}

    boundary_process.terminate_worker_process_from_public_boundary(
        process,
        terminate_worker_process_impl=lambda active_process, *, terminate_grace_seconds, kill_grace_seconds: captured.update(
            {
                "process": active_process,
                "terminate_grace_seconds": terminate_grace_seconds,
                "kill_grace_seconds": kill_grace_seconds,
            }
        ),
        terminate_grace_seconds=0.2,
        kill_grace_seconds=0.3,
    )

    assert captured == {
        "process": process,
        "terminate_grace_seconds": 0.2,
        "kill_grace_seconds": 0.3,
    }


def test_spawn_context_for_public_boundary_requests_spawn() -> None:
    """Boundary spawn helper should always request the multiprocessing spawn context."""
    captured: dict[str, object] = {}

    resolved = boundary_process.spawn_context_for_public_boundary(
        get_context=lambda mode: captured.setdefault("mode", mode) or "context"
    )

    assert resolved == "spawn"
    assert captured["mode"] == "spawn"


def test_transcription_worker_entry_from_public_boundary_casts_payload_and_adapter() -> None:
    """Boundary worker entry should resolve worker settings and adapter through boundary helpers."""
    payload = TranscriptionProcessPayload(
        file_path="sample.wav",
        language="en",
        profile=_Profile(use_vad=False),
        runtime_request=_runtime_request(),
        settings=TranscriptionWorkerSettings(
            models=TranscriptionWorkerModelsConfig(
                whisper_download_root=Path("/tmp/whisper-cache"),
            )
        ),
    )
    captured: dict[str, object] = {}

    boundary_process.transcription_worker_entry_from_public_boundary(
        payload,
        connection=object(),
        transcription_worker_entry_impl=lambda active_payload, active_connection, *, settings_resolver, adapter_resolver: captured.update(
            {
                "payload": active_payload,
                "connection": active_connection,
                "settings": settings_resolver(),
                "adapter": adapter_resolver("faster_whisper"),
            }
        ),
        adapter_resolver=lambda backend_id: {"backend_id": backend_id},
    )

    assert captured["payload"] == payload
    assert captured["settings"] == payload.settings
    assert captured["adapter"] == {"backend_id": "faster_whisper"}
