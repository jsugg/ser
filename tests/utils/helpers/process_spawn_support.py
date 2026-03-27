"""Importable module-level helpers for spawned-process test coverage."""

from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import TYPE_CHECKING

from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


@dataclass(frozen=True, slots=True)
class RuntimeWorkerPayload:
    """Serializable runtime payload for spawned worker contract tests."""

    result: str = "ok"
    compute_delay_seconds: float = 0.0
    error_type: str | None = None
    error_message: str = ""


def runtime_worker_entry(payload: RuntimeWorkerPayload, connection: Connection) -> None:
    """Emits the standard runtime worker protocol from a spawned helper worker."""
    try:
        connection.send(("phase", "setup_complete"))
        if payload.compute_delay_seconds > 0.0:
            time.sleep(payload.compute_delay_seconds)
        if isinstance(payload.error_type, str):
            connection.send(("err", payload.error_type, payload.error_message))
        else:
            connection.send(("ok", payload.result))
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class FakeTranscriptionProfile:
    """Serializable transcription profile for spawned helper workers."""

    backend_id: TranscriptionBackendId = "faster_whisper"
    model_name: str = "tiny"
    use_demucs: bool = False
    use_vad: bool = True


@dataclass(frozen=True, slots=True)
class FakeModelsConfig:
    """Serializable models config for transcription worker settings."""

    whisper_download_root: Path


@dataclass(frozen=True, slots=True)
class FakeSettings:
    """Serializable settings snapshot for spawned transcription workers."""

    models: FakeModelsConfig


@dataclass(frozen=True, slots=True)
class FakeTranscriptionPayload:
    """Serializable payload for process-isolated transcription helper tests."""

    file_path: str
    language: str
    profile: FakeTranscriptionProfile
    runtime_request: BackendRuntimeRequest
    settings: FakeSettings


def build_transcription_payload(
    *,
    file_path: str,
    language: str,
    profile: FakeTranscriptionProfile,
    runtime_request: BackendRuntimeRequest,
    settings: FakeSettings,
) -> FakeTranscriptionPayload:
    """Builds one serializable payload for spawned transcription tests."""
    return FakeTranscriptionPayload(
        file_path=file_path,
        language=language,
        profile=profile,
        runtime_request=runtime_request,
        settings=settings,
    )


def transcription_success_worker(
    payload: FakeTranscriptionPayload,
    connection: Connection,
) -> None:
    """Emits the standard transcription worker success protocol."""
    del payload
    try:
        connection.send(("phase", "setup_complete"))
        connection.send(("phase", "model_loaded"))
        connection.send(("ok", [("hello", 0.0, 0.5), ("world", 0.5, 1.0)]))
    finally:
        connection.close()


def transcription_error_worker(
    payload: FakeTranscriptionPayload,
    connection: Connection,
) -> None:
    """Emits the standard transcription worker error protocol."""
    del payload
    try:
        connection.send(("phase", "setup_complete"))
        connection.send(("err", "model_load", "RuntimeError", "boom"))
    finally:
        connection.close()
