"""Concrete worker-payload helpers for transcript extraction boundaries."""

from __future__ import annotations

import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Protocol

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


class ProcessIsolatedProfileLike(Protocol):
    """Structural contract for process-isolated transcription profiles."""

    @property
    def backend_id(self) -> TranscriptionBackendId: ...

    @property
    def model_name(self) -> str: ...

    @property
    def use_demucs(self) -> bool: ...

    @property
    def use_vad(self) -> bool: ...


@dataclass(frozen=True)
class TranscriptionProcessPayload:
    """Serializable payload for one process-isolated transcription attempt."""

    file_path: str
    language: str
    profile: ProcessIsolatedProfileLike
    runtime_request: BackendRuntimeRequest
    settings: TranscriptionWorkerSettings


@dataclass(frozen=True, slots=True)
class TranscriptionWorkerModelsConfig:
    """Serializable model settings required by process-isolated backends."""

    whisper_download_root: Path


@dataclass(frozen=True, slots=True)
class TranscriptionWorkerSettings:
    """Serializable settings snapshot required by process-isolated workers."""

    models: TranscriptionWorkerModelsConfig


def build_transcription_worker_settings(
    settings: AppConfig,
) -> TranscriptionWorkerSettings:
    """Builds the serializable settings subset needed by worker processes."""
    return TranscriptionWorkerSettings(
        models=TranscriptionWorkerModelsConfig(
            whisper_download_root=settings.models.whisper_download_root,
        ),
    )


def build_transcription_process_payload(
    *,
    file_path: str,
    language: str,
    profile: ProcessIsolatedProfileLike,
    runtime_request: BackendRuntimeRequest,
    settings: AppConfig,
) -> TranscriptionProcessPayload:
    """Builds a serializable worker payload from the active runtime settings."""
    return TranscriptionProcessPayload(
        file_path=file_path,
        language=language,
        profile=profile,
        runtime_request=runtime_request,
        settings=build_transcription_worker_settings(settings),
    )


def release_transcription_runtime_memory(
    *,
    model: object | None,
    logger: logging.Logger,
) -> None:
    """Releases best-effort accelerator memory after one transcript run."""
    del model
    gc.collect()
    torch_module = sys.modules.get("torch")
    if not isinstance(torch_module, ModuleType):
        return
    mps_module = getattr(torch_module, "mps", None)
    if isinstance(mps_module, ModuleType):
        is_available = getattr(mps_module, "is_available", None)
        empty_cache = getattr(mps_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch MPS cache after transcription.",
                exc_info=True,
            )
    cuda_module = getattr(torch_module, "cuda", None)
    if isinstance(cuda_module, ModuleType):
        is_available = getattr(cuda_module, "is_available", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch CUDA cache after transcription.",
                exc_info=True,
            )
