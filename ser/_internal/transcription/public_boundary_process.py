"""Public-boundary glue for process-isolated transcription execution."""

from __future__ import annotations

import logging
from collections.abc import Callable
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Never, Protocol, TypeVar, cast

from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId

from .process_isolation import WorkerMessage
from .process_worker import TranscriptionProcessPayload

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


class _ProcessIsolatedProfile(Protocol):
    """Minimal profile contract for public-boundary worker execution."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns the transcription backend identifier."""
        ...


type _ErrorFactory = Callable[[str], Exception]
type _RuntimeRequestResolver[_TProfile: _ProcessIsolatedProfile] = Callable[
    [_TProfile, AppConfig], BackendRuntimeRequest
]
type _RunFasterWhisperProcessIsolatedImpl = Callable[..., list[TranscriptWord]]
type _RecvWorkerMessageImpl = Callable[..., WorkerMessage]
type _RaiseWorkerErrorImpl = Callable[..., Never]
type _TerminateWorkerProcessImpl = Callable[..., None]
type _AdapterResolver = Callable[[TranscriptionBackendId], object]
type _TranscriptionWorkerEntryImpl = Callable[..., None]

_TProfile = TypeVar("_TProfile", bound=_ProcessIsolatedProfile)


def run_faster_whisper_process_isolated_from_public_boundary(
    *,
    file_path: str,
    language: str,
    profile: _TProfile,
    settings: AppConfig,
    run_faster_whisper_process_isolated_impl: _RunFasterWhisperProcessIsolatedImpl,
    runtime_request_resolver: _RuntimeRequestResolver[_TProfile],
    payload_factory: Callable[..., object],
    spawn_context_resolver: Callable[[], object],
    worker_entry: Callable[[object, object], None],
    recv_worker_message_fn: Callable[..., WorkerMessage],
    raise_worker_error_fn: Callable[..., Never],
    terminate_worker_process_fn: Callable[[object], None],
    logger: logging.Logger,
    error_factory: _ErrorFactory,
    terminate_grace_seconds: float,
) -> list[TranscriptWord]:
    """Runs the isolated faster-whisper workflow with one explicit settings snapshot."""
    return run_faster_whisper_process_isolated_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings_resolver=lambda: settings,
        runtime_request_resolver=runtime_request_resolver,
        payload_factory=payload_factory,
        get_spawn_context=spawn_context_resolver,
        worker_entry=worker_entry,
        recv_worker_message_fn=recv_worker_message_fn,
        raise_worker_error_fn=raise_worker_error_fn,
        terminate_worker_process_fn=terminate_worker_process_fn,
        transcript_word_factory=TranscriptWord,
        logger=logger,
        error_factory=error_factory,
        terminate_grace_seconds=terminate_grace_seconds,
    )


def recv_worker_message_from_public_boundary(
    connection: object,
    *,
    stage: str,
    recv_worker_message_impl: _RecvWorkerMessageImpl,
    error_factory: _ErrorFactory,
) -> WorkerMessage:
    """Receives one worker message with the public boundary's error contract."""
    return recv_worker_message_impl(
        cast(Connection, connection),
        stage=stage,
        error_factory=error_factory,
    )


def raise_worker_error_from_public_boundary(
    message: WorkerMessage,
    *,
    raise_worker_error_impl: _RaiseWorkerErrorImpl,
    error_factory: _ErrorFactory,
) -> Never:
    """Raises one worker error using the public boundary's domain exception."""
    raise_worker_error_impl(message, error_factory=error_factory)


def terminate_worker_process_from_public_boundary(
    process: object,
    *,
    terminate_worker_process_impl: _TerminateWorkerProcessImpl,
    terminate_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Terminates one worker process with the boundary's grace-period policy."""
    terminate_worker_process_impl(
        cast(BaseProcess, process),
        terminate_grace_seconds=terminate_grace_seconds,
        kill_grace_seconds=kill_grace_seconds,
    )


def spawn_context_for_public_boundary(
    *,
    get_context: Callable[[str], object],
) -> object:
    """Returns the multiprocessing spawn context used by the public boundary."""
    return get_context("spawn")


def resolve_transcription_adapter_from_public_boundary(
    backend_id: TranscriptionBackendId,
    *,
    adapter_resolver: _AdapterResolver,
) -> object:
    """Resolves one transcription adapter for process-isolated worker execution."""
    return adapter_resolver(backend_id)


def transcription_worker_entry_from_public_boundary(
    payload: object,
    connection: object,
    *,
    transcription_worker_entry_impl: _TranscriptionWorkerEntryImpl,
    adapter_resolver: _AdapterResolver,
) -> None:
    """Runs one isolated worker entrypoint with public-boundary casting rules."""
    active_payload = cast(TranscriptionProcessPayload, payload)
    transcription_worker_entry_impl(
        active_payload,
        cast(Connection, connection),
        settings_resolver=lambda: active_payload.settings,
        adapter_resolver=lambda backend_id: (
            resolve_transcription_adapter_from_public_boundary(
                backend_id,
                adapter_resolver=adapter_resolver,
            )
        ),
    )
