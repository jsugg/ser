"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from ser._internal.transcription import public_boundary_support as _boundary_support
from ser._internal.transcription.process_isolation import (
    WorkerMessage,
)
from ser._internal.transcription.process_isolation import (
    raise_worker_error as _raise_worker_error_impl,
)
from ser._internal.transcription.process_isolation import (
    recv_worker_message as _recv_worker_message_impl,
)
from ser._internal.transcription.process_isolation import (
    should_use_process_isolated_path as _should_use_process_isolated_path_impl,
)
from ser._internal.transcription.process_isolation import (
    terminate_worker_process as _terminate_worker_process_impl,
)
from ser._internal.transcription.process_isolation import (
    transcription_worker_entry as _transcription_worker_entry_impl,
)
from ser._internal.transcription.process_worker import (
    TranscriptionProcessPayload as _TranscriptionProcessPayload,
)
from ser._internal.transcription.process_worker import (
    TranscriptionWorkerModelsConfig as _TranscriptionWorkerModelsConfig,
)
from ser._internal.transcription.process_worker import (
    TranscriptionWorkerSettings as _TranscriptionWorkerSettings,
)
from ser._internal.transcription.process_worker import (
    release_transcription_runtime_memory as _release_transcription_runtime_memory_impl,
)
from ser._internal.transcription.public_boundary_process import (
    raise_worker_error_from_public_boundary as _raise_worker_error_boundary_impl,
)
from ser._internal.transcription.public_boundary_process import (
    recv_worker_message_from_public_boundary as _recv_worker_message_boundary_impl,
)
from ser._internal.transcription.public_boundary_process import (
    resolve_transcription_adapter_from_public_boundary as _resolve_transcription_adapter_boundary_impl,
)
from ser._internal.transcription.public_boundary_process import (
    spawn_context_for_public_boundary as _spawn_context_boundary_impl,
)
from ser._internal.transcription.public_boundary_process import (
    terminate_worker_process_from_public_boundary as _terminate_worker_process_boundary_impl,
)
from ser._internal.transcription.public_boundary_process import (
    transcription_worker_entry_from_public_boundary as _transcription_worker_entry_boundary_impl,
)
from ser._internal.transcription.runtime_profile import (
    runtime_request_from_profile as _runtime_request_from_profile_impl,
)
from ser.config import AppConfig, reload_settings
from ser.domain import TranscriptWord
from ser.profiles import (
    ProfileTranscriptionDefaults,
    TranscriptionBackendId,
    get_profile_catalog,
)
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.transcript.backends import (
    BackendRuntimeRequest,
    CompatibilityReport,
    resolve_transcription_backend_adapter,
)
from ser.transcript.runtime_policy import (
    DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    resolve_transcription_runtime_policy,
)
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult

logger: logging.Logger = get_logger(__name__)
_TERMINATE_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 2.0

_PROCESS_WORKER_NAMESPACE = (
    _TranscriptionProcessPayload,
    _TranscriptionWorkerModelsConfig,
    _TranscriptionWorkerSettings,
)


class TranscriptionError(RuntimeError):
    """Raised when transcript extraction fails for operational reasons."""


class WhisperWord(Protocol):
    """Protocol for stable-whisper word-level transcript entries."""

    word: str
    start: float | None
    end: float | None


@dataclass(frozen=True)
class TranscriptionProfile:
    """Runtime profile settings used by transcription backends."""

    backend_id: TranscriptionBackendId = field(
        default_factory=lambda: _resolve_catalog_transcription_defaults("fast").backend_id
    )
    model_name: str = field(
        default_factory=lambda: _resolve_catalog_transcription_defaults("fast").model_name
    )
    use_demucs: bool = field(
        default_factory=lambda: _resolve_catalog_transcription_defaults("fast").use_demucs
    )
    use_vad: bool = field(
        default_factory=lambda: _resolve_catalog_transcription_defaults("fast").use_vad
    )


type _WorkerMessage = WorkerMessage
type _CompatibilityIssueKind = Literal["noise", "operational"]

_EMITTED_COMPATIBILITY_ISSUE_KEYS: set[tuple[str, str, str]] = set()


def _resolve_catalog_transcription_defaults(
    profile: Literal["fast", "medium", "accurate", "accurate-research"],
) -> ProfileTranscriptionDefaults:
    """Returns catalog-owned transcription defaults for one runtime profile."""
    return get_profile_catalog()[profile].transcription_defaults


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or reloads a boundary-local settings snapshot."""
    return settings if settings is not None else reload_settings()


def _resolve_transcription_profile_for_settings(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig,
) -> TranscriptionProfile:
    """Resolves one transcription profile against an explicit settings snapshot."""
    return cast(
        TranscriptionProfile,
        _boundary_support.resolve_transcription_profile_for_settings(
            profile,
            settings=settings,
            profile_factory=TranscriptionProfile,
            error_factory=TranscriptionError,
        ),
    )


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    return _resolve_transcription_profile_for_settings(
        profile,
        settings=_resolve_boundary_settings(settings),
    )


def _runtime_request_from_profile(
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one backend runtime request from transcription profile settings."""
    return _runtime_request_from_profile_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_policy_resolver=resolve_transcription_runtime_policy,
        default_mps_low_memory_threshold_gb=DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    )


def _check_adapter_compatibility(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest | None = None,
) -> CompatibilityReport:
    """Validates backend compatibility and logs non-blocking compatibility issues."""
    return _boundary_support.check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        logger=logger,
        error_factory=TranscriptionError,
    )


def mark_compatibility_issues_as_emitted(
    *,
    backend_id: TranscriptionBackendId,
    issue_kind: _CompatibilityIssueKind,
    issue_codes: tuple[str, ...],
) -> None:
    """Marks compatibility issues as already emitted to prevent duplicate logs."""
    _boundary_support.mark_compatibility_issues_as_emitted(
        backend_id=backend_id,
        issue_kind=issue_kind,
        issue_codes=issue_codes,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
    )


def _transcription_setup_required(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> bool:
    """Returns whether a setup/download phase is needed before model load."""
    return _boundary_support.transcription_setup_required(
        active_profile=active_profile,
        settings=settings,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        logger=logger,
        error_factory=TranscriptionError,
    )


def _prepare_transcription_assets(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> None:
    """Ensures required stable-whisper model assets are present locally."""
    _boundary_support.prepare_transcription_assets(
        active_profile=active_profile,
        settings=settings,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        logger=logger,
        error_factory=TranscriptionError,
    )


def _load_whisper_model_for_settings(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig,
) -> object:
    """Loads one transcription model for an explicit settings snapshot."""
    return _boundary_support.load_whisper_model_for_settings(
        profile=profile,
        settings=settings,
        profile_factory=TranscriptionProfile,
        logger=logger,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        error_factory=TranscriptionError,
    )


def load_whisper_model(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> object:
    """Loads the configured transcription model for resolved runtime settings.

    Returns:
        The loaded Whisper model instance.
    """
    return _load_whisper_model_for_settings(
        profile=profile,
        settings=_resolve_boundary_settings(settings),
    )


def _should_use_process_isolated_path(profile: TranscriptionProfile) -> bool:
    """Returns whether one transcription profile should execute in a worker process."""
    return _should_use_process_isolated_path_impl(profile)


def _runtime_request_for_isolated_faster_whisper(
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one faster-whisper runtime request without importing torch in worker."""
    return _boundary_support._runtime_request_for_isolated_faster_whisper(
        profile=profile,
        settings=settings,
        error_factory=TranscriptionError,
        logger=logger,
    )


def _run_faster_whisper_process_isolated(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs faster-whisper setup/load/transcribe inside one spawned worker process."""
    return _boundary_support.run_faster_whisper_process_isolated(
        file_path=file_path,
        language=language,
        profile=profile,
        settings=settings,
        transcript_word_factory=TranscriptWord,
        spawn_context_resolver=_spawn_context,
        worker_entry=_transcription_worker_entry,
        recv_worker_message_fn=_recv_worker_message,
        raise_worker_error_fn=_raise_worker_error,
        terminate_worker_process_fn=_terminate_worker_process,
        logger=logger,
        error_factory=TranscriptionError,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
    )


def _recv_worker_message(connection: object, *, stage: str) -> _WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_boundary_impl(
        connection,
        recv_worker_message_impl=_recv_worker_message_impl,
        stage=stage,
        error_factory=TranscriptionError,
    )


def _raise_worker_error(message: object) -> Never:
    """Raises one transcription-domain error from a worker payload."""
    _raise_worker_error_boundary_impl(
        cast(_WorkerMessage, message),
        raise_worker_error_impl=_raise_worker_error_impl,
        error_factory=TranscriptionError,
    )


def _terminate_worker_process(process: object) -> None:
    """Terminates a worker process with kill fallback."""
    _terminate_worker_process_boundary_impl(
        process,
        terminate_worker_process_impl=_terminate_worker_process_impl,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        kill_grace_seconds=_KILL_GRACE_SECONDS,
    )


def _spawn_context() -> object:
    """Returns the spawn context used for faster-whisper process isolation."""
    return _spawn_context_boundary_impl(get_context=mp.get_context)


def _resolve_transcription_adapter(backend_id: TranscriptionBackendId) -> object:
    """Resolves one transcription adapter for worker execution."""
    return _resolve_transcription_adapter_boundary_impl(
        backend_id,
        adapter_resolver=resolve_transcription_backend_adapter,
    )


def _transcription_worker_entry(
    payload: object,
    connection: object,
) -> None:
    """Executes faster-whisper transcription inside one isolated worker process."""
    _transcription_worker_entry_boundary_impl(
        payload,
        connection,
        transcription_worker_entry_impl=_transcription_worker_entry_impl,
        adapter_resolver=_resolve_transcription_adapter,
    )


def extract_transcript(
    file_path: str,
    language: str | None = None,
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> list[TranscriptWord]:
    """Extracts a transcript with per-word timing for an input audio file.

    Args:
        file_path: Path to the audio file.
        language: Language code used by Whisper during transcription.
        profile: Optional runtime overrides for model and preprocessing toggles.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    active_settings = _resolve_boundary_settings(settings)
    active_language: str = language or active_settings.default_language
    return _extract_transcript(
        file_path,
        active_language,
        profile,
        settings=active_settings,
    )


def _extract_transcript(
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Internal transcript workflow with backend-specific execution strategy."""
    return _boundary_support.extract_transcript(
        file_path,
        language,
        profile,
        settings=settings,
        profile_factory=TranscriptionProfile,
        transcript_word_factory=TranscriptWord,
        logger=logger,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        error_factory=TranscriptionError,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        spawn_context_resolver=_spawn_context,
        worker_entry=_transcription_worker_entry,
        recv_worker_message_fn=_recv_worker_message,
        raise_worker_error_fn=_raise_worker_error,
        terminate_worker_process_fn=_terminate_worker_process,
        release_memory_fn=_release_transcription_runtime_memory,
        phase_started_fn=log_phase_started,
        phase_completed_fn=log_phase_completed,
        phase_failed_fn=log_phase_failed,
    )


def _extract_transcript_in_process(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs one in-process transcript workflow with phase-aware logging."""
    return _boundary_support.extract_transcript_in_process(
        file_path=file_path,
        language=language,
        profile=profile,
        settings=settings,
        profile_factory=TranscriptionProfile,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        error_factory=TranscriptionError,
        release_memory_fn=_release_transcription_runtime_memory,
        phase_started_fn=log_phase_started,
        phase_completed_fn=log_phase_completed,
        phase_failed_fn=log_phase_failed,
        logger=logger,
    )


def _release_transcription_runtime_memory(*, model: object | None) -> None:
    """Releases best-effort Torch runtime memory after one in-process transcript run."""
    _release_transcription_runtime_memory_impl(model=model, logger=logger)


def __transcribe_file(
    model: object,
    language: str,
    file_path: str,
    *,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call and normalizes return types."""
    return _transcribe_file_with_profile(
        model,
        language,
        file_path,
        profile=None,
        settings=settings,
    )


def _transcribe_file_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: TranscriptionProfile | None,
    *,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call using an explicit runtime profile."""
    return _boundary_support.transcribe_with_profile(
        model,
        language,
        file_path,
        profile,
        settings=settings,
        profile_factory=TranscriptionProfile,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        error_factory=TranscriptionError,
        passthrough_error_cls=TranscriptionError,
        logger=logger,
    )


def transcribe_with_model(
    model: object,
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> list[TranscriptWord]:
    """Transcribes one file with a pre-loaded model for profiling workloads."""
    return _transcribe_file_with_profile(
        model,
        language,
        file_path,
        profile=profile,
        settings=_resolve_boundary_settings(settings),
    )


def format_transcript(result: WhisperResult) -> list[TranscriptWord]:
    """Formats a Whisper result object into a word-level timestamp list.

    Args:
        result: Whisper transcription result.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    return _boundary_support.format_transcript(
        result,
        transcript_word_factory=TranscriptWord,
        logger=logger,
        error_factory=TranscriptionError,
    )
