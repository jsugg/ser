"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import ModuleType
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from ser._internal.transcription.compatibility import (
    check_adapter_compatibility as _check_adapter_compatibility_impl,
)
from ser._internal.transcription.compatibility import (
    mark_compatibility_issues_as_emitted as _mark_compatibility_issues_as_emitted_impl,
)
from ser._internal.transcription.in_process_orchestration import (
    extract_transcript_in_process as _extract_transcript_in_process_impl,
)
from ser._internal.transcription.in_process_orchestration import (
    load_whisper_model as _load_whisper_model_impl,
)
from ser._internal.transcription.in_process_orchestration import (
    prepare_transcription_assets as _prepare_transcription_assets_impl,
)
from ser._internal.transcription.in_process_orchestration import (
    transcription_setup_required as _transcription_setup_required_impl,
)
from ser._internal.transcription.process_isolation import (
    raise_worker_error as _raise_worker_error_impl,
)
from ser._internal.transcription.process_isolation import (
    recv_worker_message as _recv_worker_message_impl,
)
from ser._internal.transcription.process_isolation import (
    run_faster_whisper_process_isolated as _run_faster_whisper_process_isolated_impl,
)
from ser._internal.transcription.process_isolation import (
    runtime_request_for_isolated_faster_whisper as _runtime_request_for_isolated_faster_whisper_impl,
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
from ser._internal.transcription.runtime_profile import (
    resolve_backend_id as _resolve_backend_id_impl,
)
from ser._internal.transcription.runtime_profile import (
    resolve_transcription_profile as _resolve_transcription_profile_impl,
)
from ser._internal.transcription.runtime_profile import (
    runtime_request_from_profile as _runtime_request_from_profile_impl,
)
from ser.config import AppConfig, get_settings
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
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

    backend_id: TranscriptionBackendId = "stable_whisper"
    model_name: str = "large-v2"
    use_demucs: bool = True
    use_vad: bool = True


@dataclass(frozen=True)
class _TranscriptionProcessPayload:
    """Serializable payload for one process-isolated transcription attempt."""

    file_path: str
    language: str
    profile: TranscriptionProfile
    runtime_request: BackendRuntimeRequest
    settings: "_TranscriptionWorkerSettings"


@dataclass(frozen=True, slots=True)
class _TranscriptionWorkerModelsConfig:
    """Serializable model settings required by process-isolated backends."""

    whisper_download_root: Path


@dataclass(frozen=True, slots=True)
class _TranscriptionWorkerSettings:
    """Serializable settings snapshot required by process-isolated workers."""

    models: _TranscriptionWorkerModelsConfig


@dataclass(frozen=True, slots=True)
class TranscriptionRuntimeOwnershipSlice:
    """Ownership record used to stage transcription-runtime extraction safely."""

    slice_id: str
    target_module: str
    symbols: tuple[str, ...]


type _WorkerPhase = Literal["setup_complete", "model_loaded"]
type _WorkerPhaseMessage = tuple[Literal["phase"], _WorkerPhase]
type _WorkerSuccessMessage = tuple[Literal["ok"], list[tuple[str, float, float]]]
type _WorkerErrorMessage = tuple[Literal["err"], str, str, str]
type _WorkerMessage = _WorkerPhaseMessage | _WorkerSuccessMessage | _WorkerErrorMessage
type _CompatibilityIssueKind = Literal["noise", "operational"]

_EMITTED_COMPATIBILITY_ISSUE_KEYS: set[tuple[str, str, str]] = set()
TRANSCRIPTION_RUNTIME_RISK_OWNERSHIP_MAP: tuple[
    TranscriptionRuntimeOwnershipSlice, ...
] = (
    TranscriptionRuntimeOwnershipSlice(
        slice_id="profile_runtime_resolution",
        target_module="ser._internal.transcription.runtime_profile",
        symbols=(
            "resolve_transcription_profile",
            "_runtime_request_from_profile",
        ),
    ),
    TranscriptionRuntimeOwnershipSlice(
        slice_id="compatibility_issue_hygiene",
        target_module="ser._internal.transcription.compatibility",
        symbols=(
            "_check_adapter_compatibility",
            "mark_compatibility_issues_as_emitted",
        ),
    ),
    TranscriptionRuntimeOwnershipSlice(
        slice_id="process_isolation_orchestration",
        target_module="ser._internal.transcription.process_isolation",
        symbols=(
            "_should_use_process_isolated_path",
            "_runtime_request_for_isolated_faster_whisper",
            "_run_faster_whisper_process_isolated",
            "_recv_worker_message",
            "_raise_worker_error",
            "_terminate_worker_process",
            "_transcription_worker_entry",
        ),
    ),
    TranscriptionRuntimeOwnershipSlice(
        slice_id="in_process_orchestration",
        target_module="ser._internal.transcription.in_process_orchestration",
        symbols=(
            "_transcription_setup_required",
            "_prepare_transcription_assets",
            "load_whisper_model",
            "_extract_transcript",
            "_extract_transcript_in_process",
            "_release_transcription_runtime_memory",
            "_transcribe_file_with_profile",
            "transcribe_with_model",
        ),
    ),
)


def transcription_runtime_risk_ownership_map() -> (
    tuple[TranscriptionRuntimeOwnershipSlice, ...]
):
    """Returns planned ownership slices for transcription-runtime risk containment."""
    return TRANSCRIPTION_RUNTIME_RISK_OWNERSHIP_MAP


def _resolve_transcription_profile_for_settings(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig,
) -> TranscriptionProfile:
    """Resolves one transcription profile against an explicit settings snapshot."""
    return cast(
        TranscriptionProfile,
        _resolve_transcription_profile_impl(
            profile,
            settings_resolver=lambda: settings,
            profile_factory=TranscriptionProfile,
            backend_id_resolver=lambda raw_backend_id: _resolve_backend_id_impl(
                raw_backend_id,
                error_factory=TranscriptionError,
            ),
        ),
    )


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    active_settings = settings if settings is not None else get_settings()
    return _resolve_transcription_profile_for_settings(
        profile,
        settings=active_settings,
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
    return _check_adapter_compatibility_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
        runtime_request_resolver=_runtime_request_from_profile,
        adapter_resolver=resolve_transcription_backend_adapter,
        error_factory=TranscriptionError,
        emitted_issue_keys=_EMITTED_COMPATIBILITY_ISSUE_KEYS,
        logger=logger,
    )


def mark_compatibility_issues_as_emitted(
    *,
    backend_id: TranscriptionBackendId,
    issue_kind: _CompatibilityIssueKind,
    issue_codes: tuple[str, ...],
) -> None:
    """Marks compatibility issues as already emitted to prevent duplicate logs."""
    _mark_compatibility_issues_as_emitted_impl(
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
    return _transcription_setup_required_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=lambda profile, app_settings: _runtime_request_from_profile(
            cast(TranscriptionProfile, profile),
            app_settings,
        ),
        compatibility_checker=_check_adapter_compatibility,
        adapter_resolver=lambda backend_id: cast(
            object,
            resolve_transcription_backend_adapter(backend_id),
        ),
    )


def _prepare_transcription_assets(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> None:
    """Ensures required stable-whisper model assets are present locally."""
    _prepare_transcription_assets_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=lambda profile, app_settings: _runtime_request_from_profile(
            cast(TranscriptionProfile, profile),
            app_settings,
        ),
        compatibility_checker=_check_adapter_compatibility,
        adapter_resolver=lambda backend_id: cast(
            object,
            resolve_transcription_backend_adapter(backend_id),
        ),
    )


def _load_whisper_model_for_settings(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig,
) -> object:
    """Loads one transcription model for an explicit settings snapshot."""
    return _load_whisper_model_impl(
        profile=profile,
        settings_resolver=lambda: settings,
        profile_resolver=lambda value: cast(
            object,
            _resolve_transcription_profile_for_settings(
                cast(TranscriptionProfile | None, value),
                settings=settings,
            ),
        ),
        runtime_request_resolver=lambda active_profile, app_settings: _runtime_request_from_profile(
            cast(TranscriptionProfile, active_profile),
            app_settings,
        ),
        compatibility_checker=_check_adapter_compatibility,
        adapter_resolver=lambda backend_id: cast(
            object,
            resolve_transcription_backend_adapter(backend_id),
        ),
        logger=logger,
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
    active_settings = settings if settings is not None else get_settings()
    return _load_whisper_model_for_settings(
        profile=profile,
        settings=active_settings,
    )


def _should_use_process_isolated_path(profile: TranscriptionProfile) -> bool:
    """Returns whether one transcription profile should execute in a worker process."""
    return _should_use_process_isolated_path_impl(profile)


def _runtime_request_for_isolated_faster_whisper(
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one faster-whisper runtime request without importing torch in worker."""
    return _runtime_request_for_isolated_faster_whisper_impl(
        profile=profile,
        settings=settings,
        error_factory=TranscriptionError,
        logger=logger,
    )


def _build_transcription_worker_settings(
    settings: AppConfig,
) -> _TranscriptionWorkerSettings:
    """Builds the serializable settings subset needed by worker processes."""
    return _TranscriptionWorkerSettings(
        models=_TranscriptionWorkerModelsConfig(
            whisper_download_root=settings.models.whisper_download_root,
        ),
    )


def _build_transcription_process_payload(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
    runtime_request: BackendRuntimeRequest,
    settings: AppConfig,
) -> _TranscriptionProcessPayload:
    """Builds a serializable worker payload from the active runtime settings."""
    return _TranscriptionProcessPayload(
        file_path=file_path,
        language=language,
        profile=profile,
        runtime_request=runtime_request,
        settings=_build_transcription_worker_settings(settings),
    )


def _run_faster_whisper_process_isolated(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs faster-whisper setup/load/transcribe inside one spawned worker process."""
    return _run_faster_whisper_process_isolated_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings_resolver=lambda: settings,
        runtime_request_resolver=_runtime_request_for_isolated_faster_whisper,
        payload_factory=_build_transcription_process_payload,
        get_spawn_context=_spawn_context,
        worker_entry=_transcription_worker_entry,
        recv_worker_message_fn=_recv_worker_message,
        raise_worker_error_fn=_raise_worker_error,
        terminate_worker_process_fn=_terminate_worker_process,
        transcript_word_factory=TranscriptWord,
        logger=logger,
        error_factory=TranscriptionError,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
    )


def _recv_worker_message(connection: Connection, *, stage: str) -> _WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_impl(
        connection,
        stage=stage,
        error_factory=TranscriptionError,
    )


def _raise_worker_error(message: _WorkerMessage) -> Never:
    """Raises one transcription-domain error from a worker payload."""
    _raise_worker_error_impl(message, error_factory=TranscriptionError)


def _terminate_worker_process(process: object) -> None:
    """Terminates a worker process with kill fallback."""
    _terminate_worker_process_impl(
        cast(BaseProcess, process),
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        kill_grace_seconds=_KILL_GRACE_SECONDS,
    )


def _spawn_context() -> object:
    """Returns the spawn context used for faster-whisper process isolation."""
    return mp.get_context("spawn")


def _resolve_transcription_adapter(backend_id: TranscriptionBackendId) -> object:
    """Resolves one transcription adapter for worker execution."""
    return cast(object, resolve_transcription_backend_adapter(backend_id))


def _transcription_worker_entry(
    payload: object,
    connection: object,
) -> None:
    """Executes faster-whisper transcription inside one isolated worker process."""
    active_payload = cast(_TranscriptionProcessPayload, payload)
    _transcription_worker_entry_impl(
        active_payload,
        cast(Connection, connection),
        settings_resolver=lambda: active_payload.settings,
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
    active_settings = settings if settings is not None else get_settings()
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
    active_profile = _resolve_transcription_profile_for_settings(
        profile,
        settings=settings,
    )
    if _should_use_process_isolated_path(active_profile):
        return _run_faster_whisper_process_isolated(
            file_path=file_path,
            language=language,
            profile=active_profile,
            settings=settings,
        )
    return _extract_transcript_in_process(
        file_path=file_path,
        language=language,
        profile=active_profile,
        settings=settings,
    )


def _extract_transcript_in_process(
    *,
    file_path: str,
    language: str,
    profile: TranscriptionProfile,
    settings: AppConfig,
) -> list[TranscriptWord]:
    """Runs one in-process transcript workflow with phase-aware logging."""
    return _extract_transcript_in_process_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings_resolver=lambda: settings,
        setup_required_checker=_transcription_setup_required,
        prepare_assets_runner=_prepare_transcription_assets,
        load_model_fn=lambda active_profile: load_whisper_model(
            cast(TranscriptionProfile | None, active_profile),
            settings=settings,
        ),
        transcribe_with_profile_fn=lambda model, lang, path, active_profile: (
            _transcribe_file_with_profile(
                model,
                lang,
                path,
                cast(TranscriptionProfile | None, active_profile),
                settings=settings,
            )
        ),
        release_memory_fn=_release_transcription_runtime_memory,
        phase_started_fn=log_phase_started,
        phase_completed_fn=log_phase_completed,
        phase_failed_fn=log_phase_failed,
        logger=logger,
    )


def _release_transcription_runtime_memory(*, model: object | None) -> None:
    """Releases best-effort Torch runtime memory after one in-process transcript run."""
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
    active_profile = _resolve_transcription_profile_for_settings(
        profile,
        settings=settings,
    )
    runtime_request = _runtime_request_from_profile(active_profile, settings)
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    try:
        return adapter.transcribe(
            model=model,
            runtime_request=runtime_request,
            file_path=file_path,
            language=language,
            settings=settings,
        )
    except TranscriptionError:
        raise
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err


def transcribe_with_model(
    model: object,
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> list[TranscriptWord]:
    """Transcribes one file with a pre-loaded model for profiling workloads."""
    active_settings = settings if settings is not None else get_settings()
    return _transcribe_file_with_profile(
        model,
        language,
        file_path,
        profile=profile,
        settings=active_settings,
    )


def format_transcript(result: WhisperResult) -> list[TranscriptWord]:
    """Formats a Whisper result object into a word-level timestamp list.

    Args:
        result: Whisper transcription result.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    try:
        words = cast(list[WhisperWord], result.all_words())
    except AttributeError as err:
        logger.error(msg=f"Error extracting words from result: {err}", exc_info=True)
        raise TranscriptionError("Invalid Whisper result object.") from err

    text_with_timestamps: list[TranscriptWord] = [
        TranscriptWord(
            word=str(word.word),
            start_seconds=float(word.start),
            end_seconds=float(word.end),
        )
        for word in words
        if word.start is not None and word.end is not None
    ]
    return text_with_timestamps
