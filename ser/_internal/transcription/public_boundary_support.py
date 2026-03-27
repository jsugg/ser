"""Internal support owners for transcript extractor public-boundary wrappers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, Never, Protocol, cast

from ser._internal.transcription.compatibility import (
    check_adapter_compatibility as _check_adapter_compatibility_impl,
)
from ser._internal.transcription.compatibility import (
    mark_compatibility_issues_as_emitted as _mark_compatibility_issues_as_emitted_impl,
)
from ser._internal.transcription.extractor_entrypoints import (
    extract_transcript as _extract_transcript_entrypoint,
)
from ser._internal.transcription.extractor_entrypoints import (
    format_transcript as _format_transcript_entrypoint,
)
from ser._internal.transcription.extractor_entrypoints import (
    transcribe_with_profile as _transcribe_with_profile_entrypoint,
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
from ser._internal.transcription.process_isolation import WorkerMessage as _WorkerMessage
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
from ser._internal.transcription.process_worker import (
    build_transcription_process_payload as _build_transcription_process_payload,
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
    run_faster_whisper_process_isolated_from_public_boundary as _run_faster_whisper_process_isolated_boundary_impl,
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
from ser._internal.transcription.public_boundary_runtime import (
    check_adapter_compatibility_from_public_boundary as _check_adapter_compatibility_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    extract_transcript_in_process_from_public_boundary as _extract_transcript_in_process_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    load_whisper_model_from_public_boundary as _load_whisper_model_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    prepare_transcription_assets_from_public_boundary as _prepare_transcription_assets_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    resolve_transcription_profile_from_public_boundary as _resolve_transcription_profile_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    transcribe_with_profile_from_public_boundary as _transcribe_with_profile_boundary_impl,
)
from ser._internal.transcription.public_boundary_runtime import (
    transcription_setup_required_from_public_boundary as _transcription_setup_required_boundary_impl,
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
from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.transcript.backends.factory import resolve_transcription_backend_adapter
from ser.transcript.runtime_policy import (
    DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    resolve_transcription_runtime_policy,
)

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest, CompatibilityReport

_CompatibilityIssueKind = Literal["noise", "operational"]


class _BackendProfile(Protocol):
    """Minimal transcription profile contract."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns the configured backend identifier."""
        ...

    @property
    def model_name(self) -> str:
        """Returns the backend-specific model identifier."""
        ...

    @property
    def use_demucs(self) -> bool:
        """Returns whether Demucs preprocessing is enabled."""
        ...

    @property
    def use_vad(self) -> bool:
        """Returns whether voice activity detection is enabled."""
        ...


type _ProfileFactory = Callable[..., _BackendProfile]
type _TranscriptWordFactory = Callable[[str, float, float], TranscriptWord]
type _ErrorFactory = type[Exception]
type _ResolveSpawnContext = Callable[[], object]
type _WorkerMessageReceiver = Callable[..., _WorkerMessage]
type _WorkerErrorRaiser = Callable[[object], Never]
type _WorkerTerminator = Callable[[object], None]
type _WorkerEntry = Callable[[object, object], None]

_TERMINATE_GRACE_SECONDS = 5.0
_KILL_GRACE_SECONDS = 2.0


def _runtime_request_from_profile(
    active_profile: _BackendProfile,
    settings: AppConfig,
) -> BackendRuntimeRequest:
    """Builds one backend runtime request from transcription profile settings."""
    return _runtime_request_from_profile_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_policy_resolver=resolve_transcription_runtime_policy,
        default_mps_low_memory_threshold_gb=DEFAULT_MPS_LOW_MEMORY_THRESHOLD_GB,
    )


def resolve_transcription_profile_for_settings(
    profile: _BackendProfile | None = None,
    *,
    settings: AppConfig,
    profile_factory: _ProfileFactory,
    error_factory: _ErrorFactory,
) -> _BackendProfile:
    """Resolves one transcription profile against an explicit settings snapshot."""
    return cast(
        _BackendProfile,
        _resolve_transcription_profile_boundary_impl(
            profile,
            settings=settings,
            resolve_transcription_profile_impl=_resolve_transcription_profile_impl,
            profile_factory=profile_factory,
            backend_id_resolver=_resolve_backend_id_impl,
            error_factory=error_factory,
        ),
    )


def check_adapter_compatibility(
    *,
    active_profile: _BackendProfile,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest | None = None,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> CompatibilityReport:
    """Validates backend compatibility for one resolved transcription profile."""
    return _check_adapter_compatibility_boundary_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
        check_adapter_compatibility_impl=_check_adapter_compatibility_impl,
        runtime_request_resolver=_runtime_request_from_profile,
        adapter_resolver=resolve_transcription_backend_adapter,
        emitted_issue_keys=emitted_issue_keys,
        logger=logger,
        error_factory=error_factory,
    )


def mark_compatibility_issues_as_emitted(
    *,
    backend_id: TranscriptionBackendId,
    issue_kind: _CompatibilityIssueKind,
    issue_codes: tuple[str, ...],
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
) -> None:
    """Marks compatibility issues as emitted so they are logged only once."""
    _mark_compatibility_issues_as_emitted_impl(
        backend_id=backend_id,
        issue_kind=issue_kind,
        issue_codes=issue_codes,
        emitted_issue_keys=emitted_issue_keys,
    )


def transcription_setup_required(
    *,
    active_profile: _BackendProfile,
    settings: AppConfig,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> bool:
    """Returns whether model setup/download work is required before load."""
    return _transcription_setup_required_boundary_impl(
        active_profile=active_profile,
        settings=settings,
        transcription_setup_required_impl=_transcription_setup_required_impl,
        runtime_request_resolver=_runtime_request_from_profile,
        compatibility_checker=lambda **kwargs: check_adapter_compatibility(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        adapter_resolver=resolve_transcription_backend_adapter,
    )


def prepare_transcription_assets(
    *,
    active_profile: _BackendProfile,
    settings: AppConfig,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> None:
    """Ensures required transcription assets are available before model load."""
    _prepare_transcription_assets_boundary_impl(
        active_profile=active_profile,
        settings=settings,
        prepare_transcription_assets_impl=_prepare_transcription_assets_impl,
        runtime_request_resolver=_runtime_request_from_profile,
        compatibility_checker=lambda **kwargs: check_adapter_compatibility(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        adapter_resolver=resolve_transcription_backend_adapter,
    )


def load_whisper_model_for_settings(
    profile: _BackendProfile | None = None,
    *,
    settings: AppConfig,
    profile_factory: _ProfileFactory,
    logger: logging.Logger,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    error_factory: _ErrorFactory,
) -> object:
    """Loads one transcription model using an explicit settings snapshot."""
    return _load_whisper_model_boundary_impl(
        profile=profile,
        settings=settings,
        load_whisper_model_impl=_load_whisper_model_impl,
        resolve_profile_for_settings=lambda value, *, settings: (
            resolve_transcription_profile_for_settings(
                value,
                settings=settings,
                profile_factory=profile_factory,
                error_factory=error_factory,
            )
        ),
        runtime_request_resolver=_runtime_request_from_profile,
        compatibility_checker=lambda **kwargs: check_adapter_compatibility(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        adapter_resolver=resolve_transcription_backend_adapter,
        logger=logger,
        error_factory=error_factory,
    )


def _runtime_request_for_isolated_faster_whisper(
    profile: _BackendProfile,
    settings: AppConfig,
    *,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> BackendRuntimeRequest:
    """Builds one faster-whisper worker runtime request without torch imports."""
    return _runtime_request_for_isolated_faster_whisper_impl(
        profile=profile,
        settings=settings,
        error_factory=error_factory,
        logger=logger,
    )


def _spawn_context_for_public_boundary() -> object:
    """Returns the multiprocessing spawn context used by the public boundary."""
    return _spawn_context_boundary_impl(get_context=mp.get_context)


def _resolve_transcription_adapter_for_public_boundary(
    backend_id: TranscriptionBackendId,
) -> object:
    """Resolves one backend adapter for process-isolated public-boundary workers."""
    return _resolve_transcription_adapter_boundary_impl(
        backend_id,
        adapter_resolver=resolve_transcription_backend_adapter,
    )


def _transcription_worker_entry_for_public_boundary(
    payload: object,
    connection: object,
) -> None:
    """Runs one process-isolated transcription worker with module-level collaborators."""
    _transcription_worker_entry_boundary_impl(
        payload,
        connection,
        transcription_worker_entry_impl=_transcription_worker_entry_impl,
        adapter_resolver=_resolve_transcription_adapter_for_public_boundary,
    )


def run_faster_whisper_process_isolated(
    *,
    file_path: str,
    language: str,
    profile: _BackendProfile,
    settings: AppConfig,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> list[TranscriptWord]:
    """Runs faster-whisper transcription inside a spawned worker process."""

    def _recv_worker_message(connection: object, *, stage: str) -> _WorkerMessage:
        return _recv_worker_message_boundary_impl(
            connection,
            recv_worker_message_impl=_recv_worker_message_impl,
            stage=stage,
            error_factory=error_factory,
        )

    def _raise_worker_error(message: object) -> Never:
        _raise_worker_error_boundary_impl(
            cast(_WorkerMessage, message),
            raise_worker_error_impl=_raise_worker_error_impl,
            error_factory=error_factory,
        )

    def _terminate_worker_process(process: object) -> None:
        _terminate_worker_process_boundary_impl(
            process,
            terminate_worker_process_impl=_terminate_worker_process_impl,
            terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
            kill_grace_seconds=_KILL_GRACE_SECONDS,
        )

    return _run_faster_whisper_process_isolated_boundary_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings=settings,
        run_faster_whisper_process_isolated_impl=_run_faster_whisper_process_isolated_impl,
        runtime_request_resolver=lambda active_profile, active_settings: (
            _runtime_request_for_isolated_faster_whisper(
                active_profile,
                active_settings,
                logger=logger,
                error_factory=error_factory,
            )
        ),
        payload_factory=_build_transcription_process_payload,
        spawn_context_resolver=_spawn_context_for_public_boundary,
        worker_entry=_transcription_worker_entry_for_public_boundary,
        recv_worker_message_fn=_recv_worker_message,
        raise_worker_error_fn=_raise_worker_error,
        terminate_worker_process_fn=_terminate_worker_process,
        logger=logger,
        error_factory=error_factory,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
    )


def extract_transcript_in_process(
    *,
    file_path: str,
    language: str,
    profile: _BackendProfile,
    settings: AppConfig,
    profile_factory: _ProfileFactory,
    logger: logging.Logger,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    error_factory: _ErrorFactory,
    release_memory_fn: Callable[..., None],
    phase_started_fn: Callable[..., float],
    phase_completed_fn: Callable[..., float | None],
    phase_failed_fn: Callable[..., float | None],
) -> list[TranscriptWord]:
    """Runs one in-process transcript workflow with phase-aware logging."""
    return _extract_transcript_in_process_boundary_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings=settings,
        extract_transcript_in_process_impl=_extract_transcript_in_process_impl,
        setup_required_checker=lambda **kwargs: transcription_setup_required(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        prepare_assets_runner=lambda **kwargs: prepare_transcription_assets(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        load_whisper_model_fn=lambda active_profile, *, settings: (
            load_whisper_model_for_settings(
                active_profile,
                settings=settings,
                profile_factory=profile_factory,
                logger=logger,
                emitted_issue_keys=emitted_issue_keys,
                error_factory=error_factory,
            )
        ),
        transcribe_with_profile_fn=lambda model, lang, path, active_profile, *, settings: (
            transcribe_with_profile(
                model,
                lang,
                path,
                active_profile,
                settings=settings,
                profile_factory=profile_factory,
                logger=logger,
                emitted_issue_keys=emitted_issue_keys,
                error_factory=error_factory,
                passthrough_error_cls=error_factory,
            )
        ),
        release_memory_fn=release_memory_fn,
        phase_started_fn=phase_started_fn,
        phase_completed_fn=phase_completed_fn,
        phase_failed_fn=phase_failed_fn,
        logger=logger,
    )


def transcribe_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: _BackendProfile | None,
    *,
    settings: AppConfig,
    profile_factory: _ProfileFactory,
    logger: logging.Logger,
    emitted_issue_keys: set[tuple[str, str, str]] | None = None,
    error_factory: _ErrorFactory,
    passthrough_error_cls: type[Exception],
) -> list[TranscriptWord]:
    """Runs a transcription call using one resolved runtime profile."""
    return _transcribe_with_profile_boundary_impl(
        model,
        language,
        file_path,
        profile,
        settings=settings,
        transcribe_with_profile_entrypoint=_transcribe_with_profile_entrypoint,
        resolve_profile_for_settings=lambda value, *, settings: (
            resolve_transcription_profile_for_settings(
                value,
                settings=settings,
                profile_factory=profile_factory,
                error_factory=error_factory,
            )
        ),
        runtime_request_resolver=_runtime_request_from_profile,
        compatibility_checker=lambda **kwargs: check_adapter_compatibility(
            emitted_issue_keys=emitted_issue_keys,
            logger=logger,
            error_factory=error_factory,
            **kwargs,
        ),
        adapter_resolver=resolve_transcription_backend_adapter,
        passthrough_error_cls=passthrough_error_cls,
        logger=logger,
        error_factory=error_factory,
    )


def extract_transcript(
    file_path: str,
    language: str,
    profile: _BackendProfile | None = None,
    *,
    settings: AppConfig,
    profile_factory: _ProfileFactory,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
    release_memory_fn: Callable[..., None],
    phase_started_fn: Callable[..., float],
    phase_completed_fn: Callable[..., float | None],
    phase_failed_fn: Callable[..., float | None],
) -> list[TranscriptWord]:
    """Routes transcript extraction across in-process and isolated execution."""
    return cast(
        list[TranscriptWord],
        _extract_transcript_entrypoint(
            file_path,
            language,
            profile,
            settings=settings,
            resolve_profile_fn=lambda value, *, settings: (
                resolve_transcription_profile_for_settings(
                    value,
                    settings=settings,
                    profile_factory=profile_factory,
                    error_factory=error_factory,
                )
            ),
            should_use_process_isolated_path_fn=lambda active_profile: (
                _should_use_process_isolated_path_impl(active_profile)
            ),
            run_process_isolated_fn=lambda **kwargs: run_faster_whisper_process_isolated(
                logger=logger,
                error_factory=error_factory,
                **kwargs,
            ),
            run_in_process_fn=lambda **kwargs: extract_transcript_in_process(
                profile_factory=profile_factory,
                logger=logger,
                error_factory=error_factory,
                release_memory_fn=release_memory_fn,
                phase_started_fn=phase_started_fn,
                phase_completed_fn=phase_completed_fn,
                phase_failed_fn=phase_failed_fn,
                **kwargs,
            ),
        ),
    )


def format_transcript(
    result: object,
    *,
    transcript_word_factory: _TranscriptWordFactory,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> list[TranscriptWord]:
    """Formats one Whisper result object into transcript-word domain objects."""
    return cast(
        list[TranscriptWord],
        _format_transcript_entrypoint(
            result,
            transcript_word_factory=transcript_word_factory,
            logger=logger,
            error_factory=error_factory,
        ),
    )


__all__ = [
    "_runtime_request_for_isolated_faster_whisper",
    "check_adapter_compatibility",
    "extract_transcript",
    "extract_transcript_in_process",
    "format_transcript",
    "load_whisper_model_for_settings",
    "mark_compatibility_issues_as_emitted",
    "prepare_transcription_assets",
    "resolve_transcription_profile_for_settings",
    "run_faster_whisper_process_isolated",
    "transcribe_with_profile",
    "transcription_setup_required",
]
