"""Public-boundary runtime helpers for transcript extractor wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from ser.config import AppConfig
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest, CompatibilityReport

type _ResolveTranscriptionProfileImpl = Callable[..., object]
type _ResolveBackendId = Callable[..., TranscriptionBackendId]
type _RuntimeRequestResolver = Callable[..., BackendRuntimeRequest]
type _CompatibilityChecker = Callable[..., CompatibilityReport]
type _AdapterResolver = Callable[[TranscriptionBackendId], object]
type _ErrorFactory = Callable[[str], Exception]
type _PhaseLogger = Callable[..., float | None]
type _PhaseStarter = Callable[..., float]


def resolve_transcription_profile_from_public_boundary(
    profile: object | None = None,
    *,
    settings: AppConfig,
    resolve_transcription_profile_impl: _ResolveTranscriptionProfileImpl,
    profile_factory: Callable[..., object],
    backend_id_resolver: _ResolveBackendId,
    error_factory: _ErrorFactory,
) -> object:
    """Resolves one transcription profile using public-boundary settings injection."""
    return resolve_transcription_profile_impl(
        profile,
        settings_resolver=lambda: settings,
        profile_factory=profile_factory,
        backend_id_resolver=lambda raw_backend_id: backend_id_resolver(
            raw_backend_id,
            error_factory=error_factory,
        ),
    )


def check_adapter_compatibility_from_public_boundary(
    *,
    active_profile: object,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest | None = None,
    check_adapter_compatibility_impl: Callable[..., CompatibilityReport],
    runtime_request_resolver: _RuntimeRequestResolver,
    adapter_resolver: _AdapterResolver,
    emitted_issue_keys: set[tuple[str, str, str]] | None,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> CompatibilityReport:
    """Checks backend compatibility using the public boundary's contracts."""
    return check_adapter_compatibility_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
        runtime_request_resolver=runtime_request_resolver,
        adapter_resolver=adapter_resolver,
        error_factory=error_factory,
        emitted_issue_keys=emitted_issue_keys,
        logger=logger,
    )


def load_whisper_model_from_public_boundary(
    profile: object | None = None,
    *,
    settings: AppConfig,
    load_whisper_model_impl: Callable[..., object],
    resolve_profile_for_settings: Callable[..., object],
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: Callable[..., CompatibilityReport],
    adapter_resolver: _AdapterResolver,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> object:
    """Loads one transcription model using explicit public-boundary settings."""
    return load_whisper_model_impl(
        profile=profile,
        settings_resolver=lambda: settings,
        profile_resolver=lambda value: resolve_profile_for_settings(value, settings=settings),
        runtime_request_resolver=runtime_request_resolver,
        compatibility_checker=compatibility_checker,
        adapter_resolver=adapter_resolver,
        logger=logger,
        error_factory=error_factory,
    )


def transcription_setup_required_from_public_boundary(
    *,
    active_profile: object,
    settings: AppConfig,
    transcription_setup_required_impl: Callable[..., bool],
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: _CompatibilityChecker,
    adapter_resolver: _AdapterResolver,
) -> bool:
    """Checks whether transcription setup is required using boundary-owned wiring."""
    return transcription_setup_required_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=runtime_request_resolver,
        compatibility_checker=compatibility_checker,
        adapter_resolver=adapter_resolver,
    )


def prepare_transcription_assets_from_public_boundary(
    *,
    active_profile: object,
    settings: AppConfig,
    prepare_transcription_assets_impl: Callable[..., None],
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: _CompatibilityChecker,
    adapter_resolver: _AdapterResolver,
) -> None:
    """Runs asset preparation using boundary-owned runtime/setup dependencies."""
    prepare_transcription_assets_impl(
        active_profile=active_profile,
        settings=settings,
        runtime_request_resolver=runtime_request_resolver,
        compatibility_checker=compatibility_checker,
        adapter_resolver=adapter_resolver,
    )


def extract_transcript_in_process_from_public_boundary(
    *,
    file_path: str,
    language: str,
    profile: object,
    settings: AppConfig,
    extract_transcript_in_process_impl: Callable[..., list[TranscriptWord]],
    setup_required_checker: Callable[..., bool],
    prepare_assets_runner: Callable[..., None],
    load_whisper_model_fn: Callable[..., object],
    transcribe_with_profile_fn: Callable[..., list[TranscriptWord]],
    release_memory_fn: Callable[..., None],
    phase_started_fn: _PhaseStarter,
    phase_completed_fn: _PhaseLogger,
    phase_failed_fn: _PhaseLogger,
    logger: logging.Logger,
) -> list[TranscriptWord]:
    """Runs one in-process transcript workflow with explicit settings injection."""
    return extract_transcript_in_process_impl(
        file_path=file_path,
        language=language,
        profile=profile,
        settings_resolver=lambda: settings,
        setup_required_checker=setup_required_checker,
        prepare_assets_runner=prepare_assets_runner,
        load_model_fn=lambda active_profile: load_whisper_model_fn(
            active_profile,
            settings=settings,
        ),
        transcribe_with_profile_fn=lambda model, lang, path, active_profile: (
            transcribe_with_profile_fn(
                model,
                lang,
                path,
                active_profile,
                settings=settings,
            )
        ),
        release_memory_fn=release_memory_fn,
        phase_started_fn=phase_started_fn,
        phase_completed_fn=phase_completed_fn,
        phase_failed_fn=phase_failed_fn,
        logger=logger,
    )


def transcribe_with_profile_from_public_boundary(
    model: object,
    language: str,
    file_path: str,
    profile: object | None,
    *,
    settings: AppConfig,
    transcribe_with_profile_entrypoint: Callable[..., object],
    resolve_profile_for_settings: Callable[..., object],
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: Callable[..., CompatibilityReport],
    adapter_resolver: _AdapterResolver,
    passthrough_error_cls: type[Exception],
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> list[TranscriptWord]:
    """Runs one transcription call using the public boundary's runtime wiring."""
    return cast(
        list[TranscriptWord],
        transcribe_with_profile_entrypoint(
            model,
            language,
            file_path,
            profile,
            settings=settings,
            resolve_profile_fn=lambda value, *, settings: resolve_profile_for_settings(
                value,
                settings=settings,
            ),
            runtime_request_resolver=runtime_request_resolver,
            compatibility_checker=compatibility_checker,
            adapter_resolver=adapter_resolver,
            passthrough_error_cls=passthrough_error_cls,
            logger=logger,
            error_factory=error_factory,
        ),
    )


__all__ = [
    "check_adapter_compatibility_from_public_boundary",
    "extract_transcript_in_process_from_public_boundary",
    "load_whisper_model_from_public_boundary",
    "prepare_transcription_assets_from_public_boundary",
    "resolve_transcription_profile_from_public_boundary",
    "transcription_setup_required_from_public_boundary",
    "transcribe_with_profile_from_public_boundary",
]
