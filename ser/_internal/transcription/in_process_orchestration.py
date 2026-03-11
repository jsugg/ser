"""In-process transcription orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol, TypeVar, cast

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
    PHASE_TRANSCRIPTION_SETUP,
)
from ser.transcript.backends import BackendRuntimeRequest


class _SetupLoadAdapter(Protocol):
    """Minimal adapter contract for setup/load orchestration helpers."""

    def setup_required(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> bool:
        """Returns whether setup/download is required before model loading."""
        ...

    def prepare_assets(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> None:
        """Ensures required model assets exist."""
        ...

    def load_model(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> object:
        """Loads and returns backend model handle."""
        ...


class _BackendProfile(Protocol):
    """Minimal profile contract required by in-process orchestration helpers."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns backend identifier used for adapter resolution."""
        ...


type _RuntimeRequestResolver = Callable[[object, AppConfig], BackendRuntimeRequest]
type _CompatibilityChecker = Callable[..., object]
type _AdapterResolver = Callable[[TranscriptionBackendId], object]
type _SettingsResolver = Callable[[], AppConfig]
type _ProfileResolver = Callable[[object | None], object]
type _ErrorFactory = Callable[[str], Exception]

_TTranscriptWord = TypeVar("_TTranscriptWord")


def transcription_setup_required(
    *,
    active_profile: object,
    settings: AppConfig,
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: _CompatibilityChecker,
    adapter_resolver: _AdapterResolver,
) -> bool:
    """Returns whether a setup/download phase is needed before model load."""
    runtime_request = runtime_request_resolver(active_profile, settings)
    compatibility_checker(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    backend_id = cast(_BackendProfile, active_profile).backend_id
    adapter = cast(_SetupLoadAdapter, adapter_resolver(backend_id))
    return adapter.setup_required(
        runtime_request=runtime_request,
        settings=settings,
    )


def prepare_transcription_assets(
    *,
    active_profile: object,
    settings: AppConfig,
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: _CompatibilityChecker,
    adapter_resolver: _AdapterResolver,
) -> None:
    """Ensures required transcription model assets are present locally."""
    runtime_request = runtime_request_resolver(active_profile, settings)
    compatibility_checker(
        active_profile=active_profile,
        settings=settings,
        runtime_request=runtime_request,
    )
    backend_id = cast(_BackendProfile, active_profile).backend_id
    adapter = cast(_SetupLoadAdapter, adapter_resolver(backend_id))
    adapter.prepare_assets(
        runtime_request=runtime_request,
        settings=settings,
    )


def load_whisper_model(
    profile: object | None = None,
    *,
    settings_resolver: _SettingsResolver,
    profile_resolver: _ProfileResolver,
    runtime_request_resolver: _RuntimeRequestResolver,
    compatibility_checker: _CompatibilityChecker,
    adapter_resolver: _AdapterResolver,
    logger: logging.Logger,
    error_factory: _ErrorFactory,
) -> object:
    """Loads one transcription model handle for resolved runtime settings."""
    settings = settings_resolver()
    active_profile = profile_resolver(profile)
    runtime_request = runtime_request_resolver(active_profile, settings)
    try:
        compatibility_checker(
            active_profile=active_profile,
            settings=settings,
            runtime_request=runtime_request,
        )
        backend_id = cast(_BackendProfile, active_profile).backend_id
        logger.info(
            "Transcription runtime resolved (backend=%s, device=%s, precision=%s, memory_tier=%s).",
            backend_id,
            runtime_request.device_spec,
            ",".join(runtime_request.precision_candidates),
            runtime_request.memory_tier,
        )
        adapter = cast(_SetupLoadAdapter, adapter_resolver(backend_id))
        return adapter.load_model(
            runtime_request=runtime_request,
            settings=settings,
        )
    except Exception as err:
        logger.error(msg=f"Failed to load transcription model: {err}", exc_info=True)
        raise error_factory("Failed to load transcription model.") from err


def extract_transcript_in_process(
    *,
    file_path: str,
    language: str,
    profile: object,
    settings_resolver: _SettingsResolver,
    setup_required_checker: Callable[..., bool],
    prepare_assets_runner: Callable[..., None],
    load_model_fn: Callable[..., object],
    transcribe_with_profile_fn: Callable[..., list[_TTranscriptWord]],
    release_memory_fn: Callable[..., None],
    phase_started_fn: Callable[..., float],
    phase_completed_fn: Callable[..., float | None],
    phase_failed_fn: Callable[..., float | None],
    logger: logging.Logger,
) -> list[_TTranscriptWord]:
    """Runs one in-process transcript workflow with phase-aware logging."""
    settings = settings_resolver()
    active_profile = profile
    model: object | None = None

    if setup_required_checker(
        active_profile=active_profile,
        settings=settings,
    ):
        setup_started_at = phase_started_fn(logger, phase_name=PHASE_TRANSCRIPTION_SETUP)
        try:
            prepare_assets_runner(
                active_profile=active_profile,
                settings=settings,
            )
        except Exception:
            phase_failed_fn(
                logger,
                phase_name=PHASE_TRANSCRIPTION_SETUP,
                started_at=setup_started_at,
            )
            raise
        phase_completed_fn(
            logger,
            phase_name=PHASE_TRANSCRIPTION_SETUP,
            started_at=setup_started_at,
        )

    model_load_started_at = phase_started_fn(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
    )
    try:
        model = load_model_fn(active_profile)
    except Exception:
        phase_failed_fn(
            logger,
            phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
            started_at=model_load_started_at,
        )
        raise
    phase_completed_fn(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
        started_at=model_load_started_at,
    )

    transcription_started_at = phase_started_fn(
        logger,
        phase_name=PHASE_TRANSCRIPTION,
    )
    try:
        try:
            transcript_words = transcribe_with_profile_fn(
                model,
                language,
                file_path,
                active_profile,
            )
        except Exception:
            phase_failed_fn(
                logger,
                phase_name=PHASE_TRANSCRIPTION,
                started_at=transcription_started_at,
            )
            raise
        phase_completed_fn(
            logger,
            phase_name=PHASE_TRANSCRIPTION,
            started_at=transcription_started_at,
        )
        if not transcript_words:
            logger.info(msg="Transcript extraction succeeded but returned no words.")
        logger.debug(msg="Transcript output formatted successfully.")
        return transcript_words
    finally:
        release_memory_fn(model=model)


__all__ = [
    "extract_transcript_in_process",
    "load_whisper_model",
    "prepare_transcription_assets",
    "transcription_setup_required",
]
