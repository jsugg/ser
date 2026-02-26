"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

from ser.config import AppConfig, get_settings
from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
    PHASE_TRANSCRIPTION_SETUP,
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
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult

logger: logging.Logger = get_logger(__name__)


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


def _resolve_backend_id(raw_backend_id: object) -> TranscriptionBackendId:
    """Normalizes one backend id and validates supported values."""
    if raw_backend_id in {"stable_whisper", "faster_whisper"}:
        return cast(TranscriptionBackendId, raw_backend_id)
    raise TranscriptionError(
        "Unsupported transcription backend id configured. "
        "Expected 'stable_whisper' or 'faster_whisper'."
    )


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    if profile is not None:
        return profile
    settings: AppConfig = get_settings()
    return TranscriptionProfile(
        backend_id=_resolve_backend_id(settings.transcription.backend_id),
        model_name=settings.models.whisper_model.name,
        use_demucs=settings.transcription.use_demucs,
        use_vad=settings.transcription.use_vad,
    )


def _runtime_request_from_profile(
    active_profile: TranscriptionProfile,
) -> BackendRuntimeRequest:
    """Builds one backend runtime request from transcription profile settings."""
    return BackendRuntimeRequest(
        model_name=active_profile.model_name,
        use_demucs=active_profile.use_demucs,
        use_vad=active_profile.use_vad,
    )


def _check_adapter_compatibility(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> CompatibilityReport:
    """Validates backend compatibility and logs non-blocking compatibility issues."""
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    runtime_request = _runtime_request_from_profile(active_profile)
    report = adapter.check_compatibility(
        runtime_request=runtime_request,
        settings=settings,
    )
    if report.noise_issues:
        for issue in report.noise_issues:
            logger.debug(
                "Transcription backend '%s' noise issue [%s]: %s",
                active_profile.backend_id,
                issue.code,
                issue.message,
            )
    if report.operational_issues:
        for issue in report.operational_issues:
            logger.warning(
                "Transcription backend '%s' operational issue [%s]: %s",
                active_profile.backend_id,
                issue.code,
                issue.message,
            )
    if report.has_blocking_issues:
        details = (
            "; ".join(issue.message for issue in report.functional_issues)
            or "backend compatibility validation failed"
        )
        raise TranscriptionError(details)
    return report


def _transcription_setup_required(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> bool:
    """Returns whether a setup/download phase is needed before model load."""
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    return adapter.setup_required(
        runtime_request=_runtime_request_from_profile(active_profile),
        settings=settings,
    )


def _prepare_transcription_assets(
    *,
    active_profile: TranscriptionProfile,
    settings: AppConfig,
) -> None:
    """Ensures required stable-whisper model assets are present locally."""
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    adapter.prepare_assets(
        runtime_request=_runtime_request_from_profile(active_profile),
        settings=settings,
    )


def load_whisper_model(profile: TranscriptionProfile | None = None) -> object:
    """Loads the configured transcription model for CPU inference.

    Returns:
        The loaded Whisper model instance.
    """
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    try:
        _check_adapter_compatibility(
            active_profile=active_profile,
            settings=settings,
        )
        adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
        return adapter.load_model(
            runtime_request=_runtime_request_from_profile(active_profile),
            settings=settings,
        )
    except Exception as err:
        logger.error(msg=f"Failed to load transcription model: {err}", exc_info=True)
        raise TranscriptionError("Failed to load transcription model.") from err


def extract_transcript(
    file_path: str,
    language: str | None = None,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Extracts a transcript with per-word timing for an input audio file.

    Args:
        file_path: Path to the audio file.
        language: Language code used by Whisper during transcription.
        profile: Optional runtime overrides for model and preprocessing toggles.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    active_language: str = language or get_settings().default_language
    return _extract_transcript(file_path, active_language, profile)


def _extract_transcript(
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Internal transcript workflow with model loading and formatting."""
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)

    if _transcription_setup_required(
        active_profile=active_profile,
        settings=settings,
    ):
        setup_started_at = log_phase_started(
            logger,
            phase_name=PHASE_TRANSCRIPTION_SETUP,
        )
        try:
            _prepare_transcription_assets(
                active_profile=active_profile,
                settings=settings,
            )
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_TRANSCRIPTION_SETUP,
                started_at=setup_started_at,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=PHASE_TRANSCRIPTION_SETUP,
            started_at=setup_started_at,
        )

    model_load_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
    )
    try:
        model = load_whisper_model(active_profile)
    except Exception:
        log_phase_failed(
            logger,
            phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
            started_at=model_load_started_at,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
        started_at=model_load_started_at,
    )

    transcription_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION,
    )
    try:
        transcript_words: list[TranscriptWord] = (
            __transcribe_file(model, language, file_path)
            if profile is None
            else _transcribe_file_with_profile(
                model,
                language,
                file_path,
                active_profile,
            )
        )
    except Exception:
        log_phase_failed(
            logger,
            phase_name=PHASE_TRANSCRIPTION,
            started_at=transcription_started_at,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=PHASE_TRANSCRIPTION,
        started_at=transcription_started_at,
    )

    if not transcript_words:
        logger.info(msg="Transcript extraction succeeded but returned no words.")
    logger.debug(msg="Transcript output formatted successfully.")
    return transcript_words


def __transcribe_file(
    model: object, language: str, file_path: str
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call and normalizes return types."""
    return _transcribe_file_with_profile(model, language, file_path, profile=None)


def _transcribe_file_with_profile(
    model: object,
    language: str,
    file_path: str,
    profile: TranscriptionProfile | None,
) -> list[TranscriptWord]:
    """Runs a Whisper transcription call using an explicit runtime profile."""
    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    _check_adapter_compatibility(
        active_profile=active_profile,
        settings=settings,
    )
    adapter = resolve_transcription_backend_adapter(active_profile.backend_id)
    try:
        return adapter.transcribe(
            model=model,
            runtime_request=_runtime_request_from_profile(active_profile),
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
) -> list[TranscriptWord]:
    """Transcribes one file with a pre-loaded model for profiling workloads."""
    return _transcribe_file_with_profile(model, language, file_path, profile=profile)


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
