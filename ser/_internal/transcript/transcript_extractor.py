"""Backend-routed transcript extraction with word-level timestamps."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ser._internal.transcription import public_boundary_support as _boundary_support
from ser._internal.transcription.process_worker import (
    release_transcription_runtime_memory as _release_transcription_runtime_memory_impl,
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
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult

logger: logging.Logger = get_logger(__name__)
type _CompatibilityIssueKind = Literal["noise", "operational"]


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


def _resolve_catalog_transcription_defaults(
    profile: Literal["fast", "medium", "accurate", "accurate-research"],
) -> ProfileTranscriptionDefaults:
    """Returns catalog-owned transcription defaults for one runtime profile."""
    return get_profile_catalog()[profile].transcription_defaults


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or reloads a boundary-local settings snapshot."""
    return settings if settings is not None else reload_settings()


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    return cast(
        TranscriptionProfile,
        _boundary_support.resolve_transcription_profile_for_settings(
            profile,
            settings=_resolve_boundary_settings(settings),
            profile_factory=TranscriptionProfile,
            error_factory=TranscriptionError,
        ),
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
    )


def load_whisper_model(
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> object:
    """Loads the configured transcription model for resolved runtime settings."""
    return _boundary_support.load_whisper_model_for_settings(
        profile=profile,
        settings=_resolve_boundary_settings(settings),
        profile_factory=TranscriptionProfile,
        logger=logger,
        error_factory=TranscriptionError,
    )


def extract_transcript(
    file_path: str,
    language: str | None = None,
    profile: TranscriptionProfile | None = None,
    *,
    settings: AppConfig | None = None,
) -> list[TranscriptWord]:
    """Extracts a transcript with per-word timing for an input audio file."""
    active_settings = _resolve_boundary_settings(settings)
    active_language = language or active_settings.default_language
    return _boundary_support.extract_transcript(
        file_path,
        active_language,
        profile,
        settings=active_settings,
        profile_factory=TranscriptionProfile,
        logger=logger,
        error_factory=TranscriptionError,
        release_memory_fn=lambda *, model: _release_transcription_runtime_memory_impl(
            model=model,
            logger=logger,
        ),
        phase_started_fn=log_phase_started,
        phase_completed_fn=log_phase_completed,
        phase_failed_fn=log_phase_failed,
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
    return _boundary_support.transcribe_with_profile(
        model,
        language,
        file_path,
        profile,
        settings=_resolve_boundary_settings(settings),
        profile_factory=TranscriptionProfile,
        logger=logger,
        error_factory=TranscriptionError,
        passthrough_error_cls=TranscriptionError,
    )


def format_transcript(result: WhisperResult) -> list[TranscriptWord]:
    """Formats a Whisper result object into a word-level timestamp list."""
    return _boundary_support.format_transcript(
        result,
        transcript_word_factory=TranscriptWord,
        logger=logger,
        error_factory=TranscriptionError,
    )


__all__ = [
    "TranscriptionError",
    "TranscriptionProfile",
    "WhisperWord",
    "extract_transcript",
    "format_transcript",
    "load_whisper_model",
    "mark_compatibility_issues_as_emitted",
    "resolve_transcription_profile",
    "transcribe_with_model",
]
