"""Whisper-based transcript extraction with word-level timestamps."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from ser.config import AppConfig, get_settings
from ser.domain import TranscriptWord
from ser.runtime.phase_contract import (
    PHASE_TRANSCRIPTION,
    PHASE_TRANSCRIPTION_MODEL_LOAD,
)
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult
    from whisper.model import Whisper

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
    """Runtime Whisper profile settings used for transcription."""

    model_name: str
    use_demucs: bool
    use_vad: bool


def resolve_transcription_profile(
    profile: TranscriptionProfile | None = None,
) -> TranscriptionProfile:
    """Resolves profile overrides or falls back to configured defaults."""
    if profile is not None:
        return profile
    settings: AppConfig = get_settings()
    return TranscriptionProfile(
        model_name=settings.models.whisper_model.name,
        use_demucs=settings.transcription.use_demucs,
        use_vad=settings.transcription.use_vad,
    )


def load_whisper_model(profile: TranscriptionProfile | None = None) -> Whisper:
    """Loads the configured Whisper model for CPU inference.

    Returns:
        The loaded Whisper model instance.
    """
    try:
        import stable_whisper
    except ModuleNotFoundError as err:
        raise TranscriptionError(
            "Missing transcription dependencies. Ensure project dependencies are installed."
        ) from err

    settings: AppConfig = get_settings()
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    try:
        download_root: Path = settings.models.whisper_download_root
        torch_cache_root: Path = settings.models.torch_cache_root
        os.makedirs(download_root, exist_ok=True)
        os.makedirs(torch_cache_root, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_cache_root)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="stable_whisper")
            model: Whisper = stable_whisper.load_model(
                name=active_profile.model_name,
                device="cpu",
                dq=False,
                download_root=str(download_root),
                in_memory=True,
            )
        return model
    except Exception as err:
        logger.error(msg=f"Failed to load Whisper model: {err}", exc_info=True)
        raise TranscriptionError("Failed to load Whisper model.") from err


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
    model_load_started_at = log_phase_started(
        logger,
        phase_name=PHASE_TRANSCRIPTION_MODEL_LOAD,
    )
    try:
        model: Whisper = load_whisper_model(profile)
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
        transcript: WhisperResult = (
            __transcribe_file(model, language, file_path)
            if profile is None
            else _transcribe_file_with_profile(model, language, file_path, profile)
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

    formatted_transcript: list[TranscriptWord] = format_transcript(transcript)
    if not formatted_transcript:
        logger.info(msg="Transcript extraction succeeded but returned no words.")
    logger.debug(msg="Transcript output formatted successfully.")
    return formatted_transcript


def __transcribe_file(model: Whisper, language: str, file_path: str) -> WhisperResult:
    """Runs a Whisper transcription call and normalizes return types."""
    return _transcribe_file_with_profile(model, language, file_path, profile=None)


def _transcribe_file_with_profile(
    model: Whisper,
    language: str,
    file_path: str,
    profile: TranscriptionProfile | None,
) -> WhisperResult:
    """Runs a Whisper transcription call using an explicit runtime profile."""
    active_profile: TranscriptionProfile = resolve_transcription_profile(profile)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw_transcript = model.transcribe(
                audio=file_path,
                language=language,
                verbose=False,
                word_timestamps=True,
                no_speech_threshold=None,
                demucs=active_profile.use_demucs,
                vad=active_profile.use_vad,
            )
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err

    from stable_whisper.result import WhisperResult

    if isinstance(raw_transcript, WhisperResult):
        return raw_transcript

    if isinstance(raw_transcript, dict | list | str):
        return WhisperResult(raw_transcript)

    logger.error(
        "Unexpected transcription result type from stable-whisper: %s",
        type(raw_transcript).__name__,
    )
    raise TranscriptionError(
        "Unexpected transcription result type from stable-whisper."
    )


def transcribe_with_model(
    model: Whisper,
    file_path: str,
    language: str,
    profile: TranscriptionProfile | None = None,
) -> list[TranscriptWord]:
    """Transcribes one file with a pre-loaded model for profiling workloads."""
    result = _transcribe_file_with_profile(model, language, file_path, profile=profile)
    return format_transcript(result)


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
