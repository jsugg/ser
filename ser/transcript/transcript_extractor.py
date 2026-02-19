"""Whisper-based transcript extraction with word-level timestamps."""

from __future__ import annotations

import logging
import os
import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, cast

from ser.config import get_settings
from ser.domain import TranscriptWord
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from stable_whisper.result import WhisperResult
    from whisper.model import Whisper

logger: logging.Logger = get_logger(__name__)


class HaloContext(Protocol):
    """Runtime protocol for the context manager returned by `halo.Halo`."""

    def __enter__(self) -> object: ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool | None: ...


class HaloFactory(Protocol):
    """Callable protocol for constructing spinner context managers."""

    def __call__(self, *args: object, **kwargs: object) -> HaloContext: ...


Halo: HaloFactory | None
Halo = None
try:
    from halo import Halo as _Halo
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI envs.
    pass
else:
    # `halo` does not provide precise typing for this constructor shape, while
    # this module intentionally treats it as a generic context-manager factory.
    Halo = cast(HaloFactory, _Halo)


class TranscriptionError(RuntimeError):
    """Raised when transcript extraction fails for operational reasons."""


class WhisperWord(Protocol):
    """Protocol for stable-whisper word-level transcript entries."""

    word: str
    start: float | None
    end: float | None


def load_whisper_model() -> Whisper:
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

    settings = get_settings()
    try:
        download_root = settings.models.whisper_download_root
        os.makedirs(download_root, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="stable_whisper")
            model: Whisper = stable_whisper.load_model(
                name=settings.models.whisper_model.name,
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
    file_path: str, language: str | None = None
) -> list[TranscriptWord]:
    """Extracts a transcript with per-word timing for an input audio file.

    Args:
        file_path: Path to the audio file.
        language: Language code used by Whisper during transcription.

    Returns:
        A list of transcript word entries with timing metadata.
    """
    active_language: str = language or get_settings().default_language
    return _extract_transcript(file_path, active_language)


def _extract_transcript(file_path: str, language: str) -> list[TranscriptWord]:
    """Internal transcript workflow with model loading and formatting."""
    if Halo is None:
        raise TranscriptionError(
            "Missing transcription dependency 'halo'. Ensure project dependencies are installed."
        )

    with Halo(
        text="Loading the Whisper model...",
        spinner="dots",
        text_color="green",
    ):
        model: Whisper = load_whisper_model()

    logger.info(msg="Whisper model loaded successfully.")

    with Halo(
        text="Transcribing the audio file...",
        spinner="dots",
        text_color="green",
    ):
        transcript: WhisperResult = __transcribe_file(model, language, file_path)
    logger.info(msg="Audio file transcription process completed.")

    formatted_transcript: list[TranscriptWord] = format_transcript(transcript)
    if not formatted_transcript:
        logger.info(msg="Transcript extraction succeeded but returned no words.")
    logger.debug(msg="Transcript output formatted successfully.")

    logger.info("Transcript extraction process completed successfully.")
    return formatted_transcript


def __transcribe_file(model: Whisper, language: str, file_path: str) -> WhisperResult:
    """Runs a Whisper transcription call and normalizes return types."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            raw_transcript = model.transcribe(
                audio=file_path,
                language=language,
                verbose=False,
                word_timestamps=True,
                no_speech_threshold=None,
                demucs=True,
                vad=True,
            )
    except Exception as err:
        logger.error(msg=f"Error processing speech extraction: {err}", exc_info=True)
        raise TranscriptionError("Failed to transcribe audio.") from err

    from stable_whisper.result import WhisperResult

    if isinstance(raw_transcript, WhisperResult):
        return raw_transcript

    if isinstance(raw_transcript, (dict, list, str)):
        return WhisperResult(raw_transcript)

    logger.error(
        "Unexpected transcription result type from stable-whisper: %s",
        type(raw_transcript).__name__,
    )
    raise TranscriptionError(
        "Unexpected transcription result type from stable-whisper."
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

    text_with_timestamps: list[TranscriptWord] = []
    for word in words:
        if word.start is None or word.end is None:
            continue
        text_with_timestamps.append(
            TranscriptWord(
                word=str(word.word),
                start_seconds=float(word.start),
                end_seconds=float(word.end),
            )
        )
    return text_with_timestamps
