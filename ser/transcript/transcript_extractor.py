"""
Transcript Extraction for Speech Emotion Recognition (SER) Tool

This module provides functions to extract transcripts from audio files using the Whisper
model. It includes functions to load the Whisper model and extract and format the transcript.

Functions:
    - load_whisper_model: Loads the Whisper model specified in the configuration.
    - extract_transcript: Extracts the transcript from an audio file using the Whisper model.
    - format_transcript: Formats the transcript into a list of tuples containing the word,
                         start time, and end time.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
Licenserr: MIT
"""

import logging
import os
import warnings
from typing import Any

import stable_whisper
from halo import Halo
from stable_whisper.result import WhisperResult
from whisper.model import Whisper

from ser.config import Config
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def load_whisper_model() -> Whisper:
    """
    Loads the Whisper model specified in the configuration.

    Returns:
        stable_whisper.Whisper: Loaded Whisper model.
    """
    try:
        download_root = (
            f"{Config.MODELS_CONFIG['models_folder']}/"
            f"{Config.MODELS_CONFIG['whisper_model']['path']}"
        )
        os.makedirs(download_root, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="stable_whisper")
            model: Whisper = stable_whisper.load_model(
                name=Config.MODELS_CONFIG["whisper_model"]["name"],
                device="cpu",
                dq=False,
                download_root=download_root,
                in_memory=True,
            )
        return model
    except Exception as err:
        logger.error(msg=f"Failed to load Whisper model: {err}", exc_info=True)
        raise


def extract_transcript(
    file_path: str, language: str = Config.DEFAULT_LANGUAGE
) -> list[tuple[str, float, float]]:
    """
    Extracts the transcript from an audio file using the Whisper model.

    Arguments:
        file_path (str): Path to the audio file.
        language (str): Language of the audio.

    Returns:
        list: List of tuples (word, start_time, end_time).
    """
    try:
        return _extract_transcript(file_path, language)
    except Exception as err:
        logger.error(msg=f"Failed to extract transcript: {err}", exc_info=True)
        raise


def _extract_transcript(
    file_path: str, language: str
) -> list[tuple[str, float, float]]:
    with Halo(
        text="Loading the Whisper model...",
        spinner="dots",
        text_color="green",
    ):
        try:
            model: Whisper = load_whisper_model()
        except Exception as err:
            logger.error(msg=f"Error loading Whisper model: {err}", exc_info=True)
            raise
    logger.info(msg="Whisper model loaded successfully.")
    try:
        with Halo(
            text="Transcribing the audio file...",
            spinner="dots",
            text_color="green",
        ):
            transcript: WhisperResult | None = __transcribe_file(
                model, language, file_path
            )
        logger.info(msg="Audio file transcription process completed.")

        if transcript:
            formatted_transcript: list[tuple[str, float, float]] = format_transcript(
                transcript
            )
        else:
            logger.info(msg="Transcript is empty.")
            return [("", 0, 0)]
        logger.debug(msg="Transcript output formatted successfully.")
    except Exception as err:
        logger.error(msg=f"Error generating the transcript: {err}", exc_info=True)
        raise err

    logger.info("Transcript extraction process completed successfully.")
    return formatted_transcript


def __transcribe_file(
    model: Whisper, language: str, file_path: str
) -> WhisperResult | None:
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
        return None
    if isinstance(raw_transcript, WhisperResult):
        return raw_transcript

    if isinstance(raw_transcript, (dict, list, str)):
        return WhisperResult(raw_transcript)

    logger.error(
        "Unexpected transcription result type from stable-whisper: %s",
        type(raw_transcript).__name__,
    )
    return None


def format_transcript(result: WhisperResult) -> list[tuple[str, float, float]]:
    """
    Formats the transcript into a list of tuples containing the word,
    start time, and end time.

    Args:
        result (dict): The transcript result.

    Returns:
        List[Tuple[str, float, float]]: Formatted transcript with timestamps.
    """
    try:
        words: list[Any] = result.all_words()
    except AttributeError as err:
        logger.error(msg=f"Error extracting words from result: {err}", exc_info=True)
        raise

    text_with_timestamps: list[tuple[str, float, float]] = (
        [(word.word, word.start, word.end) for word in words]
        if result.text != ""
        else [("", 0, 0)]
    )
    return text_with_timestamps
