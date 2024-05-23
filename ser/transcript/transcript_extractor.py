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
License: MIT
"""

import logging
from typing import Tuple, List, Any
import warnings

import stable_whisper
from halo import Halo
from whisper.model import Whisper

from ser.utils import get_logger
from ser.config import Config


logger: logging.Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings(
    "ignore",
    message=".*Cannot set number of intraop threads after parallel work has started.*",
)


def load_whisper_model() -> Whisper:
    """
    Loads the Whisper model specified in the configuration.

    Returns:
        stable_whisper.Whisper: Loaded Whisper model.
    """
    try:
        model: Whisper = stable_whisper.load_model(
            name=Config.MODELS_CONFIG["whisper_model"]["name"],
            device="cpu",
            dq=False,
            download_root=(
                f"{Config.MODELS_CONFIG['models_folder']}/"
                f"{Config.MODELS_CONFIG['whisper_model']['path']}"
            ),
            in_memory=True,
        )
        return model
    except Exception as e:
        logger.error(msg=f"Failed to load Whisper model: {e}", exc_info=True)
        raise


def extract_transcript(
    file_path: str, language: str = Config.DEFAULT_LANGUAGE
) -> List[Tuple[str, float, float]]:
    """
    Extracts the transcript from an audio file using the Whisper model.

    Arguments:
        file_path (str): Path to the audio file.
        language (str): Language of the audio.

    Returns:
        list: List of tuples (word, start_time, end_time).
    """
    try:
        with Halo(
            text="Loading the Whisper model...",
            spinner="dots",
            text_color="green",
        ):
            model: Whisper = load_whisper_model()
        logger.info(msg="Whisper model loaded successfully.")

        with Halo(
            text="Generating the transcript...",
            spinner="dots",
            text_color="green",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transcript: dict = model.transcribe(
                    audio=file_path,
                    language=language,
                    verbose=False,
                    word_timestamps=True,
                    no_speech_threshold=None,
                    demucs=True,
                    vad=True,
                )
            formatted_transcript: List[Tuple[str, float, float]] = (
                format_transcript(transcript)
            )

        logger.info("Transcript extraction completed successfully.")
        return formatted_transcript
    except Exception as e:
        logger.error(msg=f"Failed to extract transcript: {e}", exc_info=True)
        raise


def format_transcript(result) -> List[Tuple[str, float, float]]:
    """
    Formats the transcript into a list of tuples containing the word,
    start time, and end time.

    Args:
        result (dict): The transcript result.

    Returns:
        List[Tuple[str, float, float]]: Formatted transcript with timestamps.
    """
    words: Any = result.all_words()

    text_with_timestamps: List[Tuple[str, float, float]] = [
        (word.word, word.start, word.end) for word in words
    ]
    return text_with_timestamps
