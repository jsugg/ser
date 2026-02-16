"""Audio loading utilities with fallback between librosa and soundfile."""

import logging
import time
import warnings

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ser.config import get_settings
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def _normalize_audio(audiofile: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalizes an audio buffer to the range [-1, 1]."""
    if audiofile.size == 0:
        return audiofile
    max_abs_value = float(np.max(np.abs(audiofile)))
    if max_abs_value == 0:
        return np.zeros_like(audiofile)
    return audiofile / max_abs_value


def read_audio_file(file_path: str) -> tuple[NDArray[np.float32], int]:
    """Reads an audio file and normalizes amplitude to [-1, 1].

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple of `(audio_samples, sample_rate)`.
    """
    settings = get_settings()
    logger.debug(msg=f"Starting to read audio file: {file_path}")
    for attempt in range(settings.audio_read.max_retries):
        logger.debug(msg=f"Attempt {attempt + 1} to read audio file using librosa.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                audiofile, current_sample_rate = librosa.load(file_path, sr=None)
            audiofile = np.asarray(audiofile, dtype=np.float32)
            audiofile = _normalize_audio(audiofile)

            logger.debug(msg=f"Successfully read audio file using librosa: {file_path}")
            return audiofile, int(current_sample_rate)

        except Exception as err:
            logger.warning(msg=f"Librosa failed to read audio file: {err}")
            logger.warning(msg="Falling back to soundfile...")
            try:
                with sf.SoundFile(file_path) as sound_file:
                    audiofile = np.asarray(
                        sound_file.read(dtype="float32"), dtype=np.float32
                    )
                    current_sample_rate = int(sound_file.samplerate)
                audiofile = _normalize_audio(audiofile)

                logger.debug(
                    msg=(f"Successfully read audio file using soundfile: {file_path}")
                )
                return audiofile, current_sample_rate

            except Exception as err:
                logger.warning(msg=f"Soundfile also failed: {err}")
                logger.info(
                    msg=(
                        "Retrying with librosa in "
                        f"{settings.audio_read.retry_delay_seconds} seconds..."
                    )
                )
                time.sleep(settings.audio_read.retry_delay_seconds)

    logger.error(
        msg=(
            f"Failed to read audio file {file_path} "
            f"after {settings.audio_read.max_retries} retries."
        )
    )
    raise OSError(f"Error reading {file_path}")
