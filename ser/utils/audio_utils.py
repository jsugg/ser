"""Audio loading utilities with fallback between librosa and soundfile."""

import logging
import time
import warnings
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from numpy import ndarray

from ser.config import Config
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def read_audio_file(file_path: str) -> tuple[ndarray, Any]:
    """Reads an audio file and normalizes amplitude to [-1, 1].

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple of `(audio_samples, sample_rate)`.
    """
    logger.debug(msg=f"Starting to read audio file: {file_path}")
    for attempt in range(Config.AUDIO_READ_CONFIG["max_retries"]):
        logger.debug(msg=f"Attempt {attempt + 1} to read audio file using librosa.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                audiofile, current_sample_rate = librosa.load(file_path, sr=None)
            audiofile = np.array(audiofile, dtype=np.float32)

            max_abs_value: Any = np.max(np.abs(audiofile))
            if max_abs_value == 0:
                audiofile = np.zeros_like(audiofile)
            else:
                audiofile /= max_abs_value

            logger.debug(msg=f"Successfully read audio file using librosa: {file_path}")
            return audiofile, current_sample_rate

        except Exception as e:
            logger.warning(msg=f"Librosa failed to read audio file: {e}")
            logger.warning(msg="Falling back to soundfile...")
            try:
                with sf.SoundFile(file_path) as sound_file:
                    audiofile = sound_file.read(dtype="float32")
                    current_sample_rate = sound_file.samplerate

                max_abs_value = np.max(np.abs(audiofile))
                if max_abs_value == 0:
                    audiofile = np.zeros_like(audiofile)
                else:
                    audiofile /= max_abs_value

                logger.debug(
                    msg=(f"Successfully read audio file using soundfile: {file_path}")
                )
                return audiofile, current_sample_rate

            except Exception as err:
                logger.warning(msg=f"Soundfile also failed: {err}")
                logger.info(
                    msg=(
                        "Retrying with librosa in "
                        f"{Config.AUDIO_READ_CONFIG['retry_delay']} seconds..."
                    )
                )
                time.sleep(Config.AUDIO_READ_CONFIG["retry_delay"])

    logger.error(
        msg=(
            f"Failed to read audio file {file_path} "
            f"after {Config.AUDIO_READ_CONFIG['max_retries']} retries."
        )
    )
    raise OSError(f"Error reading {file_path}")
