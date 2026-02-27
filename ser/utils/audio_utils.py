"""Audio loading utilities with fallback between librosa and soundfile."""

import logging
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ser.config import AppConfig, get_settings
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


def _to_mono(audiofile: NDArray[np.float32]) -> NDArray[np.float32]:
    """Converts multi-channel audio to mono while preserving sample order."""
    if audiofile.ndim == 1:
        return audiofile

    if audiofile.ndim == 2:
        # `soundfile` returns shape `(frames, channels)`.
        if audiofile.shape[1] == 0:
            return np.array([], dtype=np.float32)
        mixed = np.mean(audiofile, axis=1, dtype=np.float32)
        return np.asarray(mixed, dtype=np.float32)

    raise OSError(f"Unsupported audio shape: {audiofile.shape}")


def _prepare_audio_buffer(raw_audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalizes and validates decoded audio samples for downstream DSP."""
    prepared = np.asarray(raw_audio, dtype=np.float32)
    prepared = np.nan_to_num(prepared, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    prepared = _to_mono(prepared)
    if prepared.size == 0:
        raise OSError("Audio file contains no samples.")
    return _normalize_audio(prepared)


def read_audio_file(
    file_path: str,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> tuple[NDArray[np.float32], int]:
    """Reads an audio file (or segment) and normalizes amplitude to [-1, 1].

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple of `(audio_samples, sample_rate)`.
    """
    if start_seconds is not None and start_seconds < 0.0:
        raise ValueError("start_seconds must be >= 0")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be > 0")

    settings: AppConfig = get_settings()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    if not path.is_file():
        raise OSError(f"Path is not a regular file: {file_path}")

    logger.debug(msg=f"Starting to read audio file: {file_path}")
    for attempt in range(settings.audio_read.max_retries):
        logger.debug(msg=f"Attempt {attempt + 1} to read audio file using librosa.")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="librosa"
                )
                audiofile, current_sample_rate = librosa.load(
                    str(path),
                    sr=None,
                    offset=float(start_seconds or 0.0),
                    duration=(
                        float(duration_seconds)
                        if duration_seconds is not None
                        else None
                    ),
                )
            normalized_audio = _prepare_audio_buffer(
                np.asarray(audiofile, dtype=np.float32)
            )

            logger.debug(msg=f"Successfully read audio file using librosa: {file_path}")
            return normalized_audio, int(current_sample_rate)

        except Exception as err:
            logger.warning(msg=f"Librosa failed to read audio file: {err}")

            # Segment reads rely on librosa offset/duration.
            if start_seconds is not None or duration_seconds is not None:
                if attempt < settings.audio_read.max_retries - 1:
                    logger.info(
                        msg=(
                            "Retrying with librosa in "
                            f"{settings.audio_read.retry_delay_seconds} seconds..."
                        )
                    )
                    time.sleep(settings.audio_read.retry_delay_seconds)
                    continue
                raise

            logger.warning(msg="Falling back to soundfile...")
            try:
                with sf.SoundFile(str(path)) as sound_file:
                    raw_audio = np.asarray(
                        sound_file.read(dtype="float32"), dtype=np.float32
                    )
                    current_sample_rate = int(sound_file.samplerate)
                normalized_audio = _prepare_audio_buffer(raw_audio)

                logger.debug(
                    msg=(f"Successfully read audio file using soundfile: {file_path}")
                )
                return normalized_audio, current_sample_rate

            except Exception as err:
                logger.warning(msg=f"Soundfile also failed: {err}")
                if attempt < settings.audio_read.max_retries - 1:
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
