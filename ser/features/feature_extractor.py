"""Audio feature extraction utilities used by the SER model pipeline."""

import logging
from dataclasses import dataclass

import librosa
import numpy as np
from halo import Halo
from numpy.typing import NDArray

from ser.config import get_settings
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

type FeatureVector = NDArray[np.float64]


@dataclass(frozen=True)
class FeatureFrame:
    """A frame-level feature vector with explicit temporal boundaries."""

    start_seconds: float
    end_seconds: float
    features: FeatureVector


def _pad_audio_for_fft(
    audio: NDArray[np.float32], minimum_window: int = 512
) -> NDArray[np.float32]:
    """Pads short clips so spectral features can be computed safely."""
    if audio.size >= minimum_window:
        return audio
    pad_width = minimum_window - audio.size
    return np.pad(audio, (0, pad_width), mode="constant")


def extract_feature_from_signal(
    audio: NDArray[np.float32], sample_rate: int
) -> FeatureVector:
    """Extracts configured features from an in-memory mono signal.

    Args:
        audio: Mono PCM samples in `float32`.
        sample_rate: Audio sample rate in Hz.

    Returns:
        A one-dimensional feature vector combining all enabled feature groups.

    Raises:
        ValueError: If the audio buffer or sample rate is invalid.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    if audio.ndim != 1:
        raise ValueError("Audio must be mono (1D array).")
    if audio.size == 0:
        raise ValueError("Audio contains no samples.")

    settings = get_settings()
    feature_flags = settings.feature_flags
    prepared_audio = _pad_audio_for_fft(np.asarray(audio, dtype=np.float32))
    n_fft: int = min(prepared_audio.size, 2048)
    stft_magnitude = np.abs(librosa.stft(prepared_audio, n_fft=n_fft))
    stft_power_db = librosa.power_to_db(np.square(stft_magnitude), ref=np.max)

    feature_parts: list[NDArray[np.float64]] = []
    try:
        if feature_flags.mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(
                    y=prepared_audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(mfccs, dtype=np.float64))

        if feature_flags.chroma:
            chroma = np.mean(
                librosa.feature.chroma_stft(
                    S=stft_magnitude, sr=sample_rate, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(chroma, dtype=np.float64))

        if feature_flags.mel:
            mel = np.mean(
                librosa.feature.melspectrogram(
                    y=prepared_audio, sr=sample_rate, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(mel, dtype=np.float64))

        if feature_flags.contrast:
            spectral_contrast = np.mean(
                librosa.feature.spectral_contrast(
                    S=stft_power_db,
                    sr=sample_rate,
                    n_fft=n_fft,
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(spectral_contrast, dtype=np.float64))

        if feature_flags.tonnetz:
            harmonic = librosa.effects.harmonic(prepared_audio)
            tonnetz = np.mean(
                librosa.feature.tonnetz(y=harmonic, sr=sample_rate),
                axis=1,
            )
            feature_parts.append(np.asarray(tonnetz, dtype=np.float64))
    except Exception as err:
        logger.error(msg=f"Error extracting features from signal: {err}", exc_info=True)
        raise

    if not feature_parts:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(feature_parts).astype(np.float64, copy=False)


def extract_feature(file: str) -> FeatureVector:
    """Extracts the configured spectral features from one audio file.

    Args:
        file: Path to the audio file.

    Returns:
        A one-dimensional feature vector combining all enabled feature groups.
    """
    audio: NDArray[np.float32]
    sample_rate: int
    try:
        audio, sample_rate = read_audio_file(file)
    except Exception as err:
        logger.error(msg=f"Error reading file {file}: {err}")
        raise
    return extract_feature_from_signal(audio, sample_rate)


def extended_extract_feature(
    audiofile: str, frame_size: int = 3, frame_stride: int = 1
) -> list[FeatureVector]:
    """Extracts frame-wise feature vectors from an audio file.

    Args:
        audiofile: Path to the audio file.
        frame_size: Duration of each frame, in seconds.
        frame_stride: Step between successive frames, in seconds.

    Returns:
        A list of feature vectors, one for each extracted frame.
    """
    frames = extract_feature_frames(
        audiofile=audiofile,
        frame_size=frame_size,
        frame_stride=frame_stride,
    )
    return [frame.features for frame in frames]


def extract_feature_frames(
    audiofile: str,
    frame_size: int = 3,
    frame_stride: int = 1,
) -> list[FeatureFrame]:
    """Extracts frame-wise features with explicit start/end timestamps."""
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than zero.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than zero.")

    frames: list[FeatureFrame] = []
    audio: NDArray[np.float32]
    sample_rate: int
    audio, sample_rate = read_audio_file(audiofile)
    frame_length: int = max(1, int(round(frame_size * sample_rate)))
    frame_step: int = max(1, int(round(frame_stride * sample_rate)))

    with Halo(text="Processing", spinner="dots", text_color="green"):
        for start in range(0, audio.size, frame_step):
            end: int = min(start + frame_length, audio.size)
            frame_audio = audio[start:end]
            if frame_audio.size == 0:
                continue
            frame_features = extract_feature_from_signal(frame_audio, sample_rate)
            frames.append(
                FeatureFrame(
                    start_seconds=float(start) / float(sample_rate),
                    end_seconds=float(end) / float(sample_rate),
                    features=frame_features,
                )
            )

    return frames
