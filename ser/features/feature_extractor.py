"""Audio feature extraction utilities used by the SER model pipeline."""

import logging
from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, FeatureFlags, get_settings
from ser.repr import HandcraftedBackend
from ser.repr.backend import EncodedSequence
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
    pad_width: int = minimum_window - audio.size
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

    settings: AppConfig = get_settings()
    feature_flags: FeatureFlags = settings.feature_flags
    prepared_audio: NDArray[np.float32] = _pad_audio_for_fft(
        np.asarray(audio, dtype=np.float32)
    )
    if not bool(np.all(np.isfinite(prepared_audio))):
        raise ValueError("Audio buffer is not finite everywhere.")
    n_fft: int = min(prepared_audio.size, 2048)
    stft_magnitude: NDArray[np.float32] = np.abs(librosa.stft(prepared_audio, n_fft=n_fft))
    stft_power_db: NDArray[np.float32] = librosa.power_to_db(
        np.square(stft_magnitude),
        ref=np.max,
    )

    feature_parts: list[NDArray[np.float64]] = []
    try:
        if feature_flags.mfcc:
            mfccs: NDArray[np.float64] = np.mean(
                librosa.feature.mfcc(
                    y=prepared_audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(mfccs, dtype=np.float64))

        if feature_flags.chroma:
            chroma: NDArray[np.float64] = np.mean(
                librosa.feature.chroma_stft(
                    S=stft_magnitude, sr=sample_rate, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(chroma, dtype=np.float64))

        if feature_flags.mel:
            mel: NDArray[np.float64] = np.mean(
                librosa.feature.melspectrogram(
                    y=prepared_audio, sr=sample_rate, n_fft=n_fft
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(mel, dtype=np.float64))

        if feature_flags.contrast:
            spectral_contrast: NDArray[np.float64] = np.mean(
                librosa.feature.spectral_contrast(
                    S=stft_power_db,
                    sr=sample_rate,
                    n_fft=n_fft,
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(spectral_contrast, dtype=np.float64))

        if feature_flags.tonnetz:
            harmonic: NDArray[np.float32] = librosa.effects.harmonic(prepared_audio)
            tonnetz: NDArray[np.float64] = np.mean(
                librosa.feature.tonnetz(y=harmonic, sr=sample_rate),
                axis=1,
            )
            feature_parts.append(np.asarray(tonnetz, dtype=np.float64))
    except Exception as err:
        logger.warning("Error extracting features from signal: %s", err)
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
    backend = HandcraftedBackend()
    return backend.extract_vector(audio=audio, sample_rate=sample_rate)


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
    frames: list[FeatureFrame] = extract_feature_frames(
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

    audio: NDArray[np.float32]
    sample_rate: int
    audio, sample_rate = read_audio_file(audiofile)
    backend = HandcraftedBackend(
        frame_size_seconds=frame_size,
        frame_stride_seconds=frame_stride,
    )
    encoded: EncodedSequence = backend.encode_sequence(audio=audio, sample_rate=sample_rate)

    return [
        FeatureFrame(
            start_seconds=float(encoded.frame_start_seconds[index]),
            end_seconds=float(encoded.frame_end_seconds[index]),
            features=np.asarray(encoded.embeddings[index], dtype=np.float64),
        )
        for index in range(encoded.embeddings.shape[0])
    ]
