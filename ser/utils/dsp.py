"""DSP-level handcrafted feature extraction helpers."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from numpy.typing import NDArray

from ser.config import FeatureFlags
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

type FeatureVector = NDArray[np.float64]


def _pad_audio_for_fft(
    audio: NDArray[np.float32], minimum_window: int = 512
) -> NDArray[np.float32]:
    """Pads short clips so spectral features can be computed safely."""
    if audio.size >= minimum_window:
        return audio
    pad_width: int = minimum_window - audio.size
    return np.pad(audio, (0, pad_width), mode="constant")


def extract_feature_from_signal(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    feature_flags: FeatureFlags | None = None,
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

    active_feature_flags = feature_flags if feature_flags is not None else FeatureFlags()
    prepared_audio: NDArray[np.float32] = _pad_audio_for_fft(np.asarray(audio, dtype=np.float32))
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
        if active_feature_flags.mfcc:
            mfccs: NDArray[np.float64] = np.mean(
                librosa.feature.mfcc(y=prepared_audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft),
                axis=1,
            )
            feature_parts.append(np.asarray(mfccs, dtype=np.float64))

        if active_feature_flags.chroma:
            chroma: NDArray[np.float64] = np.mean(
                librosa.feature.chroma_stft(S=stft_magnitude, sr=sample_rate, n_fft=n_fft),
                axis=1,
            )
            feature_parts.append(np.asarray(chroma, dtype=np.float64))

        if active_feature_flags.mel:
            mel: NDArray[np.float64] = np.mean(
                librosa.feature.melspectrogram(y=prepared_audio, sr=sample_rate, n_fft=n_fft),
                axis=1,
            )
            feature_parts.append(np.asarray(mel, dtype=np.float64))

        if active_feature_flags.contrast:
            spectral_contrast: NDArray[np.float64] = np.mean(
                librosa.feature.spectral_contrast(
                    S=stft_power_db,
                    sr=sample_rate,
                    n_fft=n_fft,
                ),
                axis=1,
            )
            feature_parts.append(np.asarray(spectral_contrast, dtype=np.float64))

        if active_feature_flags.tonnetz:
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
