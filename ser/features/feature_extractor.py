"""Audio feature extraction utilities used by the SER model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, AudioReadConfig, FeatureFlags, get_settings
from ser.repr import HandcraftedBackend
from ser.repr.backend import EncodedSequence
from ser.utils.audio_utils import read_audio_file
from ser.utils.dsp import extract_feature_from_signal as _extract_feature_from_signal
from ser.utils.logger import get_logger

type FeatureVector = NDArray[np.float64]
logger = get_logger(__name__)


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or falls back to ambient public-boundary config."""
    return settings if settings is not None else get_settings()


@dataclass(frozen=True)
class FeatureFrame:
    """A frame-level feature vector with explicit temporal boundaries."""

    start_seconds: float
    end_seconds: float
    features: FeatureVector


def _extract_feature_from_signal_for_flags(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    feature_flags: FeatureFlags,
) -> FeatureVector:
    """Extracts one handcrafted feature vector using explicit feature flags."""
    return _extract_feature_from_signal(
        audio=audio,
        sample_rate=sample_rate,
        feature_flags=feature_flags,
    )


def _extract_feature_for_settings(
    file: str,
    *,
    feature_flags: FeatureFlags,
    audio_read_config: AudioReadConfig,
) -> FeatureVector:
    """Extracts one feature vector using explicit audio-read and feature settings."""
    audio: NDArray[np.float32]
    sample_rate: int
    try:
        audio, sample_rate = read_audio_file(
            file,
            audio_read_config=audio_read_config,
        )
    except Exception as err:
        logger.error(msg=f"Error reading file {file}: {err}")
        raise
    backend = HandcraftedBackend(feature_flags=feature_flags)
    return backend.extract_vector(audio=audio, sample_rate=sample_rate)


def _extract_feature_frames_for_settings(
    audiofile: str,
    *,
    frame_size: int,
    frame_stride: int,
    feature_flags: FeatureFlags,
    audio_read_config: AudioReadConfig,
) -> list[FeatureFrame]:
    """Extracts frame-wise features using one explicit settings snapshot."""
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than zero.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than zero.")

    audio: NDArray[np.float32]
    sample_rate: int
    audio, sample_rate = read_audio_file(
        audiofile,
        audio_read_config=audio_read_config,
    )
    backend = HandcraftedBackend(
        frame_size_seconds=frame_size,
        frame_stride_seconds=frame_stride,
        feature_flags=feature_flags,
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


def extract_feature_from_signal(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    settings: AppConfig | None = None,
) -> FeatureVector:
    """Compatibility wrapper around DSP-level handcrafted extraction."""
    active_settings = _resolve_boundary_settings(settings)
    return _extract_feature_from_signal_for_flags(
        audio=audio,
        sample_rate=sample_rate,
        feature_flags=active_settings.feature_flags,
    )


def extract_feature(file: str, *, settings: AppConfig | None = None) -> FeatureVector:
    """Extracts the configured spectral features from one audio file.

    Args:
        file: Path to the audio file.

    Returns:
        A one-dimensional feature vector combining all enabled feature groups.
    """
    active_settings = _resolve_boundary_settings(settings)
    return _extract_feature_for_settings(
        file,
        feature_flags=active_settings.feature_flags,
        audio_read_config=active_settings.audio_read,
    )


def extended_extract_feature(
    audiofile: str,
    frame_size: int = 3,
    frame_stride: int = 1,
    *,
    settings: AppConfig | None = None,
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
        settings=settings,
    )
    return [frame.features for frame in frames]


def extract_feature_frames(
    audiofile: str,
    frame_size: int = 3,
    frame_stride: int = 1,
    *,
    settings: AppConfig | None = None,
) -> list[FeatureFrame]:
    """Extracts frame-wise features with explicit start/end timestamps."""
    active_settings = _resolve_boundary_settings(settings)
    return _extract_feature_frames_for_settings(
        audiofile,
        frame_size=frame_size,
        frame_stride=frame_stride,
        feature_flags=active_settings.feature_flags,
        audio_read_config=active_settings.audio_read,
    )
