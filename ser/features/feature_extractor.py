"""Audio feature extraction utilities used by the SER model pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, get_settings
from ser.repr import HandcraftedBackend
from ser.repr.backend import EncodedSequence
from ser.utils.audio_utils import read_audio_file
from ser.utils.dsp import extract_feature_from_signal as _extract_feature_from_signal
from ser.utils.logger import get_logger

type FeatureVector = NDArray[np.float64]
logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureFrame:
    """A frame-level feature vector with explicit temporal boundaries."""

    start_seconds: float
    end_seconds: float
    features: FeatureVector


def extract_feature_from_signal(
    audio: NDArray[np.float32],
    sample_rate: int,
    *,
    settings: AppConfig | None = None,
) -> FeatureVector:
    """Compatibility wrapper around DSP-level handcrafted extraction."""
    active_settings = settings if settings is not None else get_settings()
    return _extract_feature_from_signal(
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
    audio: NDArray[np.float32]
    sample_rate: int
    active_settings = settings if settings is not None else get_settings()
    try:
        audio, sample_rate = read_audio_file(
            file,
            audio_read_config=active_settings.audio_read,
        )
    except Exception as err:
        logger.error(msg=f"Error reading file {file}: {err}")
        raise
    backend = HandcraftedBackend(feature_flags=active_settings.feature_flags)
    return backend.extract_vector(audio=audio, sample_rate=sample_rate)


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
    if frame_size <= 0:
        raise ValueError("frame_size must be greater than zero.")
    if frame_stride <= 0:
        raise ValueError("frame_stride must be greater than zero.")

    active_settings = settings if settings is not None else get_settings()
    audio: NDArray[np.float32]
    sample_rate: int
    audio, sample_rate = read_audio_file(
        audiofile,
        audio_read_config=active_settings.audio_read,
    )
    backend = HandcraftedBackend(
        frame_size_seconds=frame_size,
        frame_stride_seconds=frame_stride,
        feature_flags=active_settings.feature_flags,
    )
    encoded: EncodedSequence = backend.encode_sequence(
        audio=audio, sample_rate=sample_rate
    )

    return [
        FeatureFrame(
            start_seconds=float(encoded.frame_start_seconds[index]),
            end_seconds=float(encoded.frame_end_seconds[index]),
            features=np.asarray(encoded.embeddings[index], dtype=np.float64),
        )
        for index in range(encoded.embeddings.shape[0])
    ]
