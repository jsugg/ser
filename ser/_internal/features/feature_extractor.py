"""Audio feature extraction utilities used by the SER model pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_info

import ser
from ser._internal.repr import HandcraftedBackend
from ser._internal.repr.backend import EncodedSequence
from ser._internal.utils.audio_utils import CANONICAL_SAMPLE_RATE, read_audio_file
from ser._internal.utils.dsp import extract_feature_from_signal as _extract_feature_from_signal
from ser._internal.utils.logger import get_logger
from ser.config import AppConfig, AudioReadConfig, FeatureFlags, reload_settings

if TYPE_CHECKING:
    from ser._internal.data.manifest import Utterance

type FeatureVector = NDArray[np.float64]
logger = get_logger(__name__)
CANONICAL_FAST_FEATURE_CONTRACT_PREFIX = f"handcrafted-v1-{CANONICAL_SAMPLE_RATE}hz-"


def fast_feature_contract(feature_flags: FeatureFlags) -> str:
    """Revision persisted on fast models and in content-addressed feature keys."""
    names = "+".join(name for name, enabled in asdict(feature_flags).items() if enabled)
    return CANONICAL_FAST_FEATURE_CONTRACT_PREFIX + names


def fast_training_vectors(
    sample: Utterance, *, feature_flags: FeatureFlags, cache_dir: Path | None = None
) -> NDArray[np.float64]:
    """Return one exact inference vector per analysis window, without peak normalization."""
    from ser._internal.analysis.audio_input import DecodedAudio, canonicalize, decode_file
    from ser._internal.analysis.backend import frame_layout
    from ser._internal.models.artifact_envelope import (
        DEFAULT_FRAME_SIZE_SECONDS,
        DEFAULT_FRAME_STRIDE_SECONDS,
    )
    from ser._internal.models.training_orchestration import (  # noqa: TID251
        record_cache_activity,
        training_operation_active,
    )

    backend = HandcraftedBackend(feature_flags=feature_flags)
    cached = None
    recomputed = False
    if cache_dir is not None:
        with sample.audio_path.open("rb") as stream:
            audio_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        try:
            ser_version = metadata.version("ser")
        except metadata.PackageNotFoundError:
            ser_version = ser.__version__
        key = json.dumps(
            [
                fast_feature_contract(feature_flags),
                sorted(
                    (
                        pool["prefix"],
                        pool["internal_api"],
                        str(pool.get("version")),
                        str(pool.get("architecture")),
                        pool["num_threads"],
                    )
                    for pool in threadpool_info()
                    if pool["user_api"] == "blas"
                ),
                [ser_version, *(metadata.version(name) for name in ("librosa", "numpy", "scipy"))],
                audio_digest,
                sample.start_seconds,
                sample.duration_seconds,
                DEFAULT_FRAME_SIZE_SECONDS,
                DEFAULT_FRAME_STRIDE_SECONDS,
            ]
        )
        cached = cache_dir / (hashlib.sha256(key.encode()).hexdigest() + ".npy")
        if cached.is_file():
            try:
                with cached.open("rb") as stream:
                    rows = np.load(stream, allow_pickle=False)
                if (
                    isinstance(rows, np.ndarray)
                    and rows.dtype == np.float64
                    and rows.ndim == 2
                    and len(rows)
                    and rows.shape[1] == backend.feature_dim
                    and np.isfinite(rows).all()
                ):
                    if training_operation_active():
                        record_cache_activity(cache_hit=True, recomputed=False)
                    return np.asarray(rows, dtype=np.float64)
            except (ValueError, EOFError):
                pass
            recomputed = True

    decoded = decode_file(sample.audio_path, max_duration_s=float("inf"))
    if not isinstance(decoded, DecodedAudio):
        raise ValueError(f"Cannot decode fast training sample {sample.sample_id!r}.")
    rate = decoded.sample_rate_hz
    first = round((sample.start_seconds or 0) * rate)
    stop = (
        None if sample.duration_seconds is None else first + round(sample.duration_seconds * rate)
    )
    audio = canonicalize(DecodedAudio(decoded.samples[first:stop], rate))
    rows = np.stack(
        [
            backend.extract_vector(
                audio.samples[
                    round(start * audio.sample_rate) : (
                        None if end == audio.duration_seconds else round(end * audio.sample_rate)
                    )
                ],
                audio.sample_rate,
            )
            for start, end in frame_layout(
                audio.duration_seconds,
                frame_size_s=DEFAULT_FRAME_SIZE_SECONDS,
                stride_s=DEFAULT_FRAME_STRIDE_SECONDS,
            )
        ]
    )
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                np.save(handle, rows, allow_pickle=False)
                handle.close()
                os.replace(temporary, cached)
            finally:
                temporary.unlink(missing_ok=True)
        if training_operation_active():
            record_cache_activity(cache_hit=False, recomputed=recomputed)
    return rows


def _resolve_boundary_settings(settings: AppConfig | None) -> AppConfig:
    """Returns explicit settings or reloads a boundary-local settings snapshot."""
    return settings if settings is not None else reload_settings()


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
