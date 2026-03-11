"""Handcrafted feature backend used by the fast profile."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

import ser.utils.dsp as dsp
from ser.config import FeatureFlags
from ser.repr.backend import (
    EncodedSequence,
    FeatureBackend,
    FeatureMatrix,
    FeatureVector,
    PoolingWindow,
    overlap_frame_mask,
)


class HandcraftedBackend(FeatureBackend):
    """Backend wrapper around the existing librosa handcrafted feature extractor."""

    def __init__(
        self,
        *,
        frame_size_seconds: int = 3,
        frame_stride_seconds: int = 1,
        feature_flags: FeatureFlags | None = None,
    ) -> None:
        if frame_size_seconds <= 0:
            raise ValueError("frame_size_seconds must be greater than zero.")
        if frame_stride_seconds <= 0:
            raise ValueError("frame_stride_seconds must be greater than zero.")
        self._frame_size_seconds = frame_size_seconds
        self._frame_stride_seconds = frame_stride_seconds
        self._feature_flags = feature_flags if feature_flags is not None else FeatureFlags()

    @property
    def backend_id(self) -> str:
        """Stable handcrafted backend identifier."""
        return "handcrafted"

    @property
    def feature_dim(self) -> int:
        """Returns configured handcrafted feature dimension."""
        feature_size = 0
        if self._feature_flags.mfcc:
            feature_size += 40
        if self._feature_flags.chroma:
            feature_size += 12
        if self._feature_flags.mel:
            feature_size += 128
        if self._feature_flags.contrast:
            feature_size += 7
        if self._feature_flags.tonnetz:
            feature_size += 6
        return feature_size

    def prepare_runtime(self) -> None:
        """No-op warmup hook for runtime contract parity with other backends."""
        return None

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        """Encodes an audio signal into frame-level handcrafted features."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if audio.ndim != 1:
            raise ValueError("audio must be mono (1D array).")
        if audio.size == 0:
            raise ValueError("audio must contain at least one sample.")

        frame_length = max(1, int(round(self._frame_size_seconds * sample_rate)))
        frame_step = max(1, int(round(self._frame_stride_seconds * sample_rate)))

        starts: list[float] = []
        ends: list[float] = []
        frame_embeddings: list[NDArray[np.float32]] = []

        for start_index in range(0, audio.size, frame_step):
            end_index = min(start_index + frame_length, audio.size)
            frame_audio = audio[start_index:end_index]
            if frame_audio.size == 0:
                continue
            frame_features = dsp.extract_feature_from_signal(
                frame_audio,
                sample_rate,
                feature_flags=self._feature_flags,
            )
            frame_embeddings.append(np.asarray(frame_features, dtype=np.float32))
            starts.append(float(start_index) / float(sample_rate))
            ends.append(float(end_index) / float(sample_rate))

        if not frame_embeddings:
            raise ValueError("Could not extract handcrafted features from provided audio.")

        return EncodedSequence(
            embeddings=np.vstack(frame_embeddings).astype(np.float32, copy=False),
            frame_start_seconds=np.asarray(starts, dtype=np.float64),
            frame_end_seconds=np.asarray(ends, dtype=np.float64),
            backend_id=self.backend_id,
        )

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> FeatureMatrix:
        """Mean-pools encoded frames for each window."""
        if not windows:
            return np.empty((0, encoded.embeddings.shape[1]), dtype=np.float64)

        pooled_rows: list[FeatureVector] = []
        for window in windows:
            mask = overlap_frame_mask(encoded, window)
            pooled_rows.append(np.asarray(encoded.embeddings[mask].mean(axis=0), dtype=np.float64))
        return np.vstack(pooled_rows)

    def extract_vector(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> FeatureVector:
        """Extracts one handcrafted feature vector for whole-audio training paths."""
        return np.asarray(
            dsp.extract_feature_from_signal(
                audio,
                sample_rate,
                feature_flags=self._feature_flags,
            ),
            dtype=np.float64,
        )
