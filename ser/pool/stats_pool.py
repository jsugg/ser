"""Deterministic statistics pooling helpers for encoded frame sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ser.repr import EncodedSequence, PoolingWindow, overlap_frame_mask

type PooledFeatureMatrix = NDArray[np.float64]


def mean_std_pool(
    encoded: EncodedSequence,
    windows: Sequence[PoolingWindow],
) -> PooledFeatureMatrix:
    """Pools encoded frames into mean+std feature vectors for each window.

    Args:
        encoded: Encoded frame sequence.
        windows: Temporal windows to pool.

    Returns:
        Matrix of shape `(len(windows), feature_dim * 2)`.

    Raises:
        ValueError: If a window is outside encoded bounds or overlaps no frames.
    """
    feature_dim = int(encoded.embeddings.shape[1])
    if not windows:
        return np.empty((0, feature_dim * 2), dtype=np.float64)

    pooled_rows: list[NDArray[np.float64]] = []
    for window in windows:
        mask = overlap_frame_mask(encoded, window)
        selected = np.asarray(encoded.embeddings[mask], dtype=np.float64)
        mean_vector = selected.mean(axis=0)
        std_vector = selected.std(axis=0)
        pooled_rows.append(np.concatenate((mean_vector, std_vector), axis=0))

    return np.vstack(pooled_rows).astype(np.float64, copy=False)
