"""Tests for deterministic stats pooling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ser.pool import mean_std_pool
from ser.repr import EncodedSequence, PoolingWindow


def test_mean_std_pool_returns_expected_stats_per_window() -> None:
    """Mean/std pooling should be deterministic for overlapping windows."""
    encoded = EncodedSequence(
        embeddings=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        frame_start_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )

    pooled = mean_std_pool(
        encoded,
        [
            PoolingWindow(start_seconds=0.0, end_seconds=2.0),
            PoolingWindow(start_seconds=1.0, end_seconds=3.0),
        ],
    )

    np.testing.assert_allclose(
        pooled,
        np.asarray(
            [
                [2.0, 3.0, 1.0, 1.0],
                [4.0, 5.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        ),
    )


def test_mean_std_pool_returns_empty_matrix_for_empty_windows() -> None:
    """Empty window lists should return an empty `(0, feature_dim * 2)` matrix."""
    encoded = EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )

    pooled = mean_std_pool(encoded, [])

    assert pooled.shape == (0, 4)


def test_mean_std_pool_rejects_windows_outside_encoded_range() -> None:
    """Out-of-range windows should fail through overlap validation."""
    encoded = EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )

    with pytest.raises(ValueError, match="outside encoded sequence range"):
        mean_std_pool(
            encoded,
            [PoolingWindow(start_seconds=0.0, end_seconds=3.0)],
        )
