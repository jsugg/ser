"""Contract tests for representation backend interfaces."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from ser.repr import (
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    VectorFeatureBackend,
    overlap_frame_mask,
)

type FeatureMatrix = NDArray[np.float64]


class StubBackend:
    """Deterministic backend used to validate protocol behavior."""

    @property
    def backend_id(self) -> str:
        """Stable backend identifier."""
        return "stub"

    @property
    def feature_dim(self) -> int:
        """Returns pooled vector width."""
        return 2

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        """Encodes into three fixed frames."""
        del audio, sample_rate
        return EncodedSequence(
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
            backend_id=self.backend_id,
        )

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> FeatureMatrix:
        """Mean-pools all overlapping frames per window."""
        if not windows:
            return np.empty((0, self.feature_dim), dtype=np.float64)
        pooled_rows: list[NDArray[np.float64]] = []
        for window in windows:
            mask = overlap_frame_mask(encoded, window)
            pooled_rows.append(
                np.asarray(encoded.embeddings[mask].mean(axis=0), dtype=np.float64)
            )
        return np.vstack(pooled_rows)

    def extract_vector(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float64]:
        """Returns one pooled vector for convenience API coverage."""
        encoded = self.encode_sequence(audio, sample_rate)
        pooled = self.pool(
            encoded,
            [PoolingWindow(start_seconds=0.0, end_seconds=3.0)],
        )
        return np.asarray(pooled[0], dtype=np.float64)


def test_encoded_sequence_accepts_valid_frame_alignment() -> None:
    """Aligned arrays and valid boundaries should construct successfully."""
    encoded = EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0], dtype=np.float64),
        backend_id="handcrafted",
    )
    assert encoded.embeddings.shape == (2, 2)


def test_encoded_sequence_rejects_misaligned_frame_arrays() -> None:
    """Mismatched frame timestamps should fail fast."""
    with pytest.raises(ValueError, match="frame_end_seconds length"):
        EncodedSequence(
            embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
            frame_end_seconds=np.asarray([1.0], dtype=np.float64),
            backend_id="broken",
        )


def test_encoded_sequence_rejects_non_positive_frame_duration() -> None:
    """Each encoded frame must have end > start."""
    with pytest.raises(ValueError, match="end_seconds > start_seconds"):
        EncodedSequence(
            embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
            frame_end_seconds=np.asarray([1.0, 1.0], dtype=np.float64),
            backend_id="broken",
        )


def test_stub_backend_conforms_to_protocol_and_pools_deterministically() -> None:
    """Stub backend should satisfy protocol and produce deterministic pooling."""
    backend = StubBackend()
    assert isinstance(backend, FeatureBackend)
    assert isinstance(backend, VectorFeatureBackend)

    encoded = backend.encode_sequence(
        audio=np.zeros(3200, dtype=np.float32),
        sample_rate=16000,
    )
    pooled = backend.pool(
        encoded,
        [
            PoolingWindow(start_seconds=0.0, end_seconds=2.0),
            PoolingWindow(start_seconds=1.0, end_seconds=3.0),
        ],
    )

    assert pooled.shape == (2, backend.feature_dim)
    np.testing.assert_allclose(
        pooled,
        np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
    )


def test_overlap_frame_mask_rejects_out_of_range_window() -> None:
    """Pooling windows outside encoded range should raise."""
    encoded = EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0, 2.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 3.0], dtype=np.float64),
        backend_id="handcrafted",
    )

    with pytest.raises(ValueError, match="outside encoded sequence range"):
        overlap_frame_mask(encoded, PoolingWindow(start_seconds=0.5, end_seconds=3.5))


def test_overlap_frame_mask_rejects_window_without_any_overlap() -> None:
    """Pooling windows that overlap no frames should raise."""
    encoded = EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0, 2.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 3.0], dtype=np.float64),
        backend_id="handcrafted",
    )

    with pytest.raises(ValueError, match="does not overlap any encoded frames"):
        overlap_frame_mask(encoded, PoolingWindow(start_seconds=1.2, end_seconds=1.8))
