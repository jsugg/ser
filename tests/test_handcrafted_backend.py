"""Behavior tests for the handcrafted feature backend."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from ser.repr import HandcraftedBackend, PoolingWindow


def test_handcrafted_backend_feature_dim_reflects_feature_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature dimension should follow enabled handcrafted feature groups."""
    monkeypatch.setattr(
        "ser.repr.handcrafted.get_settings",
        lambda: SimpleNamespace(
            feature_flags=SimpleNamespace(
                mfcc=True,
                chroma=True,
                mel=True,
                contrast=True,
                tonnetz=True,
            )
        ),
    )
    backend = HandcraftedBackend()
    assert backend.feature_dim == 193


def test_handcrafted_backend_encode_sequence_uses_frame_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encoding should preserve deterministic frame start/end semantics."""
    monkeypatch.setattr(
        "ser.features.feature_extractor.extract_feature_from_signal",
        lambda audio, _sample_rate: np.asarray(
            [float(audio[0]), float(audio[-1])],
            dtype=np.float64,
        ),
    )
    backend = HandcraftedBackend(frame_size_seconds=2, frame_stride_seconds=1)
    audio: NDArray[np.float32] = np.arange(10, dtype=np.float32)

    encoded = backend.encode_sequence(audio, sample_rate=2)

    np.testing.assert_allclose(
        encoded.frame_start_seconds,
        np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        encoded.frame_end_seconds,
        np.asarray([2.0, 3.0, 4.0, 5.0, 5.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        encoded.embeddings,
        np.asarray(
            [
                [0.0, 3.0],
                [2.0, 5.0],
                [4.0, 7.0],
                [6.0, 9.0],
                [8.0, 9.0],
            ],
            dtype=np.float32,
        ),
    )


def test_handcrafted_backend_pool_mean_aggregates_overlapping_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pooling should average all frames overlapping each window."""
    monkeypatch.setattr(
        "ser.features.feature_extractor.extract_feature_from_signal",
        lambda audio, _sample_rate: np.asarray(
            [float(audio[0]), float(audio[-1])],
            dtype=np.float64,
        ),
    )
    backend = HandcraftedBackend(frame_size_seconds=2, frame_stride_seconds=1)
    encoded = backend.encode_sequence(np.arange(10, dtype=np.float32), sample_rate=2)

    pooled = backend.pool(
        encoded,
        [
            PoolingWindow(start_seconds=0.0, end_seconds=1.0),
            PoolingWindow(start_seconds=2.0, end_seconds=4.0),
        ],
    )

    np.testing.assert_allclose(
        pooled,
        np.asarray([[0.0, 3.0], [4.0, 7.0]], dtype=np.float64),
    )


def test_handcrafted_backend_extract_vector_uses_signal_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """extract_vector should preserve legacy whole-signal feature semantics."""
    monkeypatch.setattr(
        "ser.features.feature_extractor.extract_feature_from_signal",
        lambda audio, sample_rate: np.asarray(
            [float(audio.size), float(sample_rate)],
            dtype=np.float64,
        ),
    )
    backend = HandcraftedBackend(frame_size_seconds=2, frame_stride_seconds=1)
    audio = np.arange(10, dtype=np.float32)

    vector = backend.extract_vector(audio, sample_rate=2)
    np.testing.assert_allclose(vector, np.asarray([10.0, 2.0], dtype=np.float64))


def test_handcrafted_backend_rejects_invalid_frame_arguments() -> None:
    """Frame size and stride must be strictly positive."""
    with pytest.raises(ValueError, match="frame_size_seconds"):
        HandcraftedBackend(frame_size_seconds=0, frame_stride_seconds=1)
    with pytest.raises(ValueError, match="frame_stride_seconds"):
        HandcraftedBackend(frame_size_seconds=1, frame_stride_seconds=0)
