"""Typed backend contracts for representation encoding and pooling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

type EmbeddingMatrix = NDArray[np.float32]
type TimeVector = NDArray[np.float64]
type FeatureMatrix = NDArray[np.float64]
type FeatureVector = NDArray[np.float64]
type WindowMask = NDArray[np.bool_]


@dataclass(frozen=True)
class PoolingWindow:
    """Temporal window used when pooling encoded frame features."""

    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        """Validates basic window invariants."""
        if not np.isfinite(self.start_seconds) or not np.isfinite(self.end_seconds):
            raise ValueError("PoolingWindow bounds must be finite numbers.")
        if self.start_seconds < 0.0:
            raise ValueError("PoolingWindow start_seconds must be non-negative.")
        if self.end_seconds <= self.start_seconds:
            raise ValueError(
                "PoolingWindow end_seconds must be greater than start_seconds."
            )


@dataclass(frozen=True)
class EncodedSequence:
    """Frame-level encoded representation with explicit temporal boundaries."""

    embeddings: EmbeddingMatrix
    frame_start_seconds: TimeVector
    frame_end_seconds: TimeVector
    backend_id: str

    def __post_init__(self) -> None:
        """Validates encoded sequence shape and temporal consistency."""
        if not self.backend_id:
            raise ValueError("EncodedSequence backend_id must be a non-empty string.")

        if self.embeddings.ndim != 2:
            raise ValueError("EncodedSequence embeddings must be 2D (frames, features).")

        if self.frame_start_seconds.ndim != 1 or self.frame_end_seconds.ndim != 1:
            raise ValueError("Frame timestamp arrays must be 1D.")

        frame_count: int = int(self.embeddings.shape[0])
        if frame_count <= 0:
            raise ValueError("EncodedSequence must contain at least one frame.")

        if self.frame_start_seconds.size != frame_count:
            raise ValueError(
                "frame_start_seconds length must match embeddings frame count."
            )
        if self.frame_end_seconds.size != frame_count:
            raise ValueError(
                "frame_end_seconds length must match embeddings frame count."
            )

        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("EncodedSequence embeddings contain non-finite values.")
        if not np.all(np.isfinite(self.frame_start_seconds)):
            raise ValueError(
                "EncodedSequence frame_start_seconds contain non-finite values."
            )
        if not np.all(np.isfinite(self.frame_end_seconds)):
            raise ValueError("EncodedSequence frame_end_seconds contain non-finite values.")

        if np.any(np.diff(self.frame_start_seconds) < 0.0):
            raise ValueError("frame_start_seconds must be non-decreasing.")
        if np.any(np.diff(self.frame_end_seconds) < 0.0):
            raise ValueError("frame_end_seconds must be non-decreasing.")

        if np.any(self.frame_end_seconds <= self.frame_start_seconds):
            raise ValueError("Each frame must satisfy end_seconds > start_seconds.")


def overlap_frame_mask(encoded: EncodedSequence, window: PoolingWindow) -> WindowMask:
    """Returns a mask for frames that overlap the provided pooling window.

    Args:
        encoded: Sequence containing frame embeddings and boundaries.
        window: Time range to pool.

    Returns:
        Boolean mask aligned with encoded frames.

    Raises:
        ValueError: If window bounds are outside encoded range or overlap no frames.
    """
    min_start = float(encoded.frame_start_seconds[0])
    max_end = float(encoded.frame_end_seconds[-1])
    if window.start_seconds < min_start or window.end_seconds > max_end:
        raise ValueError(
            "Pooling window is outside encoded sequence range: "
            f"[{window.start_seconds}, {window.end_seconds}] vs [{min_start}, {max_end}]"
        )

    mask = np.logical_and(
        encoded.frame_end_seconds > window.start_seconds,
        encoded.frame_start_seconds < window.end_seconds,
    )
    if not np.any(mask):
        raise ValueError(
            "Pooling window does not overlap any encoded frames: "
            f"[{window.start_seconds}, {window.end_seconds}]"
        )
    return mask


@runtime_checkable
class FeatureBackend(Protocol):
    """Backend protocol for sequence encoding and temporal pooling."""

    @property
    def backend_id(self) -> str:
        """Unique backend identifier persisted for compatibility checks."""
        ...

    @property
    def feature_dim(self) -> int:
        """Feature dimension produced per pooled vector."""
        ...

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        """Encodes audio into frame-level representations."""
        ...

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> FeatureMatrix:
        """Pools encoded representations over one or more temporal windows."""
        ...


@runtime_checkable
class VectorFeatureBackend(FeatureBackend, Protocol):
    """Optional convenience protocol for direct vector extraction."""

    def extract_vector(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> FeatureVector:
        """Extracts one feature vector from a full clip or window."""
        ...

