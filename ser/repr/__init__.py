"""Representation backend contracts and implementations."""

from .backend import (
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    VectorFeatureBackend,
    overlap_frame_mask,
)
from .handcrafted import HandcraftedBackend

__all__ = [
    "EncodedSequence",
    "FeatureBackend",
    "HandcraftedBackend",
    "PoolingWindow",
    "VectorFeatureBackend",
    "overlap_frame_mask",
]
