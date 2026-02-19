"""Representation backend contracts and implementations."""

from .backend import (
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    VectorFeatureBackend,
    overlap_frame_mask,
)
from .handcrafted import HandcraftedBackend
from .hf_xlsr import XLSRBackend

__all__ = [
    "EncodedSequence",
    "FeatureBackend",
    "HandcraftedBackend",
    "PoolingWindow",
    "XLSRBackend",
    "VectorFeatureBackend",
    "overlap_frame_mask",
]
