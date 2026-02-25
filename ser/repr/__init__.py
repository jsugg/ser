"""Representation backend contracts and implementations."""

from .backend import (
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    VectorFeatureBackend,
    overlap_frame_mask,
)
from .emotion2vec import Emotion2VecBackend
from .handcrafted import HandcraftedBackend
from .hf_whisper import WhisperBackend
from .hf_xlsr import XLSRBackend

__all__ = [
    "EncodedSequence",
    "Emotion2VecBackend",
    "FeatureBackend",
    "HandcraftedBackend",
    "PoolingWindow",
    "WhisperBackend",
    "XLSRBackend",
    "VectorFeatureBackend",
    "overlap_frame_mask",
]
