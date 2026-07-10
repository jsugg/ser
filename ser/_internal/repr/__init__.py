"""Internal representation backend contracts and implementations."""

from ser._internal.repr.backend import (
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    VectorFeatureBackend,
    overlap_frame_mask,
)
from ser._internal.repr.emotion2vec import Emotion2VecBackend
from ser._internal.repr.handcrafted import HandcraftedBackend
from ser._internal.repr.hf_whisper import WhisperBackend
from ser._internal.repr.hf_xlsr import XLSRBackend

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
