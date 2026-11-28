"""Caller audio in, canonical audio out, through the one shared conversion seam.

The seam is ``ser._internal.utils.audio_utils.to_canonical_audio``, which training
uses too. Analysis must call it as ``audio_utils.to_canonical_audio`` so that tests
can observe it at its owner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ser._internal.utils.audio_utils import CANONICAL_SAMPLE_RATE, CanonicalAudio


@dataclass(frozen=True)
class DecodedAudio:
    """Audio as stored: native rate, frames first, every channel kept."""

    samples: NDArray[np.float32]
    sample_rate_hz: int


@dataclass(frozen=True)
class UnreadableAudio:
    """Input that could not be decoded; carries no path and no exception text."""


def decode_file(path: str | os.PathLike[str]) -> DecodedAudio | UnreadableAudio:
    """Reads one audio file without mixing channels or resampling.

    A file whose data is shorter than its header declares is unreadable.
    """
    raise NotImplementedError("Utterance decoding is not implemented yet (brief A1-2).")


def canonicalize(audio: DecodedAudio) -> CanonicalAudio:
    """Converts decoded audio with ``audio_utils.to_canonical_audio``, frames first;
    never mixes or resamples itself."""
    raise NotImplementedError("Canonical conversion wiring is not implemented yet (brief A1-2).")


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "CanonicalAudio",
    "DecodedAudio",
    "UnreadableAudio",
    "canonicalize",
    "decode_file",
]
