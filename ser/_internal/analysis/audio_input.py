"""Caller audio in, canonical audio out, through the one shared conversion seam.

``to_canonical_audio``, ``CanonicalAudio`` and ``CANONICAL_SAMPLE_RATE`` below are
placeholders with the signature of ``ser._internal.utils.audio_utils`` (brief M5-1).
When that lands, replace them with an import of the same names; analysis must call
``to_canonical_audio`` through this module so that tests can observe the seam.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CANONICAL_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class CanonicalAudio:
    """Mono float32 audio at ``CANONICAL_SAMPLE_RATE``."""

    samples: NDArray[np.float32]
    sample_rate: int
    original_sample_rate: int
    duration_seconds: float


def to_canonical_audio(
    samples: NDArray[np.floating],
    sample_rate: int,
    *,
    channel_axis: int = 1,
) -> CanonicalAudio:
    """Placeholder for the shared train/infer conversion of brief M5-1."""
    raise NotImplementedError("Canonical audio conversion lands with brief M5-1.")


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
    """Converts decoded audio with ``to_canonical_audio``; never resamples itself."""
    raise NotImplementedError("Canonical conversion wiring is not implemented yet (brief A1-2).")


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "CanonicalAudio",
    "DecodedAudio",
    "UnreadableAudio",
    "canonicalize",
    "decode_file",
    "to_canonical_audio",
]
