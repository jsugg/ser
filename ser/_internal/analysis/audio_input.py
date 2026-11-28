"""Caller audio in, canonical audio out, through the one shared conversion seam.

The seam is ``ser._internal.utils.audio_utils.to_canonical_audio``, which training
uses too. Analysis must call it as ``audio_utils.to_canonical_audio`` so that tests
can observe it at its owner.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import BinaryIO, Literal

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ser._internal.utils import audio_utils
from ser._internal.utils.audio_utils import CANONICAL_SAMPLE_RATE, CanonicalAudio


@dataclass(frozen=True)
class DecodedAudio:
    """Audio as stored: native rate, frames first, every channel kept."""

    samples: NDArray[np.generic]
    sample_rate_hz: int


@dataclass(frozen=True)
class UnreadableAudio:
    """Input that could not be decoded; carries no path and no exception text."""


def _riff_is_complete(stream: BinaryIO, size: int) -> bool:
    header = stream.read(12)
    if header[:4] not in (b"RIFF", b"RIFX"):
        return True
    if len(header) != 12:
        return False
    endian: Literal["little", "big"] = "little" if header[:4] == b"RIFF" else "big"
    end = int.from_bytes(header[4:8], endian) + 8
    if not 12 <= end <= size:
        return False
    if header[8:] != b"WAVE":
        return True
    # libsndfile reduces the reported frame count when a WAV data chunk is truncated.
    while stream.tell() < end:
        chunk = stream.read(8)
        if len(chunk) != 8:
            return False
        length = int.from_bytes(chunk[4:], endian)
        next_chunk = stream.tell() + length
        if next_chunk > end:
            return False
        if chunk[:4] == b"data":
            return True
        stream.seek(next_chunk + length % 2)
    return False


def decode_file(path: str | os.PathLike[str]) -> DecodedAudio | UnreadableAudio:
    """Reads one audio file without mixing channels or resampling.

    A file whose data is shorter than its header declares is unreadable.
    """
    try:
        with open(path, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not _riff_is_complete(stream, info.st_size):
                return UnreadableAudio()
            stream.seek(0)
            with sf.SoundFile(stream) as source:
                dtype: Literal["int16", "int32", "float32"] = "float32"
                if source.subtype == "PCM_16":
                    dtype = "int16"
                elif source.subtype in ("PCM_24", "PCM_32"):
                    dtype = "int32"
                samples = source.read(dtype=dtype)
                if not 0 < len(samples) == source.frames:
                    return UnreadableAudio()
                return DecodedAudio(samples, source.samplerate)
    except (OSError, ValueError, sf.LibsndfileError):
        return UnreadableAudio()


def canonicalize(audio: DecodedAudio) -> CanonicalAudio:
    """Converts decoded audio with ``audio_utils.to_canonical_audio``, frames first;
    never mixes or resamples itself."""
    return audio_utils.to_canonical_audio(audio.samples, audio.sample_rate_hz, channel_axis=1)


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "CanonicalAudio",
    "DecodedAudio",
    "UnreadableAudio",
    "canonicalize",
    "decode_file",
]
