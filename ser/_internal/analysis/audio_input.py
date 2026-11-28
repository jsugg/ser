"""Caller audio in, canonical audio out, through the one shared conversion seam.

The seam is ``ser._internal.utils.audio_utils.to_canonical_audio``, which training
uses too. Analysis must call it as ``audio_utils.to_canonical_audio`` so that tests
can observe it at its owner.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from ser._internal.utils import audio_utils
from ser._internal.utils.audio_utils import CANONICAL_SAMPLE_RATE, CanonicalAudio


@dataclass(frozen=True)
class DecodedAudio:
    """Audio as stored: native rate, frames first, every channel kept."""

    samples: NDArray[np.float32] | NDArray[np.int16] | NDArray[np.int32]
    sample_rate_hz: int


@dataclass(frozen=True)
class UnreadableAudio:
    """Input that could not be decoded; carries no path and no exception text."""


def decode_file(path: str | os.PathLike[str]) -> DecodedAudio | UnreadableAudio:
    """Reads one audio file without mixing channels or resampling.

    A file whose data is shorter than its header declares is unreadable.
    """
    try:
        with open(path, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return UnreadableAudio()
            header = stream.read(12)
            # libsndfile may silently reduce its reported frame count for a truncated WAV.
            if header[:4] in (b"RIFF", b"RIFX"):
                endian = "little" if header[:4] == b"RIFF" else "big"
                if len(header) != 12 or int.from_bytes(header[4:8], endian) + 8 > info.st_size:
                    return UnreadableAudio()
            stream.seek(0)
            with sf.SoundFile(stream) as source:
                dtype = {
                    "PCM_16": "int16",
                    "PCM_24": "int32",
                    "PCM_32": "int32",
                }.get(source.subtype, "float32")
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
