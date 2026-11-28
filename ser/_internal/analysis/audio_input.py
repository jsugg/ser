"""Decoding seam between caller input and the canonical 16 kHz mono form."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DecodedUtterance:
    """Caller audio before canonical conversion, with the facts the record reports."""

    samples: NDArray[np.float32]
    input_sample_rate_hz: int
    duration_s: float


@dataclass(frozen=True)
class UnreadableUtterance:
    """Input that could not be decoded; carries no path and no exception text."""


def decode_file(path: str | os.PathLike[str]) -> DecodedUtterance | UnreadableUtterance:
    """Reads one audio file at its native rate, mixing channels down to mono."""
    raise NotImplementedError("Utterance decoding is not implemented yet (brief A1-2).")


def to_canonical(utterance: DecodedUtterance) -> NDArray[np.float32]:
    """The only place where analysis converts audio to 16 kHz mono float32.

    Delegates to the canonical conversion owned by WS5 (brief M5-1); this module
    must not resample on its own.
    """
    raise NotImplementedError("Canonical conversion wiring is not implemented yet (brief A1-2).")


__all__ = ["DecodedUtterance", "UnreadableUtterance", "decode_file", "to_canonical"]
