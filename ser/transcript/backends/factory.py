"""Factory for transcription backend adapters."""

from __future__ import annotations

from typing import Final

from ser.profiles import TranscriptionBackendId
from ser.transcript.backends.base import TranscriptionBackendAdapter
from ser.transcript.backends.faster_whisper import FasterWhisperAdapter
from ser.transcript.backends.stable_whisper import StableWhisperAdapter

_TRANSCRIPTION_ADAPTERS: Final[
    dict[TranscriptionBackendId, TranscriptionBackendAdapter]
] = {
    "stable_whisper": StableWhisperAdapter(),
    "faster_whisper": FasterWhisperAdapter(),
}


def resolve_transcription_backend_adapter(
    backend_id: TranscriptionBackendId,
) -> TranscriptionBackendAdapter:
    """Returns one adapter implementation for the requested backend id."""
    try:
        return _TRANSCRIPTION_ADAPTERS[backend_id]
    except KeyError as err:
        raise RuntimeError(
            f"Unsupported transcription backend id configured: {backend_id!r}."
        ) from err
