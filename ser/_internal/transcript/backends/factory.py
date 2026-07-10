"""Factory for transcription backend adapters."""

from __future__ import annotations

from functools import lru_cache

from ser.profiles import TranscriptionBackendId
from ser.transcript.backends.base import TranscriptionBackendAdapter


@lru_cache(maxsize=2)
def _build_adapter(backend_id: TranscriptionBackendId) -> TranscriptionBackendAdapter:
    """Builds one backend adapter lazily to avoid importing unused heavy stacks."""
    if backend_id == "stable_whisper":
        from ser.transcript.backends.stable_whisper import StableWhisperAdapter

        return StableWhisperAdapter()
    if backend_id == "faster_whisper":
        from ser.transcript.backends.faster_whisper import FasterWhisperAdapter

        return FasterWhisperAdapter()
    raise KeyError(backend_id)


def resolve_transcription_backend_adapter(
    backend_id: TranscriptionBackendId,
) -> TranscriptionBackendAdapter:
    """Returns one adapter implementation for the requested backend id."""
    try:
        return _build_adapter(backend_id)
    except KeyError as err:
        raise RuntimeError(
            f"Unsupported transcription backend id configured: {backend_id!r}."
        ) from err
