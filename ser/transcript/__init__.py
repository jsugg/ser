# pyright: reportUnsupportedDunderAll=false
"""Public transcript extraction facade."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULE = "ser.transcript.transcript_extractor"
__all__ = [
    "extract_transcript",
    "TranscriptionError",
    "TranscriptionProfile",
    "load_whisper_model",
    "transcribe_with_model",
]


def __getattr__(name: str) -> Any:
    """Lazily resolves one transcript facade export."""
    if name not in __all__:
        raise AttributeError(f"module 'ser.transcript' has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORT_MODULE), name)
    globals()[name] = value
    return value
