# pyright: reportUnsupportedDunderAll=false
"""Public representation backend facade."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "EncodedSequence": "ser._internal.repr.backend",
    "Emotion2VecBackend": "ser._internal.repr.emotion2vec",
    "FeatureBackend": "ser._internal.repr.backend",
    "HandcraftedBackend": "ser._internal.repr.handcrafted",
    "PoolingWindow": "ser._internal.repr.backend",
    "WhisperBackend": "ser._internal.repr.hf_whisper",
    "XLSRBackend": "ser._internal.repr.hf_xlsr",
    "VectorFeatureBackend": "ser._internal.repr.backend",
    "overlap_frame_mask": "ser._internal.repr.backend",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily resolves one public representation facade export."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ser.repr' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
