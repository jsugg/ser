# pyright: reportUnsupportedDunderAll=false
"""Public data package facade with curated training data exports."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "DataSplit": "ser._internal.data.data_loader",
    "EmbeddingCache": "ser._internal.data.embedding_cache",
    "EmbeddingCacheEntry": "ser._internal.data.embedding_cache",
    "LabeledAudioSample": "ser._internal.data.data_loader",
    "Utterance": "ser._internal.data.manifest",
    "load_data": "ser._internal.data.data_loader",
    "load_labeled_audio_paths": "ser._internal.data.data_loader",
    "load_utterances": "ser._internal.data.data_loader",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily resolves one curated data facade export."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ser.data' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
