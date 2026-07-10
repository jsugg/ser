# pyright: reportUnsupportedDunderAll=false
"""Public feature extraction facade."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULE = "ser._internal.features.feature_extractor"
__all__ = [
    "FeatureFrame",
    "extract_feature",
    "extract_feature_frames",
    "extract_feature_from_signal",
    "extended_extract_feature",
]


def __getattr__(name: str) -> Any:
    """Lazily resolves one public feature facade export."""
    if name not in __all__:
        raise AttributeError(f"module 'ser.features' has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORT_MODULE), name)
    globals()[name] = value
    return value
