# pyright: reportUnsupportedDunderAll=false
"""Public classifier-head facade."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULE = "ser._internal.heads.torch_head"
__all__ = ["build_torch_mlp_head", "forward_torch_head"]


def __getattr__(name: str) -> Any:
    """Lazily resolves one public classifier-head export."""
    if name not in __all__:
        raise AttributeError(f"module 'ser.heads' has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORT_MODULE), name)
    globals()[name] = value
    return value
