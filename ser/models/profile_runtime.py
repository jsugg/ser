# pyright: reportUnsupportedDunderAll=false
"""Public profile-runtime facade."""

from __future__ import annotations

import importlib
from typing import Any

_IMPL_MODULE = "ser._internal.models.profile_runtime"
_EXPORT_NAMES: tuple[str, ...] = (
    "ACCURATE_BACKEND_ID",
    "ACCURATE_MODEL_ID",
    "ACCURATE_POOLING_STRATEGY",
    "ACCURATE_PROFILE_ID",
    "ACCURATE_RESEARCH_BACKEND_ID",
    "ACCURATE_RESEARCH_MODEL_ID",
    "ACCURATE_RESEARCH_PROFILE_ID",
    "MEDIUM_BACKEND_ID",
    "MEDIUM_FRAME_SIZE_SECONDS",
    "MEDIUM_FRAME_STRIDE_SECONDS",
    "MEDIUM_MODEL_ID",
    "MEDIUM_POOLING_STRATEGY",
    "MEDIUM_PROFILE_ID",
    "build_accurate_backend_for_settings",
    "build_accurate_research_backend_for_settings",
    "build_medium_backend_for_settings",
    "resolve_accurate_model_id",
    "resolve_accurate_research_model_id",
    "resolve_medium_model_id",
    "resolve_model_id_from_settings",
    "resolve_runtime_selectors_for_backend_id",
)
__all__ = list(_EXPORT_NAMES)


def __getattr__(name: str) -> Any:
    """Lazily resolves one public profile-runtime export."""
    if name not in _EXPORT_NAMES:
        raise AttributeError(
            f"module 'ser._internal.models.profile_runtime' has no attribute {name!r}"
        )
    value = getattr(importlib.import_module(_IMPL_MODULE), name)
    globals()[name] = value
    return value
