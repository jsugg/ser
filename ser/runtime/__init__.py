# pyright: reportUnsupportedDunderAll=false
"""Runtime helpers for profiling, orchestration, and benchmarks."""

from __future__ import annotations

import importlib
from typing import Any

from ser.runtime.contracts import InferenceExecution, InferenceRequest
from ser.runtime.registry import (
    RuntimeCapability,
    UnsupportedProfileError,
    ensure_profile_supported,
    resolve_runtime_capability,
)

_DYNAMIC_EXPORTS: dict[str, str] = {
    "RuntimePipeline": "ser.runtime.pipeline",
    "create_runtime_pipeline": "ser.runtime.pipeline",
}

__all__ = [
    "InferenceExecution",
    "InferenceRequest",
    "RuntimeCapability",
    "RuntimePipeline",
    "UnsupportedProfileError",
    "create_runtime_pipeline",
    "ensure_profile_supported",
    "resolve_runtime_capability",
]


def __getattr__(name: str) -> Any:
    """Lazily resolves runtime facade exports that import orchestration code."""
    module_name = _DYNAMIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ser.runtime' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
