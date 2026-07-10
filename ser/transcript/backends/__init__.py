# pyright: reportUnsupportedDunderAll=false
"""Public transcription backend facade package."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MODULES: dict[str, str] = {
    "BackendRuntimeRequest": "ser.transcript.backends.base",
    "CompatibilityIssue": "ser.transcript.backends.base",
    "CompatibilityIssueImpact": "ser.transcript.backends.base",
    "CompatibilityReport": "ser.transcript.backends.base",
    "TranscriptionBackendAdapter": "ser.transcript.backends.base",
    "resolve_transcription_backend_adapter": "ser._internal.transcript.backends.factory",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily resolves one public backend facade export."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ser.transcript.backends' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
