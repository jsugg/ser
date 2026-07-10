# pyright: reportUnsupportedDunderAll=false
"""Model facades for artifact loading, profile runtimes, and training entrypoints.

Only the modules listed in ``__all__`` are intended facades; every other module
in this package is implementation detail. Facade modules are imported lazily to
keep package import free of heavy model dependencies.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = [
    "emotion_model",
    "profile_runtime",
    "training_entrypoints",
]


def __getattr__(name: str) -> ModuleType:
    """Lazily imports one intended facade module on first attribute access."""
    if name in __all__:
        return importlib.import_module(f"ser.models.{name}")
    raise AttributeError(f"module 'ser.models' has no attribute {name!r}")
