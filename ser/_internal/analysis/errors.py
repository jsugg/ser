"""Exception family raised by utterance analysis.

Bad audio never raises: it becomes a record status. These exceptions cover caller
mistakes and models that cannot be loaded. Messages never contain caller-supplied
paths or audio content.
"""

from __future__ import annotations


class AnalysisError(Exception):
    """Base class of every exception raised by utterance analysis."""


class AnalysisUsageError(AnalysisError, ValueError):
    """The caller broke the call contract (argument types, values, or lifecycle)."""


class AnalysisModelUnavailableError(AnalysisError):
    """No model file exists for the requested profile; nothing is downloaded."""


class AnalysisModelLoadError(AnalysisError):
    """A model file exists but cannot be loaded or is incompatible with the profile."""


__all__ = [
    "AnalysisError",
    "AnalysisModelLoadError",
    "AnalysisModelUnavailableError",
    "AnalysisUsageError",
]
