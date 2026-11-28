"""Exception family raised by utterance analysis.

Bad audio never raises: it becomes a record status. These exceptions cover caller
mistakes and models that cannot be loaded. Messages never contain caller-supplied
paths or audio content, and are raised ``from None`` so that no chained exception
carries them either. ``KeyboardInterrupt`` and ``MemoryError`` propagate unchanged in
inline isolation; inside the worker process they end the call as ``model_error``.
"""

from __future__ import annotations


class AnalysisError(Exception):
    """Base class of every exception raised by utterance analysis."""


class AnalysisUsageError(AnalysisError, ValueError):
    """The caller broke the call contract (argument types, values, or lifecycle)."""


class AnalysisModelUnavailableError(AnalysisError):
    """No model file exists for the requested profile; nothing is downloaded."""


class AnalysisModelLoadError(AnalysisError):
    """A model file exists but cannot be loaded, fails verification, or is incompatible.

    Verification failures include a payload whose SHA-256 differs from its provenance
    record, a safe artifact without provenance, and an ``expected_sha256`` mismatch.
    """


class AnalysisModelNotConformingError(AnalysisModelLoadError):
    """The model is a baseline and the caller did not pass ``allow_legacy_baseline``.

    Raised for legacy pickles (before anything is unpickled) and for models whose
    recorded training audio is not canonical or is unknown.
    """


__all__ = [
    "AnalysisError",
    "AnalysisModelLoadError",
    "AnalysisModelNotConformingError",
    "AnalysisModelUnavailableError",
    "AnalysisUsageError",
]
