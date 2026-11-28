"""Public utterance analyser: load a model once, analyse many utterances quietly."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from ser._internal.analysis.backend import AnalysisBackend
from ser.config import AppConfig
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig, LanguageHint, ModelIdentity
from ser.runtime.schema import UtteranceRecord

_NOT_IMPLEMENTED = "UtteranceAnalyzer is not implemented yet (brief A1-2)."


class UtteranceAnalyzer:
    """Analyses single utterances with one model loaded at construction.

    Create instances with ``UtteranceAnalyzer.load``. ``analyze`` is synchronous and
    safe to call from several threads; calls on one instance run one at a time.
    """

    def __init__(self, backend: AnalysisBackend, config: AnalysisConfig) -> None:
        """Wraps an already loaded backend; use ``load`` instead."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @classmethod
    def load(
        cls,
        *,
        profile: ProfileName,
        model_path: str | os.PathLike[str] | None = None,
        config: AnalysisConfig | None = None,
        settings: AppConfig | None = None,
    ) -> Self:
        """Reads and verifies the model file once.

        Raises:
            AnalysisUsageError: Unsupported profile or invalid ``config``.
            AnalysisModelUnavailableError: No model file for ``profile``.
            AnalysisModelLoadError: The model file is unreadable or incompatible.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def model_identity(self) -> ModelIdentity:
        """Identity fields copied into every record."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @overload
    def analyze(
        self,
        audio: str | os.PathLike[str],
        *,
        language: LanguageHint | None = None,
    ) -> UtteranceRecord: ...

    @overload
    def analyze(
        self,
        audio: NDArray[np.float32],
        *,
        sample_rate_hz: int,
        language: LanguageHint | None = None,
    ) -> UtteranceRecord: ...

    def analyze(
        self,
        audio: str | os.PathLike[str] | NDArray[np.float32],
        *,
        sample_rate_hz: int | None = None,
        language: LanguageHint | None = None,
    ) -> UtteranceRecord:
        """Analyses one utterance: a WAV path, or mono float32 samples and their rate.

        Bad audio yields a non-``ok`` status. Raises ``AnalysisUsageError`` only for
        caller mistakes and after ``close``.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Releases the model; later ``analyze`` calls raise ``AnalysisUsageError``."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["UtteranceAnalyzer"]
