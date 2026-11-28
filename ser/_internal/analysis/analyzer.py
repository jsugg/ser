"""Public utterance analyser: load a model once, analyse many utterances quietly."""

from __future__ import annotations

import os
from types import TracebackType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig, LanguageHint, ModelManifest
from ser.runtime.schema import UtteranceRecord

_NOT_IMPLEMENTED = "UtteranceAnalyzer is not implemented yet (brief A1-2)."


class _LoadedSession:
    """Verified model bytes, manifest, configuration and worker; built by ``load``."""


class UtteranceAnalyzer:
    """Analyses single utterances with one model, loaded once.

    Create instances with ``UtteranceAnalyzer.load``. ``analyze`` is synchronous and
    safe to call from several threads; calls on one instance run one at a time, and
    time spent waiting counts towards the caller's deadline.
    """

    def __init__(self, session: _LoadedSession, /) -> None:
        """Not part of the interface; use ``load``."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @classmethod
    def load(
        cls,
        *,
        profile: ProfileName,
        model_path: str | os.PathLike[str] | None = None,
        expected_sha256: str | None = None,
        allow_legacy_baseline: bool = False,
        config: AnalysisConfig | None = None,
        settings: AppConfig | None = None,
    ) -> Self:
        """Reads and verifies the model file once and starts the worker.

        Raises:
            AnalysisUsageError: Unsupported profile or invalid ``config``.
            AnalysisModelUnavailableError: No model file for ``profile``.
            AnalysisModelNotConformingError: A baseline model without
                ``allow_legacy_baseline``.
            AnalysisModelLoadError: The file is unreadable, fails verification or
                ``expected_sha256``, or the backend cannot start.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @staticmethod
    def inspect_model(
        *,
        profile: ProfileName,
        model_path: str | os.PathLike[str] | None = None,
        allow_legacy_baseline: bool = False,
        settings: AppConfig | None = None,
    ) -> ModelManifest:
        """Verifies a model file and returns its manifest without loading the model.

        A caller applies its licence allow-list to the result, then passes
        ``manifest.identity.model_sha256`` as ``expected_sha256`` to ``load``.
        Raises like ``load``.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def model_manifest(self) -> ModelManifest:
        """Manifest of the loaded model; its identity is copied into every record."""
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
        """Analyses one utterance: an audio file path, or mono float32 samples and rate.

        Bad audio yields a non-``ok`` status and a deadline yields ``timeout``. Raises
        ``AnalysisUsageError`` only for caller mistakes and after ``close``.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Waits for a call in flight, then stops the worker; idempotent.

        Later ``analyze`` calls raise ``AnalysisUsageError``.
        """
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
