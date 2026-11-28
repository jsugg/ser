"""Public utterance analyser: load a model once, analyse many utterances quietly."""

from __future__ import annotations

import math
import os
from types import TracebackType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from ser._internal.analysis import artifact, backend
from ser._internal.analysis.errors import AnalysisUsageError
from ser.config import AppConfig, reload_settings
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig, AnalysisTargets, LanguageHint, ModelManifest
from ser.runtime.schema import UtteranceRecord

_NOT_IMPLEMENTED = "UtteranceAnalyzer is not implemented yet (brief A1-2)."


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_config(config: AnalysisConfig) -> None:
    if not isinstance(config, AnalysisConfig) or not isinstance(config.targets, AnalysisTargets):
        raise AnalysisUsageError("Invalid analysis configuration.") from None
    targets = config.targets
    positive = (
        config.min_duration_s,
        config.max_duration_s,
        config.timeout_floor_s,
        config.timeout_per_audio_second,
        config.worker_start_timeout_s,
        targets.p95_real_time_factor,
        targets.cold_start_s,
    )
    if (
        not all(_finite_number(value) and value > 0 for value in positive)
        or config.max_duration_s < config.min_duration_s
        or not _finite_number(config.silence_rms_dbfs)
        or config.silence_rms_dbfs > 0
        or config.isolation not in ("worker", "inline")
        or type(config.max_threads) is not int
        or config.max_threads < 1
        or type(config.worker_nice) is not int
        or not 0 <= config.worker_nice <= 19
        or type(targets.steady_memory_bytes) is not int
        or targets.steady_memory_bytes < 1
    ):
        raise AnalysisUsageError("Invalid analysis configuration.") from None


def _loader(profile: ProfileName) -> backend.BackendLoader:
    loader = backend.backend_loaders().get(profile)
    if loader is None:
        raise AnalysisUsageError("No analysis backend serves the requested profile.") from None
    return loader


def _open_model(
    *,
    profile: ProfileName,
    model_path: str | os.PathLike[str] | None,
    expected_sha256: str | None,
    allow_legacy_baseline: bool,
    settings: AppConfig | None,
) -> tuple[ModelManifest, artifact.OpenedArtifact]:
    if model_path is None:
        model_path = (settings if settings is not None else reload_settings()).models.model_file
    return artifact.open_model(
        artifact.read_model_file(model_path),
        profile=profile,
        expected_sha256=expected_sha256,
        allow_legacy_baseline=allow_legacy_baseline,
    )


class _LoadedSession:
    """Verified model bytes, manifest, configuration and worker supervisor; built by
    ``load``."""


class UtteranceAnalyzer:
    """Analyses single utterances with one model, loaded once.

    Create instances with ``UtteranceAnalyzer.load``. ``analyze`` is synchronous and
    safe to call from several threads; calls on one instance run one at a time in
    arrival order, and time spent waiting counts towards the caller's deadline.
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
        _validate_config(config if config is not None else AnalysisConfig())
        _loader(profile)
        _open_model(
            profile=profile,
            model_path=model_path,
            expected_sha256=expected_sha256,
            allow_legacy_baseline=allow_legacy_baseline,
            settings=settings,
        )
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
        _loader(profile)
        manifest, _ = _open_model(
            profile=profile,
            model_path=model_path,
            expected_sha256=None,
            allow_legacy_baseline=allow_legacy_baseline,
            settings=settings,
        )
        return manifest

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

        Argument type, dtype, shape and rate are checked first and raise
        ``AnalysisUsageError``. Bad audio yields a non-``ok`` status; a deadline
        yields ``timeout`` no later than the deadline, and a result fully received
        before it is returned. Raises ``AnalysisClosedError`` if ``close()`` began
        before this call started.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Lets the call in flight finish or reach its deadline, then stops the worker.

        Queued calls and later calls raise ``AnalysisClosedError``. Idempotent;
        leaving a ``with`` block calls it.
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
