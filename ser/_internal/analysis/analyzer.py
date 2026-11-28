"""Public utterance analyser: load a model once, analyse many utterances quietly."""

from __future__ import annotations

import logging
import math
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from threading import Condition
from types import TracebackType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

from ser import __version__
from ser._internal.analysis import artifact, backend, worker
from ser._internal.analysis.errors import (
    AnalysisClosedError,
    AnalysisModelLoadError,
    AnalysisUsageError,
)
from ser._internal.analysis.pipeline import (
    AnalysisJob,
    PipelineOutcome,
    analysis_deadline_s,
    run_pipeline,
    top_label,
)
from ser.config import AppConfig, reload_settings
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig, AnalysisTargets, LanguageHint, ModelManifest
from ser.runtime.schema import UTTERANCE_SCHEMA_VERSION, UtteranceFrame, UtteranceRecord

_LOGGER = logging.getLogger(__name__)


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


class _InlineSession:
    def __init__(self, request: backend.BackendLoadRequest, loader: backend.BackendLoader) -> None:
        try:
            self._backend = loader(request)
        except MemoryError:
            raise
        except Exception:
            raise AnalysisModelLoadError("The analysis backend could not start.") from None
        self._labels = request.manifest.identity.labels
        self._condition = Condition()
        self._queue: deque[object] = deque()
        self._busy = self._closing = self._closed = False

    def check_open(self) -> None:
        with self._condition:
            if self._closing:
                raise AnalysisClosedError("The analyzer is closed.") from None

    def run(
        self,
        job: AnalysisJob,
        *,
        entry_monotonic_s: float,
        known_duration_s: float | None,
        config: AnalysisConfig,
    ) -> PipelineOutcome:
        deadline = entry_monotonic_s + analysis_deadline_s(config, known_duration_s)
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            try:
                while True:
                    self.check_open()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return PipelineOutcome(
                            "timeout", None, known_duration_s, job.sample_rate_hz, 0, 0
                        )
                    if not self._busy and self._queue[0] is ticket:
                        self._busy = True
                        break
                    self._condition.wait(timeout=remaining)
            finally:
                self._queue.remove(ticket)
                self._condition.notify_all()

        def extend(duration_s: float, _sample_rate_hz: int) -> None:
            nonlocal deadline
            if time.monotonic() < deadline:
                deadline = entry_monotonic_s + analysis_deadline_s(config, duration_s)

        try:
            return run_pipeline(
                self._backend,
                job,
                config,
                deadline_monotonic_s=lambda: deadline,
                report_duration=extend,
                labels=self._labels,
            )
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closing:
                self._condition.wait_for(lambda: self._closed)
                return
            self._closing = True
            self._condition.notify_all()
            self._condition.wait_for(lambda: not self._busy)
        try:
            self._backend.close()
        except MemoryError:
            raise
        except Exception:
            _LOGGER.debug("Could not close the inline analysis backend.")
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()


@dataclass(frozen=True)
class _LoadedSession:
    manifest: ModelManifest
    config: AnalysisConfig
    runner: _InlineSession | worker.WorkerSupervisor


def _record(manifest: ModelManifest, outcome: PipelineOutcome) -> UtteranceRecord:
    identity = manifest.identity
    labels: dict[str, float] | None = None
    name: str | None = None
    confidence: float | None = None
    frames: list[UtteranceFrame] | None = None
    if outcome.status == "ok" and outcome.scores is not None:
        labels = dict(outcome.scores.labels)
        name, confidence = top_label(labels)
        frames = [
            {"start_s": frame.start_s, "end_s": frame.end_s, "labels": dict(frame.labels)}
            for frame in outcome.scores.frames
        ]
    return {
        "schema_version": UTTERANCE_SCHEMA_VERSION,
        "ser_version": __version__,
        "model_id": identity.model_id,
        "model_sha256": identity.model_sha256,
        "label_set_id": identity.label_set_id,
        "labels": labels,
        "top_label": name,
        "confidence": confidence,
        "frames": frames,
        "dimensions": None,
        "audio": {
            "duration_s": outcome.duration_s,
            "input_sample_rate_hz": outcome.input_sample_rate_hz,
        },
        "status": outcome.status,
        "timing_ms": {"features": outcome.features_ms, "predict": outcome.predict_ms},
    }


def _job(
    audio: str | os.PathLike[str] | NDArray[np.float32],
    rate: int | None,
    language: LanguageHint | None,
) -> AnalysisJob:
    if language is not None and (
        not isinstance(language, str) or language not in ("en", "es", "pt")
    ):
        raise AnalysisUsageError("Unsupported language hint.") from None
    if isinstance(audio, (str, os.PathLike)):
        if rate is not None:
            raise AnalysisUsageError("A path must not specify a sample rate.") from None
        try:
            path = os.fspath(audio)
            if isinstance(path, str):
                return AnalysisJob(os.path.abspath(path) if path else path, None, None, language)
        except (TypeError, ValueError, OSError):
            pass
        raise AnalysisUsageError("Invalid audio path.") from None
    if (
        not isinstance(audio, np.ndarray)
        or audio.dtype != np.float32
        or audio.ndim != 1
        or type(rate) is not int
        or rate <= 0
    ):
        raise AnalysisUsageError(
            "Expected mono float32 samples and a positive integer rate."
        ) from None
    return AnalysisJob(None, audio, rate, language)


class UtteranceAnalyzer:
    """Analyses single utterances with one model, loaded once.

    Create instances with ``UtteranceAnalyzer.load``. ``analyze`` is synchronous and
    safe to call from several threads; calls on one instance run one at a time in
    arrival order, and time spent waiting counts towards the caller's deadline.
    """

    def __init__(self, session: _LoadedSession, /) -> None:
        """Not part of the interface; use ``load``."""
        self._session = session
        self._finalizer: weakref.finalize | None = None

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

        In scripts, call worker-mode ``load`` under ``if __name__ == "__main__":``
        so the spawned worker can import the main module safely.

        Raises:
            AnalysisUsageError: Unsupported profile or invalid ``config``.
            AnalysisModelUnavailableError: No model file for ``profile``.
            AnalysisModelNotConformingError: A baseline model without
                ``allow_legacy_baseline``.
            AnalysisModelLoadError: The file is unreadable, fails verification or
                ``expected_sha256``, or the backend cannot start.
        """
        config = config if config is not None else AnalysisConfig()
        _validate_config(config)
        loader = _loader(profile)
        settings = settings if settings is not None else reload_settings()
        manifest, opened = _open_model(
            profile=profile,
            model_path=model_path,
            expected_sha256=expected_sha256,
            allow_legacy_baseline=allow_legacy_baseline,
            settings=settings,
        )
        request = backend.BackendLoadRequest(
            manifest.artifact_format,
            opened.payload,
            manifest,
            config.max_threads if config.isolation == "worker" else None,
            feature_flags=settings.feature_flags,
            enabled_features=opened.metadata.enabled_features,
        )
        worker_runner: worker.WorkerSupervisor | None = None
        if config.isolation == "inline":
            runner: _InlineSession | worker.WorkerSupervisor = _InlineSession(request, loader)
        else:
            worker_runner = worker.WorkerSupervisor(
                request,
                loader,
                worker.WorkerSettings(
                    config.max_threads, config.worker_nice, config.worker_start_timeout_s
                ),
            )
            runner = worker_runner
        analyzer = cls(_LoadedSession(manifest, config, runner))
        if worker_runner is not None:
            analyzer._finalizer = weakref.finalize(analyzer, worker_runner.close_nowait)
        if opened.legacy_version_mismatch:
            _LOGGER.warning(
                "Legacy analysis model was saved with a different scikit-learn version."
            )
        return analyzer

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
        return self._session.manifest

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
        entry = time.monotonic()
        self._session.runner.check_open()
        job = _job(audio, sample_rate_hz, language)
        duration = (
            len(job.samples) / job.sample_rate_hz
            if job.samples is not None and job.sample_rate_hz is not None
            else None
        )
        outcome = self._session.runner.run(
            job,
            entry_monotonic_s=entry,
            known_duration_s=duration,
            config=self._session.config,
        )
        return _record(self.model_manifest, outcome)

    def close(self) -> None:
        """Lets the call in flight finish or reach its deadline, then stops the worker.

        Queued calls and later calls raise ``AnalysisClosedError``. Idempotent;
        leaving a ``with`` block calls it.
        """
        self._session.runner.close()
        if self._finalizer is not None:
            self._finalizer.detach()

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
