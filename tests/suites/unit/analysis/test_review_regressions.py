"""Regression tests for the compatibility and lifecycle review findings."""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import pickle
import selectors
import signal
import subprocess
import sys
import warnings
import weakref
from multiprocessing.connection import wait
from pathlib import Path

import numpy as np
import pytest
import sklearn
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier
from tests.suites.unit.analysis import _scripted_backend
from tests.suites.unit.analysis._contract import INLINE, load_artifact
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone, write_model
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser._internal.analysis import worker
from ser.api import AnalysisConfig, UtteranceAnalyzer


@pytest.mark.parametrize("module", ["ser.api", "ser._internal.analysis.worker"])
@pytest.mark.parametrize("configured", [False, True])
def test_fresh_import_preserves_application_logging(module: str, configured: bool) -> None:
    probe = """
import importlib, logging, sys
root = logging.getLogger()
other = logging.getLogger('application')
if sys.argv[2] == 'True':
    root.setLevel(logging.ERROR)
    handler = logging.StreamHandler()
    handler.setLevel(logging.CRITICAL)
    handler.setFormatter(logging.Formatter('application: %(message)s'))
    root.addHandler(handler)
def snapshot():
    return [(log.level, [(h, h.level, h.formatter) for h in log.handlers])
            for log in (root, other)]
before = snapshot()
importlib.import_module(sys.argv[1])
assert snapshot() == before
assert any(isinstance(h, logging.NullHandler) for h in logging.getLogger('ser').handlers)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, module, str(configured)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (result.stdout, result.stderr) == ("", "")


@pytest.mark.parametrize("config", [INLINE, AnalysisConfig()], ids=["inline", "worker"])
def test_legacy_version_mismatch_is_one_content_free_log(
    tmp_path: Path,
    fast_model_path: Path,
    capfd: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    config: AnalysisConfig,
) -> None:
    current = sklearn.__version__.encode()
    old = bytes(57 if char == 48 else 48 if 48 < char <= 57 else char for char in current)
    original = fast_model_path.read_bytes()
    assert old != current and current in original
    path = tmp_path / "private-model-token.pkl"
    path.write_bytes(original.replace(current, old))
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        warnings.simplefilter("always")
        with UtteranceAnalyzer.load(
            profile="fast", model_path=path, allow_legacy_baseline=True, config=config
        ) as analyzer:
            assert analyzer.analyze(tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "ok"
    assert not caught
    assert capfd.readouterr() == ("", "")
    assert caplog.messages == [
        "Legacy analysis model was saved with a different scikit-learn version."
    ]


class _WarningClassifier(MLPClassifier):
    def predict_proba(self, X: object) -> NDArray[np.float64]:
        warnings.warn("private scoring token", stacklevel=2)
        return np.asarray(super().predict_proba(X), dtype=np.float64)


@pytest.mark.parametrize("config", [INLINE, AnalysisConfig()], ids=["inline", "worker"])
def test_legacy_scoring_warnings_do_not_escape(
    tmp_path: Path,
    fast_feature_size: int,
    capfd: pytest.CaptureFixture[str],
    config: AnalysisConfig,
) -> None:
    path = write_model(
        tmp_path / "warn.pkl", feature_size=fast_feature_size, classifier_type=_WarningClassifier
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # A spawned interpreter otherwise inherits no pytest warning configuration.
        with UtteranceAnalyzer.load(
            profile="fast", model_path=path, allow_legacy_baseline=True, config=config
        ) as analyzer:
            assert analyzer.analyze(tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "ok"
    assert not caught
    assert capfd.readouterr() == ("", "")


@pytest.mark.usefixtures("scripted")
@pytest.mark.skipif(os.name != "posix", reason="duplicates a POSIX worker sentinel")
def test_collecting_analyzer_reaps_its_real_worker(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser")
    analyzer = UtteranceAnalyzer.load(profile="fast", model_path=path)
    runner = analyzer._session.runner
    assert isinstance(runner, worker.WorkerSupervisor)
    child = runner._worker
    assert child is not None
    reference = weakref.ref(analyzer)
    try:
        with os.fdopen(os.dup(child._process.sentinel), "rb") as sentinel:
            del analyzer
            gc.collect()
            assert reference() is None
            assert wait([sentinel.fileno()], timeout=3), "worker survived collection"
    finally:
        runner.close()


@pytest.mark.usefixtures("scripted")
def test_unexpected_restart_exception_sets_failed_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_start = worker.start_worker
    starts = 0

    def start(
        request: worker.BackendLoadRequest,
        loader: worker.BackendLoader,
        settings: worker.WorkerSettings,
    ) -> worker.AnalysisWorker:
        nonlocal starts
        starts += 1
        if starts == 2:
            raise pickle.PicklingError("private start token")
        return real_start(request, loader, settings)

    monkeypatch.setattr(worker, "start_worker", start)
    with load_artifact(
        write_artifact(tmp_path / "model.ser"), AnalysisConfig(timeout_floor_s=2.0)
    ) as analyzer:
        runner = analyzer._session.runner
        assert isinstance(runner, worker.WorkerSupervisor)
        assert runner._worker is not None
        runner._worker.kill()
        with runner._condition:
            assert runner._condition.wait_for(
                lambda: not runner._worker or not runner._worker.alive, 3
            )
        assert (
            analyzer.analyze(tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "model_error"
        )
        assert runner.state == "failed"
        assert analyzer.analyze(tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)["status"] == "ok"


def _load_blocking(request: worker.BackendLoadRequest) -> _scripted_backend.ScriptedBackend:
    backend = _scripted_backend.load_scripted(request)
    original = backend.score

    def score(
        features: object,
        *,
        language: _scripted_backend.LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> _scripted_backend.BackendScores:
        if language in ("es", "de"):
            pause = ctypes.PyDLL(None).pause if language == "de" else signal.pause
            os.write(1, f"{os.getpid()}\n".encode())
            pause()
        return original(features, language=language, deadline_monotonic_s=deadline_monotonic_s)

    backend.score = score  # type: ignore[method-assign]
    return backend


class _WarmupBackend(_scripted_backend.ScriptedBackend):
    warmed = False

    def warm_up(self, *, deadline_monotonic_s: float) -> None:
        audio = _scripted_backend.CanonicalAudio(tone(0.5), 16_000, 16_000, 0.5)
        features = self.extract(audio, deadline_monotonic_s=deadline_monotonic_s)
        super().score(features, language=None, deadline_monotonic_s=deadline_monotonic_s)
        self.warmed = True

    def score(
        self,
        features: object,
        *,
        language: _scripted_backend.LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> _scripted_backend.BackendScores:
        if language == "es":
            signal.pause()
        if not self.warmed:
            raise RuntimeError("The scorer is still cold.")
        return super().score(features, language=language, deadline_monotonic_s=deadline_monotonic_s)


def _load_warmup(request: worker.BackendLoadRequest) -> _WarmupBackend:
    return _WarmupBackend({}, request, None)


@pytest.mark.usefixtures("scripted")
def test_restart_is_ready_to_score_before_admitting_next_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ser._internal.analysis import backend

    monkeypatch.setattr(backend, "backend_loaders", lambda: {"fast": _load_warmup})
    config = AnalysisConfig(timeout_floor_s=0.5, timeout_per_audio_second=0.1)
    with load_artifact(write_artifact(tmp_path / "model.ser"), config) as analyzer:
        samples = tone(0.5)
        assert (
            analyzer.analyze(samples, sample_rate_hz=16_000, language="es")["status"] == "timeout"
        )
        runner = analyzer._session.runner
        assert isinstance(runner, worker.WorkerSupervisor)
        with runner._condition:
            assert runner._condition.wait_for(lambda: runner.state == "ready", 8)
        assert analyzer.analyze(samples, sample_rate_hz=16_000)["status"] == "ok"
        assert analyzer.analyze(samples, sample_rate_hz=16_000)["status"] == "ok"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGKILL and pipe selection")
@pytest.mark.parametrize(
    "language",
    [
        "es",
        pytest.param("de", marks=pytest.mark.skipif(sys.platform != "linux", reason="Linux guard")),
    ],
    ids=["releases-gil", "holds-gil"],
)
def test_busy_worker_exits_when_parent_is_killed(language: str) -> None:
    parent_code = """
from tests.suites.unit.analysis.test_review_regressions import _load_blocking, tone
from ser._internal.analysis.backend import BackendLoadRequest
from ser._internal.analysis.pipeline import AnalysisJob
from ser._internal.analysis.worker import WorkerSettings, WorkerSupervisor
from ser.api import AnalysisConfig, ModelIdentity, ModelManifest
import sys, time
identity = ModelIdentity('fast', 'test', '0' * 64, 'test', ('a', 'b', 'c'))
manifest = ModelManifest(identity, 'safe', None, None, False)
request = BackendLoadRequest('safe', b'{}', manifest, 2)
runner = WorkerSupervisor(request, _load_blocking, WorkerSettings(2, 19, 10.0))
runner.run(AnalysisJob(None, tone(0.5), 16000, sys.argv[1]),
           entry_monotonic_s=time.monotonic(), known_duration_s=0.5,
           config=AnalysisConfig(timeout_floor_s=30.0))
"""
    with subprocess.Popen(
        [sys.executable, "-c", parent_code, language], stdout=subprocess.PIPE
    ) as parent:
        assert parent.stdout is not None
        worker_pid = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(parent.stdout, selectors.EVENT_READ)
                assert selector.select(12), "worker did not enter scoring"
                worker_pid = int(parent.stdout.readline())
            parent.kill()
            parent.communicate(timeout=4)
            worker_pid = None
        finally:
            if parent.poll() is None:
                parent.kill()
            if worker_pid is not None:
                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            parent.communicate(timeout=5)
