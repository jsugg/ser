"""Regression tests for the compatibility and lifecycle review findings."""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
import pickle
import selectors
import signal
import subprocess
import sys
import time
import warnings
import weakref
from collections.abc import Callable
from multiprocessing.connection import wait
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
from ser._internal.analysis.pipeline import AnalysisJob, PipelineOutcome
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


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize(
    ("name", "delegate", "args", "kwargs"),
    [
        ("configure_dataset_consents", "_data_api", (), {}),
        ("infer", "_runtime_api", ("unused.wav",), {}),
        ("list_dataset_registry_health_issues", "_data_api", (), {}),
        ("list_datasets", "_data_api", (), {}),
        ("list_profiles", "_runtime_api", (), {}),
        ("list_registered_datasets", "_data_api", (), {}),
        ("load_profile", "_runtime_api", ("fast",), {}),
        ("prepare_dataset", "_data_api", (), {"dataset_id": "unused"}),
        ("run_startup_preflight", "_diagnostics_api", (), {"include_transcription_checks": False}),
        ("show_dataset_consents", "_data_api", (), {}),
        ("train", "_runtime_api", (), {}),
    ],
)
def test_workflow_logging_defaults_preserve_caller_configuration(
    configured: bool, name: str, delegate: str, args: tuple[str, ...], kwargs: dict[str, object]
) -> None:
    probe = """
import json, logging, os, sys
os.environ['LOG_LEVEL'] = 'INFO'
root = logging.getLogger()
name, delegate, args, kwargs, configured = json.loads(sys.argv[1])
if configured:
    root.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter('CALLER %(message)s'))
    root.addHandler(handler)
def snapshot():
    return root.level, [(h, h.level, h.formatter) for h in root.handlers]
before = snapshot()
from ser import api
assert snapshot() == before

def emit(*args, **kwargs):
    log = logging.getLogger('ser.workflow_probe')
    log.warning('workflow warning')
    log.error('workflow error')

# Exercise the public boundary without running training, downloads or consent writes.
setattr(getattr(api, delegate), name, emit)
getattr(api, name)(*args, **kwargs)
if configured:
    assert snapshot() == before
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, json.dumps([name, delegate, args, kwargs, configured])],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout
    if configured:
        assert result.stderr == "CALLER workflow error\n"
    else:
        assert [line.partition(" - ")[2] for line in result.stderr.splitlines()] == [
            "WARNING - ser.workflow_probe - workflow warning",
            "ERROR - ser.workflow_probe - workflow error",
        ]


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
    assert [w.category for w in caught] == [FutureWarning]
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
    assert [w.category for w in caught] == [FutureWarning]
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


def _load_blocking(
    request: worker.BackendLoadRequest, *, holds_gil: bool, watcher_only: bool
) -> _scripted_backend.ScriptedBackend:
    backend = _scripted_backend.load_scripted(request)
    original = backend.score

    def score(
        features: object,
        *,
        language: _scripted_backend.LanguageHint | None,
        deadline_monotonic_s: float,
    ) -> _scripted_backend.BackendScores:
        if language == "es":
            pause = ctypes.PyDLL(None).pause if holds_gil else signal.pause
            if watcher_only:
                # Ensure only the parent-pipe watcher can stop this worker.
                signal.alarm(0)
            os.write(1, f"{os.getpid()}\n".encode())
            pause()
        return original(features, language=language, deadline_monotonic_s=deadline_monotonic_s)

    backend.score = score  # type: ignore[method-assign]
    return backend


def _portable_loader(holds_gil: bool) -> worker.BackendLoader:
    from functools import partial

    worker._arm_linux_parent_death = lambda expected_parent_pid: None
    return partial(_load_blocking, holds_gil=holds_gil, watcher_only=not holds_gil)


class _PortableLoader:
    def __init__(self, holds_gil: bool) -> None:
        self._holds_gil = holds_gil

    def __reduce__(self) -> tuple[Callable[[bool], worker.BackendLoader], tuple[bool]]:
        return _portable_loader, (self._holds_gil,)


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


@pytest.mark.skipif(not hasattr(signal, "alarm"), reason="requires POSIX alarm")
def test_job_alarm_tracks_path_deadline_and_disarms_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from multiprocessing import Pipe

    alarms: list[int] = []
    monkeypatch.setattr(signal, "alarm", alarms.append)
    monkeypatch.setattr(
        worker, "time", SimpleNamespace(monotonic=lambda: 10.25, monotonic_ns=lambda: 0)
    )

    def fail_after_source(
        *args: object, report_duration: Callable[[float, int], None], **kwargs: object
    ) -> PipelineOutcome:
        report_duration(4.0, 16_000)
        raise RuntimeError("job failed")

    monkeypatch.setattr(worker, "run_pipeline", fail_after_source)
    parent, child = Pipe()
    with parent, child, pytest.raises(RuntimeError, match="job failed"):
        worker._run_work(
            child,
            Mock(spec=worker.AnalysisBackend),
            ("a", "b", "c"),
            worker._Work(
                AnalysisJob("clip.wav", None, None, None),
                10.0,
                11.0,
                AnalysisConfig(timeout_floor_s=1.0, timeout_per_audio_second=2.0),
            ),
        )
    assert alarms == [2, 9, 0]


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGKILL and pipe selection")
@pytest.mark.parametrize(
    ("holds_gil", "disable_linux_guard"),
    [
        (False, True),
        (True, True),
        pytest.param(
            True, False, marks=pytest.mark.skipif(sys.platform != "linux", reason="Linux guard")
        ),
    ],
    ids=["watcher", "alarm", "linux-guard"],
)
def test_busy_worker_exits_when_parent_is_killed(
    holds_gil: bool, disable_linux_guard: bool, record_property: Callable[[str, object], None]
) -> None:
    parent_code = """
from functools import partial
from tests.suites.unit.analysis.test_review_regressions import _load_blocking, _PortableLoader, tone
from ser._internal.analysis.backend import BackendLoadRequest
from ser._internal.analysis.pipeline import AnalysisJob
from ser._internal.analysis.worker import WorkerSettings, WorkerSupervisor
from ser.api import AnalysisConfig, ModelIdentity, ModelManifest
import sys, time
identity = ModelIdentity('fast', 'test', '0' * 64, 'test', ('a', 'b', 'c'))
manifest = ModelManifest(identity, 'safe', None, None, False)
request = BackendLoadRequest('safe', b'{}', manifest, 2)
holds_gil = sys.argv[1] == 'True'
disable_linux_guard = sys.argv[2] == 'True'
loader = partial(_load_blocking, holds_gil=holds_gil, watcher_only=not holds_gil)
if disable_linux_guard:
    loader = _PortableLoader(holds_gil)
runner = WorkerSupervisor(request, loader, WorkerSettings(2, 19, 10.0))
runner.run(AnalysisJob(None, tone(0.5), 16000, 'es'),
           entry_monotonic_s=time.monotonic(), known_duration_s=0.5,
           config=AnalysisConfig(timeout_floor_s=1.5 if disable_linux_guard and holds_gil else 30.0))
"""
    with subprocess.Popen(
        [sys.executable, "-c", parent_code, str(holds_gil), str(disable_linux_guard)],
        stdout=subprocess.PIPE,
    ) as parent:
        assert parent.stdout is not None
        worker_pid = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(parent.stdout, selectors.EVENT_READ)
                assert selector.select(12), "worker did not enter scoring"
                worker_pid = int(parent.stdout.readline())
            assert parent.poll() is None
            parent.kill()
            started = time.monotonic()
            parent.communicate(timeout=4)
            record_property("parent_kill_to_worker_exit_seconds", time.monotonic() - started)
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
