"""Exercise the public analyzer's supervisor without scheduler-dependent sleeps."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest
from tests.suites.unit.analysis._contract import load_artifact
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone
from tests.suites.unit.analysis._synthetic_artifacts import write_artifact

from ser._internal.analysis import analyzer as analyzer_module
from ser._internal.analysis import worker as worker_module
from ser._internal.analysis.pipeline import PipelineOutcome
from ser._internal.analysis.worker import WorkerEvent, WorkerState, WorkerSupervisor, _Work
from ser.api import AnalysisClosedError, AnalysisConfig, AnalysisModelLoadError, UtteranceRecord

_WAIT = 4.0
_CONFIG = AnalysisConfig(timeout_floor_s=1.0, timeout_per_audio_second=2.0)


class Clock:
    def __init__(self) -> None:
        self.now = 100.0
        self.notify: Callable[[], None] = lambda: None

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.notify()


class StartGate:
    def __init__(self) -> None:
        self.entered, self.release = Event(), Event()
        self.fail = False


class GatedWorker:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.jobs: Queue[_Work] = Queue()
        self.events: deque[WorkerEvent] = deque()
        self.stop_entered, self.stop_release = Event(), Event()
        self.stop_release.set()
        self.alive = True
        self.notify: Callable[[], None] = lambda: None

    def notify_on_event(self, notify: Callable[[], None]) -> None:
        self.notify = notify

    def submit(self, job: _Work) -> None:
        self.jobs.put_nowait(job)

    def next_event(self, *, until_monotonic_s: float) -> WorkerEvent | None:
        return self.events.popleft() if self.events else None

    def stop(self, *, graceful: bool) -> None:
        self.stop_entered.set()
        assert self.stop_release.wait(_WAIT)
        self.alive = False

    def die(self) -> None:
        self.alive = False
        self.notify()

    def started(self) -> None:
        self.events.append(
            WorkerEvent(
                "started", stage_started_ns=round(self.clock() * 1e9), received_at=self.clock()
            )
        )
        self.notify()

    def outcome(self, duration: float, *, stamp: float | None = None) -> None:
        self.events.append(
            WorkerEvent(
                "outcome",
                outcome=PipelineOutcome("silent", None, duration, CANONICAL_RATE_HZ, 22, 0),
                received_at=self.clock() if stamp is None else stamp,
            )
        )
        self.notify()


class Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.clock = Clock()
        self.children = [GatedWorker(self.clock) for _ in range(4)]
        self.gates = [StartGate() for _ in self.children]
        self.gates[0].release.set()
        self.starts = 0
        self.supervisor: WorkerSupervisor
        self.pool = ThreadPoolExecutor(max_workers=4)
        monkeypatch.setattr(worker_module, "start_worker", self.start_worker)
        monkeypatch.setattr(worker_module, "WorkerSupervisor", self.supervise)
        monkeypatch.setattr(analyzer_module, "time", SimpleNamespace(monotonic=self.clock))

    def start_worker(self, *_: object) -> GatedWorker:
        index = self.starts
        self.starts += 1
        gate = self.gates[index]
        gate.entered.set()
        if not gate.release.wait(_WAIT) or gate.fail:
            raise AnalysisModelLoadError("Controlled start failure.")
        return self.children[index]

    def supervise(
        self,
        request: worker_module.BackendLoadRequest,
        loader: worker_module.BackendLoader,
        settings: worker_module.WorkerSettings,
    ) -> WorkerSupervisor:
        supervisor = WorkerSupervisor.__new__(WorkerSupervisor)
        self.supervisor = supervisor
        WorkerSupervisor.__init__(
            supervisor,
            request,
            loader,
            settings,
            clock=self.clock,
            clock_ns=lambda: round(self.clock() * 1_000_000_000),
        )
        self.clock.notify = self.supervisor._wake
        return self.supervisor

    def wait_state(self, state: WorkerState) -> None:
        with self.supervisor._condition:
            assert self.supervisor._condition.wait_for(
                lambda: self.supervisor.state == state, timeout=_WAIT
            )

    def wait_queue(self, size: int) -> None:
        with self.supervisor._condition:
            assert self.supervisor._condition.wait_for(
                lambda: len(self.supervisor._queued) == size, timeout=_WAIT
            )


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch, scripted: None) -> Iterator[Harness]:
    harness = Harness(monkeypatch)
    yield harness
    for gate, child in zip(harness.gates, harness.children, strict=True):
        gate.release.set()
        child.stop_release.set()
    harness.clock.advance(10_000)
    harness.pool.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("fail", [False, True], ids=["ready", "load-error"])
def test_first_start_controls_whether_load_returns(
    tmp_path: Path, gated: Harness, fail: bool
) -> None:
    initial = gated.gates[0]
    initial.release.clear()
    initial.fail = fail
    path = write_artifact(tmp_path / "model.ser")
    loading = gated.pool.submit(load_artifact, path, _CONFIG)
    assert initial.entered.wait(_WAIT)
    gated.wait_state("starting")
    assert not loading.done()
    initial.release.set()
    if fail:
        with pytest.raises(AnalysisModelLoadError):
            loading.result(timeout=_WAIT)
        gated.wait_state("closed")
    else:
        analyzer = loading.result(timeout=_WAIT)
        gated.wait_state("ready")
        analyzer.close()
        gated.wait_state("closed")


def test_blocked_decode_returns_before_kill_reap_or_restart(tmp_path: Path, gated: Harness) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]
    child.stop_release.clear()
    call = gated.pool.submit(analyzer.analyze, tmp_path / "blocked.wav")
    assert child.jobs.get(timeout=_WAIT).job.path is not None
    gated.wait_state("busy")
    child.started()
    gated.clock.advance(1.0)

    assert child.stop_entered.wait(_WAIT)
    record = call.result(timeout=_WAIT)
    assert (record["status"], record["timing_ms"]) == ("timeout", {"features": 1_000, "predict": 0})
    assert not gated.gates[1].entered.is_set()
    gated.wait_state("restarting")
    child.stop_release.set()
    assert gated.gates[1].entered.wait(_WAIT)
    gated.gates[1].release.set()
    gated.wait_state("ready")


def test_path_source_extends_deadline_and_scoring_keeps_feature_timing(
    tmp_path: Path, gated: Harness
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]
    call = gated.pool.submit(analyzer.analyze, tmp_path / "long.wav")
    assert child.jobs.get(timeout=_WAIT).deadline == 101.0
    child.started()
    gated.clock.advance(0.5)
    child.events.append(
        WorkerEvent("source", duration_s=2.0, input_sample_rate_hz=8_000, received_at=100.5)
    )
    child.events.append(
        WorkerEvent(
            "features", features_ms=500, stage_started_ns=100_500_000_000, received_at=100.5
        )
    )
    child.notify()
    gated.clock.advance(0.5)
    assert not call.done()
    gated.clock.advance(3.0)

    record = call.result(timeout=_WAIT)
    assert record["status"] == "timeout"
    assert record["audio"] == {"duration_s": 2.0, "input_sample_rate_hz": 8_000}
    assert record["timing_ms"] == {"features": 500, "predict": 3_500}


@pytest.mark.parametrize(("received", "status"), [(100.999, "silent"), (101.0, "timeout")])
def test_result_deadline_tie_uses_receive_time(
    tmp_path: Path, gated: Harness, received: float, status: str
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]
    call = gated.pool.submit(analyzer.analyze, tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)
    assert child.jobs.get(timeout=_WAIT).deadline == 101.0
    with gated.supervisor._condition:
        child.outcome(0.5, stamp=received)
        gated.clock.advance(1.0)

    assert call.result(timeout=_WAIT)["status"] == status


def test_a_call_that_times_out_in_the_queue_does_not_evict_the_running_one(
    tmp_path: Path, gated: Harness
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]

    def submit(seconds: float) -> Future[UtteranceRecord]:
        return gated.pool.submit(analyzer.analyze, tone(seconds), sample_rate_hz=CANONICAL_RATE_HZ)

    running = submit(2.0)
    assert child.jobs.get(timeout=_WAIT).deadline == 104.0
    expires = submit(0.5)
    gated.wait_queue(1)
    first = submit(1.5)
    gated.wait_queue(2)
    second = submit(1.0)
    gated.wait_queue(3)
    gated.clock.advance(1.0)
    assert expires.result(timeout=_WAIT)["status"] == "timeout"
    child.outcome(2.0)
    assert running.result(timeout=_WAIT)["status"] == "silent"
    for duration, future in ((1.5, first), (1.0, second)):
        assert child.jobs.get(timeout=_WAIT).deadline == 100.0 + duration * 2
        child.outcome(duration)
        assert future.result(timeout=_WAIT)["status"] == "silent"
    assert gated.starts == 1


def test_idle_death_restarts_on_demand_and_waiters_keep_their_deadlines(
    tmp_path: Path, gated: Harness
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    gated.wait_state("ready")
    gated.children[0].die()
    gated.wait_state("ready")
    assert gated.starts == 1
    expires = gated.pool.submit(analyzer.analyze, tone(0.5), sample_rate_hz=CANONICAL_RATE_HZ)
    assert gated.gates[1].entered.wait(_WAIT)
    gated.wait_state("restarting")
    waits = gated.pool.submit(analyzer.analyze, tone(2.0), sample_rate_hz=CANONICAL_RATE_HZ)
    gated.wait_queue(2)
    gated.clock.advance(1.0)
    assert expires.result(timeout=_WAIT)["status"] == "timeout"
    gated.gates[1].release.set()
    child = gated.children[1]
    assert child.jobs.get(timeout=_WAIT).deadline == 104.0
    child.outcome(2.0)
    assert waits.result(timeout=_WAIT)["status"] == "silent"
    gated.wait_state("ready")


def test_busy_death_failed_restart_without_waiters_and_failed_retry(
    tmp_path: Path, gated: Harness
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]
    dies = gated.pool.submit(analyzer.analyze, tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    child.jobs.get(timeout=_WAIT)
    child.die()
    assert dies.result(timeout=_WAIT)["status"] == "model_error"
    assert gated.gates[1].entered.wait(_WAIT)
    gated.gates[1].fail = True
    gated.gates[1].release.set()
    gated.wait_state("failed")
    assert gated.starts == 2

    waits = gated.pool.submit(analyzer.analyze, tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    assert gated.gates[2].entered.wait(_WAIT)
    gated.gates[2].fail = True
    gated.gates[2].release.set()
    assert waits.result(timeout=_WAIT)["status"] == "model_error"
    gated.wait_state("failed")
    assert gated.starts == 3


def test_close_rejects_queued_and_new_calls_but_waits_for_active(
    tmp_path: Path, gated: Harness
) -> None:
    analyzer = load_artifact(write_artifact(tmp_path / "model.ser"), _CONFIG)
    child = gated.children[0]
    active = gated.pool.submit(analyzer.analyze, tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    child.jobs.get(timeout=_WAIT)
    queued = gated.pool.submit(analyzer.analyze, tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    gated.wait_queue(1)
    closing = gated.pool.submit(analyzer.close)
    gated.wait_state("closing")
    with pytest.raises(AnalysisClosedError):
        queued.result(timeout=_WAIT)
    with pytest.raises(AnalysisClosedError):
        analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    assert not closing.done()
    child.outcome(1.0)
    assert active.result(timeout=_WAIT)["status"] == "silent"
    closing.result(timeout=_WAIT)
    gated.wait_state("closed")
