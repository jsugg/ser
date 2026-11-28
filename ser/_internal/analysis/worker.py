"""Persistent worker process that holds one loaded model (ADR 0001, D3).

Reused from ``ser._internal.runtime``: the spawn context of the fast public boundary
and ``worker_lifecycle.terminate_worker_process``, on the supervisor thread only. Not
reused: the blocking ``recv_worker_message`` (a caller's wait must be bounded), the
per-call setup/compute handshake (one process per call breaks load-once), and the
thread-based ``run_with_timeout`` (it cannot stop native code).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from queue import SimpleQueue
from threading import Condition, Thread
from typing import Literal

from threadpoolctl import threadpool_limits

from ser._internal.analysis.backend import AnalysisBackend, BackendLoader, BackendLoadRequest
from ser._internal.analysis.errors import AnalysisClosedError, AnalysisModelLoadError
from ser._internal.analysis.pipeline import (
    AnalysisJob,
    PipelineOutcome,
    analysis_deadline_s,
    run_pipeline,
)
from ser._internal.runtime.worker_lifecycle import terminate_worker_process
from ser.runtime.contracts import AnalysisConfig

_LOGGER = logging.getLogger(__name__)

type WorkerState = Literal["starting", "ready", "busy", "restarting", "failed", "closing", "closed"]


@dataclass(frozen=True)
class WorkerSettings:
    """Process settings applied in the child before the loader runs.

    The child also hides every GPU; analysis is CPU-only.
    """

    max_threads: int
    nice: int
    start_timeout_s: float


@dataclass(frozen=True)
class WorkerEvent:
    """One message from the child about the job in flight.

    ``source``: the source duration and rate, sent as soon as the child knows them.
    ``features``: decode, conversion and extraction finished. ``outcome``: the result.
    """

    kind: Literal["ready", "started", "source", "features", "outcome"]
    duration_s: float | None = None
    input_sample_rate_hz: int | None = None
    outcome: PipelineOutcome | None = None
    stage_started_ns: int | None = None
    features_ms: int = 0
    received_at: float | None = None


@dataclass(frozen=True)
class _Work:
    job: AnalysisJob
    entry: float
    deadline: float
    config: AnalysisConfig


def _worker_main(
    connection: Connection,
    request: BackendLoadRequest,
    loader: BackendLoader,
    settings: WorkerSettings,
) -> None:
    sys.dont_write_bytecode = True
    try:
        threads = str(settings.max_threads)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            os.environ[key] = threads
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "ZE_AFFINITY_MASK",
        ):
            os.environ[key] = ""
        if hasattr(os, "nice"):
            current_nice = os.nice(0)
            os.nice(max(0, settings.nice - current_nice))
        with threadpool_limits(limits=settings.max_threads):
            backend = loader(request)
            try:
                connection.send(WorkerEvent("ready"))
                while (work := connection.recv()) is not None:
                    if not isinstance(work, _Work):
                        return
                    _run_work(connection, backend, request.manifest.identity.labels, work)
            finally:
                backend.close()
    except (Exception, KeyboardInterrupt):
        _LOGGER.debug("Analysis worker ended before completing its operation.")
    finally:
        connection.close()


def _run_work(
    connection: Connection,
    backend: AnalysisBackend,
    labels: tuple[str, ...],
    work: _Work,
) -> None:
    deadline = work.deadline
    if time.monotonic() < deadline:
        connection.send(WorkerEvent("started", stage_started_ns=time.monotonic_ns()))

    def report_source(duration: float, rate: int) -> None:
        nonlocal deadline
        if time.monotonic() < deadline:
            deadline = work.entry + analysis_deadline_s(work.config, duration)
        connection.send(WorkerEvent("source", duration_s=duration, input_sample_rate_hz=rate))

    def report_features(features_ms: int, predict_ns: int) -> None:
        connection.send(
            WorkerEvent("features", features_ms=features_ms, stage_started_ns=predict_ns)
        )

    result = run_pipeline(
        backend,
        work.job,
        work.config,
        deadline_monotonic_s=lambda: deadline,
        report_duration=report_source,
        labels=labels,
        report_features=report_features,
    )
    connection.send(WorkerEvent("outcome", outcome=result))


class AnalysisWorker:
    """Handle on one spawned child that runs one job at a time; nothing here blocks
    longer than the time it is given."""

    def __init__(self, process: BaseProcess, connection: Connection) -> None:
        self._process = process
        self._connection = connection
        self._inbox = Condition()
        self._events: deque[WorkerEvent] = deque()
        self._outbox: SimpleQueue[_Work | None] = SimpleQueue()
        self._disconnected = False
        self._notify: Callable[[], None] = lambda: None
        self._reader = Thread(target=self._receive, daemon=True, name="ser-analysis-receive")
        self._writer = Thread(target=self._send, daemon=True, name="ser-analysis-send")
        self._reader.start()
        self._writer.start()

    def notify_on_event(self, notify: Callable[[], None]) -> None:
        self._notify = notify

    def _receive(self) -> None:
        try:
            while True:
                message = self._connection.recv()
                if not isinstance(message, WorkerEvent):
                    break
                with self._inbox:
                    self._events.append(replace(message, received_at=time.monotonic()))
                    self._inbox.notify_all()
                self._notify()
        except (EOFError, OSError):
            pass
        finally:
            self._disconnect()

    def _send(self) -> None:
        try:
            while True:
                work = self._outbox.get()
                self._connection.send(work)
                if work is None:
                    return
        except (EOFError, OSError, ValueError):
            self._disconnect()

    def _disconnect(self) -> None:
        with self._inbox:
            self._disconnected = True
            self._inbox.notify_all()
        self._notify()

    def submit(self, job: _Work) -> None:
        """Sends a job to the idle child; path decoding happens in the child."""
        self._outbox.put(job)

    def next_event(self, *, until_monotonic_s: float) -> WorkerEvent | None:
        """Waits on the pipe until ``until_monotonic_s``; ``None`` if nothing arrived.

        The reader owns blocking receives, including partial messages; this wait
        only sees fully received messages and never touches the pipe.
        """
        with self._inbox:
            while not self._events and not self._disconnected:
                remaining = until_monotonic_s - time.monotonic()
                if remaining <= 0:
                    break
                self._inbox.wait(timeout=remaining)
            return self._events.popleft() if self._events else None

    @property
    def alive(self) -> bool:
        """False once the child was killed or exited."""
        with self._inbox:
            return not self._disconnected and self._process.exitcode is None

    def kill(self) -> None:
        """Sends SIGKILL and returns at once; reaping is the supervisor's."""
        if self._process.exitcode is None:
            self._process.kill()

    def stop(self, *, graceful: bool) -> None:
        self._outbox.put(None)
        if graceful:
            self._process.join(timeout=0.1)
            if self._process.exitcode is None:
                terminate_worker_process(
                    process=self._process, terminate_grace_seconds=0.1, kill_grace_seconds=0.5
                )
        else:
            self.kill()
            self._process.join(timeout=0.5)
        self._connection.close()
        self._reader.join(timeout=0.5)
        self._writer.join(timeout=0.5)
        if self._process.exitcode is not None:
            self._process.close()


def start_worker(
    request: BackendLoadRequest,
    loader: BackendLoader,
    settings: WorkerSettings,
) -> AnalysisWorker:
    """Spawns a child, applies ``settings``, and loads the backend once.

    Raises ``AnalysisModelLoadError`` if the child does not report ready within
    ``settings.start_timeout_s``.
    """
    deadline = time.monotonic() + settings.start_timeout_s
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_worker_main, args=(child, request, loader, settings))
    try:
        process.start()
    except (OSError, ValueError, TypeError):
        parent.close()
        raise AnalysisModelLoadError("The analysis worker could not start.") from None
    finally:
        child.close()
    worker = AnalysisWorker(process, parent)
    ready = worker.next_event(until_monotonic_s=deadline)
    if (
        ready is not None
        and ready.kind == "ready"
        and ready.received_at is not None
        and ready.received_at < deadline
    ):
        return worker
    worker.stop(graceful=False)
    raise AnalysisModelLoadError("The analysis worker did not become ready.") from None


@dataclass(eq=False)
class _Pending:
    work: _Work
    duration: float | None
    rate: int | None
    features_started_ns: int | None = None
    predict_started_ns: int | None = None
    features_ms: int = 0
    result: PipelineOutcome | None = None
    closed: bool = False

    def interrupted(self, status: Literal["timeout", "model_error"], now: float) -> PipelineOutcome:
        end_ns = int(now * 1_000_000_000)

        def elapsed(start_ns: int | None) -> int:
            return 0 if start_ns is None else max(0, (end_ns - start_ns) // 1_000_000)

        return PipelineOutcome(
            status,
            None,
            self.duration,
            self.rate,
            (
                self.features_ms
                if self.predict_started_ns is not None
                else elapsed(self.features_started_ns)
            ),
            elapsed(self.predict_started_ns),
        )


class WorkerSupervisor:
    """Owns one analyser's worker, its state and its restarts (ADR 0001, D3).

    Transitions:

    ============  ==================================================  ============
    From          Event                                               To
    ============  ==================================================  ============
    starting      child ready (``load`` returns)                      ready
    starting      start fails or exceeds ``worker_start_timeout_s``   (``load`` raises)
    ready         next call admitted, child alive                     busy
    ready         next call admitted, child died while idle           restarting
    busy          outcome fully received before the deadline          ready
    busy          deadline passed: call returns ``timeout``           restarting
    busy          child died: call returns ``model_error``            restarting
    restarting    child ready                                         ready
    restarting    start fails or exceeds ``worker_start_timeout_s``   failed
    failed        next call admitted: one new start attempt           restarting
    any open      ``close()`` begins                                  closing
    closing       in-flight call ended, child stopped                 closed
    ============  ==================================================  ============

    Calls are admitted in arrival order. A call waiting on ``restarting`` gets its
    result if the child comes back, ``timeout`` if its own deadline passes first,
    and ``model_error`` if that start fails. A failed start with nobody waiting
    leaves ``failed``. Kill, reap and restart run on the supervisor's own thread,
    never on a caller's. ``clock`` is injectable for tests.
    """

    def __init__(
        self,
        request: BackendLoadRequest,
        loader: BackendLoader,
        settings: WorkerSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request = request
        self._loader = loader
        self._settings = settings
        self._clock = clock
        self._condition = Condition()
        self._state: WorkerState = "starting"
        self._closing = False
        self._worker: AnalysisWorker | None = None
        self._active: _Pending | None = None
        self._queued: deque[_Pending] = deque()
        self._thread = Thread(target=self._supervise, daemon=True, name="ser-analysis-supervisor")
        self._thread.start()
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._state != "starting", timeout=settings.start_timeout_s
            )
            if not ready or self._state != "ready":
                self._closing = True
                self._state = "closing"
                self._condition.notify_all()
                raise AnalysisModelLoadError("The analysis worker did not become ready.") from None

    @property
    def state(self) -> WorkerState:
        with self._condition:
            return self._state

    def check_open(self) -> None:
        with self._condition:
            if self._closing:
                raise AnalysisClosedError("The analyzer is closed.") from None

    def _wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def run(
        self,
        job: AnalysisJob,
        *,
        entry_monotonic_s: float,
        known_duration_s: float | None,
        config: AnalysisConfig,
    ) -> PipelineOutcome:
        """Queues the job and waits for its outcome or its deadline, whichever is first.

        The deadline starts from ``known_duration_s`` when given (array input);
        otherwise from ``timeout_floor_s``, extended once when a ``source`` event
        brings the duration. Raises ``AnalysisClosedError`` if ``close()`` began
        before the job started.
        """
        pending = _Pending(
            _Work(
                job,
                entry_monotonic_s,
                entry_monotonic_s + analysis_deadline_s(config, known_duration_s),
                config,
            ),
            known_duration_s,
            job.sample_rate_hz,
        )
        with self._condition:
            self.check_open()
            self._queued.append(pending)
            self._condition.notify_all()
            while True:
                if pending.closed:
                    raise AnalysisClosedError("The analyzer is closed.") from None
                if self._active is pending:
                    self._drain_events()
                if pending.result is not None:
                    return pending.result
                remaining = pending.work.deadline - self._clock()
                if remaining <= 0:
                    pending.result = pending.interrupted("timeout", pending.work.deadline)
                    if self._active is not pending:
                        self._queued.remove(pending)
                    self._condition.notify_all()
                    return pending.result
                self._condition.wait(timeout=remaining)

    def _drain_events(self) -> None:
        pending = self._active
        worker = self._worker
        if pending is None or worker is None or pending.result is not None:
            return
        while (event := worker.next_event(until_monotonic_s=time.monotonic())) is not None:
            received = event.received_at if event.received_at is not None else self._clock()
            if received >= pending.work.deadline:
                break
            if event.kind == "started":
                pending.features_started_ns = event.stage_started_ns
            elif event.kind == "source" and event.duration_s is not None:
                pending.duration, pending.rate = event.duration_s, event.input_sample_rate_hz
                pending.work = replace(
                    pending.work,
                    deadline=pending.work.entry
                    + analysis_deadline_s(pending.work.config, event.duration_s),
                )
            elif event.kind == "features":
                pending.features_ms = event.features_ms
                pending.predict_started_ns = event.stage_started_ns
            elif event.kind == "outcome" and event.outcome is not None:
                pending.result = event.outcome
                self._condition.notify_all()
                return

    def _start(self, *, first: bool) -> None:
        try:
            worker = start_worker(self._request, self._loader, self._settings)
            worker.notify_on_event(self._wake)
        except AnalysisModelLoadError:
            if not first:
                _LOGGER.warning("Analysis worker restart failed.")
            with self._condition:
                if not self._closing:
                    self._state = "failed"
                while self._queued:
                    pending = self._queued.popleft()
                    now = self._clock()
                    if now >= pending.work.deadline:
                        pending.result = pending.interrupted("timeout", pending.work.deadline)
                    else:
                        pending.result = pending.interrupted("model_error", now)
                self._condition.notify_all()
        else:
            with self._condition:
                self._worker = worker
                if not self._closing:
                    self._state = "ready"
                self._condition.notify_all()

    def _supervise(self) -> None:
        self._start(first=True)
        while True:
            with self._condition:
                self._drain_events()
                worker, pending = self._worker, self._active
                now = self._clock()
                if pending is not None:
                    assert worker is not None
                    dead = not worker.alive
                    if pending.result is None:
                        if now >= pending.work.deadline:
                            pending.result = pending.interrupted("timeout", pending.work.deadline)
                        elif dead:
                            pending.result = pending.interrupted("model_error", now)
                        else:
                            self._condition.wait(timeout=pending.work.deadline - now)
                            continue
                    self._active = None
                    self._condition.notify_all()
                    evict = pending.result.status == "timeout" or dead
                    if not evict:
                        if not self._closing:
                            self._state = "ready"
                        continue
                    reason = "died" if dead else "was evicted"
                elif self._closing:
                    break
                elif not self._queued:
                    self._condition.wait()
                    continue
                elif now >= self._queued[0].work.deadline:
                    expired = self._queued.popleft()
                    expired.result = expired.interrupted("timeout", expired.work.deadline)
                    self._condition.notify_all()
                    continue
                elif worker is not None and worker.alive:
                    self._active = self._queued.popleft()
                    self._state = "busy"
                    worker.submit(self._active.work)
                    continue
                else:
                    reason = "died" if worker is not None else None
                self._worker = None
                if not self._closing:
                    self._state = "restarting"
            if reason is not None:
                _LOGGER.warning("Analysis worker %s.", reason)
            if worker is not None:
                worker.stop(graceful=False)
            with self._condition:
                if self._closing:
                    break
            self._start(first=False)
        if self._worker is not None:
            self._worker.stop(graceful=True)
        with self._condition:
            self._worker = None
            self._state = "closed"
            self._condition.notify_all()

    def close(self) -> None:
        """Refuses queued and new calls, lets the call in flight end, stops the child.

        Idempotent.
        """
        with self._condition:
            if not self._closing:
                self._closing = True
                self._state = "closing"
                while self._queued:
                    self._queued.popleft().closed = True
                self._condition.notify_all()
        self._thread.join()


__all__ = [
    "AnalysisWorker",
    "WorkerEvent",
    "WorkerSettings",
    "WorkerState",
    "WorkerSupervisor",
    "start_worker",
]
