"""Persistent worker process that holds one loaded model (ADR 0001, D3).

Reused from ``ser._internal.runtime``: the spawn context of the fast public boundary
and ``worker_lifecycle.terminate_worker_process``, on the supervisor thread only. Not
reused: the blocking ``recv_worker_message`` (a caller's wait must be bounded), the
per-call setup/compute handshake (one process per call breaks load-once), and the
thread-based ``run_with_timeout`` (it cannot stop native code).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ser._internal.analysis.backend import BackendLoader, BackendLoadRequest
from ser._internal.analysis.pipeline import AnalysisJob, PipelineOutcome
from ser.runtime.contracts import AnalysisConfig

_NOT_IMPLEMENTED = "The analysis worker is not implemented yet (brief A1-2)."

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

    kind: Literal["source", "features", "outcome"]
    duration_s: float | None = None
    input_sample_rate_hz: int | None = None
    outcome: PipelineOutcome | None = None


class AnalysisWorker:
    """Handle on one spawned child that runs one job at a time; nothing here blocks
    longer than the time it is given."""

    def submit(self, job: AnalysisJob) -> None:
        """Sends a job to the idle child; path decoding happens in the child."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def next_event(self, *, until_monotonic_s: float) -> WorkerEvent | None:
        """Waits on the pipe until ``until_monotonic_s``; ``None`` if nothing arrived.

        A bounded wait (``multiprocessing.connection.wait`` with a timeout), never a
        blocking receive. An event counts only once fully received.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def alive(self) -> bool:
        """False once the child was killed or exited."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def kill(self) -> None:
        """Sends SIGKILL and returns at once; reaping is the supervisor's."""
        raise NotImplementedError(_NOT_IMPLEMENTED)


def start_worker(
    request: BackendLoadRequest,
    loader: BackendLoader,
    settings: WorkerSettings,
) -> AnalysisWorker:
    """Spawns a child, applies ``settings``, and loads the backend once.

    Raises ``AnalysisModelLoadError`` if the child does not report ready within
    ``settings.start_timeout_s``.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)


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
        raise NotImplementedError(_NOT_IMPLEMENTED)

    @property
    def state(self) -> WorkerState:
        raise NotImplementedError(_NOT_IMPLEMENTED)

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
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        """Refuses queued and new calls, lets the call in flight end, stops the child.

        Idempotent.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)


__all__ = [
    "AnalysisWorker",
    "WorkerEvent",
    "WorkerSettings",
    "WorkerState",
    "WorkerSupervisor",
    "start_worker",
]
