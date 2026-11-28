"""Persistent worker process that holds one loaded model (ADR 0001, hard deadline).

Reused from ``ser._internal.runtime``: the spawn context of the fast public boundary
and ``worker_lifecycle.terminate_worker_process``. Not reused: the per-call
setup/compute handshake (one process per call breaks load-once) and the thread-based
``run_with_timeout`` (it cannot stop native code).
"""

from __future__ import annotations

from dataclasses import dataclass

from ser._internal.analysis.backend import BackendLoader, BackendLoadRequest
from ser._internal.analysis.pipeline import AnalysisJob, PipelineOutcome
from ser.runtime.contracts import AnalysisConfig


@dataclass(frozen=True)
class WorkerSettings:
    """Process settings applied in the child before the loader runs.

    The child also hides every GPU; analysis is CPU-only.
    """

    max_threads: int
    nice: int
    start_timeout_s: float


class AnalysisWorker:
    """Handle on one spawned child that runs jobs one at a time until it is evicted."""

    def run(
        self, job: AnalysisJob, *, entry_monotonic_s: float, config: AnalysisConfig
    ) -> PipelineOutcome:
        """Runs one job; at the deadline kills the child and returns ``timeout``."""
        raise NotImplementedError("The analysis worker is not implemented yet (brief A1-2).")

    @property
    def alive(self) -> bool:
        """False once the child was evicted or exited."""
        raise NotImplementedError("The analysis worker is not implemented yet (brief A1-2).")

    def close(self) -> None:
        """Stops the child; idempotent."""
        raise NotImplementedError("The analysis worker is not implemented yet (brief A1-2).")


def start_worker(
    request: BackendLoadRequest,
    loader: BackendLoader,
    settings: WorkerSettings,
) -> AnalysisWorker:
    """Spawns a child, applies ``settings``, and loads the backend once.

    Raises ``AnalysisModelLoadError`` if the child does not report ready within
    ``settings.start_timeout_s``.
    """
    raise NotImplementedError("The analysis worker is not implemented yet (brief A1-2).")


__all__ = ["AnalysisWorker", "WorkerSettings", "start_worker"]
