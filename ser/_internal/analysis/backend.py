"""Backend seam: one implementation per profile behind the public analyser."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.profiles import ProfileName
from ser.runtime.contracts import AnalysisConfig, ModelIdentity


@dataclass(frozen=True)
class FrameScores:
    """Label probabilities for one backend window, in seconds of canonical audio."""

    start_s: float
    end_s: float
    labels: dict[str, float]


@dataclass(frozen=True)
class BackendPrediction:
    """Backend output for one canonical utterance."""

    labels: dict[str, float]
    frames: tuple[FrameScores, ...]
    features_ms: int
    predict_ms: int


class AnalysisDeadlineExceeded(Exception):
    """Raised inside a backend when the analysis deadline passes between stages."""


class AnalysisBackend(Protocol):
    """A loaded model for one profile; holds no per-call state."""

    @property
    def identity(self) -> ModelIdentity:
        """Identity of the loaded model file."""
        ...

    def predict(
        self,
        samples: NDArray[np.float32],
        *,
        deadline_monotonic_s: float,
    ) -> BackendPrediction:
        """Scores canonical 16 kHz mono samples or raises AnalysisDeadlineExceeded."""
        ...

    def close(self) -> None:
        """Releases the model."""
        ...


type BackendLoader = Callable[[Path | None, AnalysisConfig, AppConfig], AnalysisBackend]


def backend_loaders() -> dict[ProfileName, BackendLoader]:
    """Returns the loader for each profile the analyser supports."""
    raise NotImplementedError("Analysis backends are not implemented yet (briefs A1-2, A1-4).")


__all__ = [
    "AnalysisBackend",
    "AnalysisDeadlineExceeded",
    "BackendLoader",
    "BackendPrediction",
    "FrameScores",
    "backend_loaders",
]
