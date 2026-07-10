"""Shared typing contracts for training preparation and execution flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import numpy as np

_UtteranceT = TypeVar("_UtteranceT")
_SplitMetaT = TypeVar("_SplitMetaT")
_MetaT = TypeVar("_MetaT")
_NoiseStatsT = TypeVar("_NoiseStatsT")
_ModelT = TypeVar("_ModelT")
_PersistedArtifactsT = TypeVar("_PersistedArtifactsT")


@dataclass(frozen=True)
class TrainingEvaluation:
    """Deterministic evaluation payload for one train/test prediction pass."""

    accuracy: float
    macro_f1: float
    uar: float
    ser_metrics: dict[str, object]


class PersistedArtifactsLike(Protocol):
    """Protocol for persisted artifact locations used in training reporting."""

    @property
    def pickle_path(self) -> Path:
        """Returns persisted pickle artifact path."""
        ...

    @property
    def secure_path(self) -> Path | None:
        """Returns optional persisted secure artifact path."""
        ...


class SplitMetadataLike(Protocol):
    """Minimal split-metadata contract for training diagnostics."""

    @property
    def split_strategy(self) -> str:
        """Returns deterministic split strategy identifier."""
        ...


class MediumRuntimeConfigLike(Protocol):
    """Minimal medium-runtime settings used by orchestration helpers."""

    @property
    def pool_window_size_seconds(self) -> float:
        """Returns medium pooling window size."""
        ...

    @property
    def pool_window_stride_seconds(self) -> float:
        """Returns medium pooling window stride."""
        ...


class MediumTrainingConfigLike(Protocol):
    """Minimal medium-training settings used by orchestration helpers."""

    @property
    def min_window_std(self) -> float:
        """Returns medium minimum-window standard deviation threshold."""
        ...

    @property
    def max_windows_per_clip(self) -> int:
        """Returns medium maximum windows per clip threshold."""
        ...


class ModelPathConfigLike(Protocol):
    """Minimal model-path settings used by orchestration helpers."""

    @property
    def training_report_file(self) -> Path:
        """Returns training report output path."""
        ...


class MediumTrainingSettingsLike(Protocol):
    """Minimal settings shape required by medium orchestration helpers."""

    @property
    def medium_runtime(self) -> MediumRuntimeConfigLike:
        """Returns medium runtime settings."""
        ...

    @property
    def medium_training(self) -> MediumTrainingConfigLike:
        """Returns medium training settings."""
        ...

    @property
    def models(self) -> ModelPathConfigLike:
        """Returns model artifact path settings."""
        ...


@dataclass(frozen=True)
class MediumTrainingPreparation(Generic[_UtteranceT, _SplitMetaT, _MetaT, _NoiseStatsT]):
    """Prepared medium-training payload produced before model fitting."""

    train_utterances: list[_UtteranceT]
    test_utterances: list[_UtteranceT]
    split_metadata: _SplitMetaT
    model_id: str
    runtime_device: str
    runtime_dtype: str
    x_train: np.ndarray
    y_train: list[str]
    x_test: np.ndarray
    y_test: list[str]
    test_meta: list[_MetaT]
    train_noise_stats: _NoiseStatsT
    test_noise_stats: _NoiseStatsT


@dataclass(frozen=True)
class AccurateTrainingPreparation(Generic[_UtteranceT, _SplitMetaT, _MetaT]):
    """Prepared accurate-training payload produced before model fitting."""

    train_utterances: list[_UtteranceT]
    test_utterances: list[_UtteranceT]
    split_metadata: _SplitMetaT
    model_id: str
    runtime_device: str
    runtime_dtype: str
    x_train: np.ndarray
    y_train: list[str]
    x_test: np.ndarray
    y_test: list[str]
    test_meta: list[_MetaT]


@dataclass(frozen=True)
class ProfileTrainingExecution(Generic[_ModelT, _PersistedArtifactsT]):
    """Outputs of one non-fast profile train/evaluate/persist execution."""

    model: _ModelT
    evaluation: TrainingEvaluation
    ser_metrics: dict[str, object]
    artifact_metadata: dict[str, object]
    persisted_artifacts: _PersistedArtifactsT


__all__ = [
    "AccurateTrainingPreparation",
    "MediumTrainingPreparation",
    "MediumTrainingSettingsLike",
    "PersistedArtifactsLike",
    "ProfileTrainingExecution",
    "SplitMetadataLike",
    "TrainingEvaluation",
]
