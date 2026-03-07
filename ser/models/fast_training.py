"""Fast-profile training workflow helpers extracted from emotion_model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ser.data import Utterance

type EmotionClassifier = MLPClassifier | Pipeline
type DatasetSplit = tuple[np.ndarray, np.ndarray, list[str], list[str]]


class TrainingConfigLike(Protocol):
    """Minimal training-settings shape required by fast training helpers."""

    @property
    def test_size(self) -> float: ...


class ModelsConfigLike(Protocol):
    """Minimal model-settings shape required by fast training helpers."""

    @property
    def training_report_file(self) -> Path: ...


class SettingsLike(Protocol):
    """Minimal settings contract required by fast training helpers."""

    @property
    def training(self) -> TrainingConfigLike: ...

    @property
    def models(self) -> ModelsConfigLike: ...


class TrainingEvaluationLike(Protocol):
    """Minimal training-evaluation payload used by report generation."""

    @property
    def accuracy(self) -> float: ...

    @property
    def macro_f1(self) -> float: ...

    @property
    def uar(self) -> float: ...

    @property
    def ser_metrics(self) -> dict[str, object]: ...


class PersistedArtifactsLike(Protocol):
    """Minimal persisted-artifact contract used by fast training helpers."""

    @property
    def pickle_path(self) -> Path: ...

    @property
    def secure_path(self) -> Path | None: ...


class LoadDataFn(Protocol):
    """Callable contract for dataset loading with explicit test-size selector."""

    def __call__(self, *, test_size: float) -> DatasetSplit | None: ...


class EvaluateTrainingPredictionsFn(Protocol):
    """Callable contract for training evaluation from true/pred labels."""

    def __call__(
        self,
        *,
        y_true: list[str],
        y_pred: list[str],
    ) -> TrainingEvaluationLike: ...


class EnsureDatasetConsentsFn(Protocol):
    """Callable contract for dataset consent enforcement."""

    def __call__(self, *, utterances: list[Utterance]) -> None: ...


class PersistModelArtifactsFn(Protocol):
    """Callable contract for model artifact persistence."""

    def __call__(
        self,
        *,
        model: EmotionClassifier,
        artifact: dict[str, object],
    ) -> PersistedArtifactsLike: ...


class PersistTrainingReportFn(Protocol):
    """Callable contract for training-report persistence."""

    def __call__(self, report: dict[str, object], path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class FastTrainingHooks:
    """Dependency hooks required by the extracted fast training workflow."""

    logger: logging.Logger
    settings: SettingsLike
    load_utterances: Callable[[], Sequence[Utterance] | None]
    ensure_dataset_consents_for_training: EnsureDatasetConsentsFn
    load_data: LoadDataFn
    create_classifier: Callable[[], EmotionClassifier]
    evaluate_training_predictions: EvaluateTrainingPredictionsFn
    build_provenance_metadata: Callable[..., dict[str, object]]
    build_model_artifact: Callable[..., dict[str, object]]
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]]
    persist_model_artifacts: PersistModelArtifactsFn
    build_training_report: Callable[..., dict[str, object]]
    persist_training_report: PersistTrainingReportFn
    default_backend_id: str
    default_profile_id: str


def train_fast_model(*, hooks: FastTrainingHooks) -> None:
    """Runs the fast-profile training workflow via injected dependencies."""
    settings = hooks.settings
    utterances_for_consent = hooks.load_utterances()
    if utterances_for_consent:
        hooks.ensure_dataset_consents_for_training(
            utterances=list(utterances_for_consent),
        )

    data = hooks.load_data(test_size=float(settings.training.test_size))
    if data is None:
        hooks.logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    x_train, x_test, y_train, y_test = data
    model = hooks.create_classifier()
    hooks.logger.info(msg="Dataset loaded successfully.")

    model.fit(x_train, y_train)
    hooks.logger.info(msg=f"Model trained with {len(x_train)} samples")

    y_pred = [str(item) for item in model.predict(x_test)]
    evaluation = hooks.evaluate_training_predictions(y_true=y_test, y_pred=y_pred)
    hooks.logger.info(msg=f"Accuracy: {evaluation.accuracy * 100:.2f}%")
    hooks.logger.info(msg=f"Macro F1 score: {evaluation.macro_f1:.4f}")
    hooks.logger.info(msg=f"UAR: {evaluation.uar:.4f}")

    provenance = hooks.build_provenance_metadata(
        settings=settings,
        backend_id=hooks.default_backend_id,
        profile=hooks.default_profile_id,
    )
    artifact = hooks.build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        provenance=provenance,
    )
    artifact_metadata = hooks.extract_artifact_metadata(artifact)
    persisted_artifacts = hooks.persist_model_artifacts(model=model, artifact=artifact)
    hooks.logger.info(msg=f"Model saved to {persisted_artifacts.pickle_path}")
    if persisted_artifacts.secure_path is not None:
        hooks.logger.info(
            msg=f"Secure model saved to {persisted_artifacts.secure_path}"
        )

    report = hooks.build_training_report(
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        ser_metrics=evaluation.ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
    )
    hooks.persist_training_report(report, settings.models.training_report_file)
    hooks.logger.info(
        msg=f"Training report saved to {settings.models.training_report_file}"
    )
