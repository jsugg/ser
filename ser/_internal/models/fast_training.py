"""Fast-profile training workflow helpers extracted from emotion_model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ser._internal.data import Utterance
from ser._internal.data.data_loader import CheckedFastData
from ser._internal.features.feature_extractor import fast_feature_contract
from ser._internal.models.dataset_splitting import (
    fast_split_digest,
    split_utterances,
    split_utterances_three_way,
)
from ser._internal.models.training_orchestration import (
    build_training_robustness_provenance,
    current_training_state,
    prepare_until_quarantine_stable,
    publish_prepared_features,
    read_prepared_feature_payload,
    validate_operation_plan,
)
from ser._internal.models.training_readiness import TrainingMode
from ser.config import AppConfig, FeatureFlags

type EmotionClassifier = MLPClassifier | Pipeline
type DatasetSplit = tuple[np.ndarray, np.ndarray, list[str], list[str]]


def validate_fast_feature_contract(
    metadata: Mapping[str, object] | None, flags: FeatureFlags
) -> None:
    expected = fast_feature_contract(flags)
    if metadata is None or metadata.get("model_revision") != expected:
        raise ValueError(
            f"Incompatible fast feature contract: expected {expected!r}; retrain or prepare again."
        )


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
    settings: AppConfig
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
    load_checked_data: Callable[[Sequence[Utterance]], CheckedFastData | None] | None = None


def train_fast_model(*, hooks: FastTrainingHooks) -> None:
    """Runs the fast-profile training workflow via injected dependencies."""
    settings = hooks.settings
    utterances_for_consent = hooks.load_utterances()
    if not utterances_for_consent:
        raise RuntimeError(
            "Dataset not loaded: fast training requires labeled utterances with speaker identity."
        )
    hooks.ensure_dataset_consents_for_training(utterances=list(utterances_for_consent))
    revision = fast_feature_contract(settings.feature_flags)

    state = current_training_state()
    plan = None
    if state.operation.prepared_plan is not None:
        plan = validate_operation_plan(
            settings=settings,
            backend_id=hooks.default_backend_id,
            model_id="builtin",
            device="cpu",
            dtype="float64",
        )
    train_ids: list[str] = []
    test_ids: list[str] = []
    if plan is not None:
        payload = read_prepared_feature_payload(plan)
        validate_fast_feature_contract(payload.metadata, settings.feature_flags)
        x_train, x_test = payload.x_train, payload.x_test
        y_train, y_test = payload.y_train, payload.y_test
    else:
        checked_loader = hooks.load_checked_data
        if checked_loader is None:
            raise RuntimeError("Fast training requires the speaker-checked feature loader.")
        data = prepare_until_quarantine_stable(
            settings=settings,
            prepare=lambda: checked_loader(
                current_training_state().utterances or utterances_for_consent
            ),
        )
        if data is None:
            hooks.logger.error("Dataset not loaded. Please load the dataset first.")
            raise RuntimeError("Dataset not loaded. Please load the dataset first.")
        (x_train, x_test, y_train, y_test), train_ids, test_ids = data
    inventory = list(state.utterances or utterances_for_consent)
    partitions: dict[str, list[Utterance]]
    if state.readiness is not None:
        train, dev, test, _ = split_utterances_three_way(
            samples=inventory, settings=settings, logger=hooks.logger
        )
        partitions = {"train": train, "dev": dev, "test": test}
    else:
        train, test, _ = split_utterances(samples=inventory, settings=settings, logger=hooks.logger)
        partitions = {"train": train, "test": test}
    split_digest = fast_split_digest(partitions)
    if state.operation.mode is TrainingMode.PREPARE_ONLY:
        readiness_utterances = list(state.utterances)
        publish_prepared_features(
            settings=settings,
            backend_id=hooks.default_backend_id,
            model_id="builtin",
            device="cpu",
            dtype="float64",
            utterances=readiness_utterances,
            x_train=np.asarray(x_train, dtype=np.float64),
            x_test=np.asarray(x_test, dtype=np.float64),
            y_train=y_train,
            y_test=y_test,
            metadata={
                "model_revision": revision,
                "split_digest": split_digest,
                "train_meta": [{"sample_id": i} for i in train_ids],
                "test_meta": [{"sample_id": i} for i in test_ids],
            },
            cache_namespace="fast_features",
            windowing_policy={"strategy": "analysis_windows", "feature_contract": revision},
            noise_statistics={},
        )
        return
    if plan is None and state.readiness is not None and state.utterances:
        from ser._internal.models.training_orchestration import canonical_train_rows  # noqa: TID251

        x_train, y_train = canonical_train_rows(
            settings=settings,
            x_train=np.asarray(x_train, dtype=np.float64),
            y_train=y_train,
            train_sample_ids=train_ids,
        )
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
    provenance = {
        **provenance,
        "training_robustness": build_training_robustness_provenance(),
    }
    artifact = hooks.build_model_artifact(
        model=model,
        feature_vector_size=int(x_train.shape[1]),
        training_samples=int(x_train.shape[0]),
        labels=y_train,
        provenance=provenance,
        model_revision=revision,
        split_ledger_digest=split_digest,
    )
    artifact_metadata = hooks.extract_artifact_metadata(artifact)
    persisted_artifacts = hooks.persist_model_artifacts(model=model, artifact=artifact)
    hooks.logger.info(msg=f"Model saved to {persisted_artifacts.pickle_path}")
    if persisted_artifacts.secure_path is not None:
        hooks.logger.info(msg=f"Secure model saved to {persisted_artifacts.secure_path}")

    report_provenance = {
        **provenance,
        "training_robustness": build_training_robustness_provenance(),
    }
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
        provenance=report_provenance,
        data_controls={
            "feature_contract": revision,
            "split_digest": split_digest,
            "speaker_overlap_count": 0,
            "clip_counts": {name: len(clips) for name, clips in partitions.items()},
            "evaluation_unit": "window",
        },
    )
    skipped_clips = len(state.quarantine_records)
    report.update(
        dataset_corpus_samples=len(inventory) + skipped_clips,
        dataset_effective_samples=len(inventory),
        dataset_skipped_samples=skipped_clips,
    )
    hooks.persist_training_report(report, settings.models.training_report_file)
    hooks.logger.info(msg=f"Training report saved to {settings.models.training_report_file}")
