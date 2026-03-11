"""Training execution and reporting helpers extracted from training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from ser.models.training_types import (
    AccurateTrainingPreparation,
    MediumTrainingPreparation,
    MediumTrainingSettingsLike,
    PersistedArtifactsLike,
    ProfileTrainingExecution,
    SplitMetadataLike,
    TrainingEvaluation,
)

_UtteranceT = TypeVar("_UtteranceT")
_MetaT = TypeVar("_MetaT")
_NoiseStatsT = TypeVar("_NoiseStatsT")
_ModelT = TypeVar("_ModelT")
_PersistedArtifactsT = TypeVar("_PersistedArtifactsT")
_SplitMetaWithStrategyT = TypeVar(
    "_SplitMetaWithStrategyT",
    bound=SplitMetadataLike,
)


class _PredictableModel(Protocol):
    """Structural contract for classifiers exposing fit/predict methods."""

    def fit(
        self,
        x_train: np.ndarray,
        y_train: Sequence[str],
    ) -> object | None:
        """Fits one model on the provided labels."""
        ...

    def predict(self, x_test: np.ndarray) -> Sequence[object]:
        """Predicts labels for one feature matrix."""
        ...


def execute_profile_training(
    *,
    create_classifier: Callable[[], _ModelT],
    fit_model: Callable[[_ModelT, np.ndarray, Sequence[str]], object | None],
    predict_model: Callable[[_ModelT, np.ndarray], Sequence[str]],
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    y_test: Sequence[str],
    test_meta: Sequence[_MetaT],
    min_support: int,
    evaluate_predictions: Callable[[list[str], list[str]], TrainingEvaluation],
    attach_grouped_metrics_for_prediction: Callable[
        [dict[str, object], list[str], list[str], Sequence[_MetaT], int],
        dict[str, object],
    ],
    build_model_artifact: Callable[[_ModelT], dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        _PersistedArtifactsT,
    ],
) -> ProfileTrainingExecution[_ModelT, _PersistedArtifactsT]:
    """Runs model fit/eval/grouped-metrics/artifact persistence for profile training."""
    y_train_labels = [str(item) for item in y_train]
    y_test_labels = [str(item) for item in y_test]
    model = create_classifier()
    fit_model(model, x_train, y_train_labels)
    y_pred = [str(item) for item in predict_model(model, x_test)]
    evaluation = evaluate_predictions(y_test_labels, y_pred)
    ser_metrics = attach_grouped_metrics_for_prediction(
        evaluation.ser_metrics,
        y_test_labels,
        y_pred,
        test_meta,
        min_support,
    )
    artifact = build_model_artifact(model)
    artifact_metadata = extract_artifact_metadata(artifact)
    persisted_artifacts = persist_model_artifacts(model, artifact)
    return ProfileTrainingExecution(
        model=model,
        evaluation=evaluation,
        ser_metrics=ser_metrics,
        artifact_metadata=artifact_metadata,
        persisted_artifacts=persisted_artifacts,
    )


def execute_default_profile_training(
    *,
    create_classifier: Callable[[], _ModelT],
    x_train: np.ndarray,
    y_train: Sequence[str],
    x_test: np.ndarray,
    y_test: Sequence[str],
    test_meta: Sequence[_MetaT],
    min_support: int,
    evaluate_predictions: Callable[..., TrainingEvaluation],
    attach_grouped_metrics: Callable[..., dict[str, object]],
    build_model_artifact: Callable[[_ModelT], dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        _PersistedArtifactsT,
    ],
) -> ProfileTrainingExecution[_ModelT, _PersistedArtifactsT]:
    """Runs default fit/predict-string profile-training execution."""

    def _fit_model(
        model: _ModelT,
        train_x: np.ndarray,
        train_y: Sequence[str],
    ) -> object | None:
        fit_callable = getattr(cast(_PredictableModel, model), "fit", None)
        if not callable(fit_callable):
            raise RuntimeError("Training model does not expose callable fit(...).")
        fit_result = fit_callable(train_x, train_y)
        return cast(object | None, fit_result)

    def _predict_model(
        model: _ModelT,
        test_x: np.ndarray,
    ) -> Sequence[str]:
        predict_callable = getattr(cast(_PredictableModel, model), "predict", None)
        if not callable(predict_callable):
            raise RuntimeError("Training model does not expose callable predict(...).")
        raw_predictions: Any = predict_callable(test_x)
        return [str(item) for item in raw_predictions]

    return execute_profile_training(
        create_classifier=create_classifier,
        fit_model=_fit_model,
        predict_model=_predict_model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        test_meta=test_meta,
        min_support=min_support,
        evaluate_predictions=lambda y_true, y_pred: evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
        ),
        attach_grouped_metrics_for_prediction=(
            lambda ser_metrics, y_true, y_pred, grouped_meta, grouped_min_support: (
                attach_grouped_metrics(
                    ser_metrics=ser_metrics,
                    y_true=y_true,
                    y_pred=y_pred,
                    test_meta=list(grouped_meta),
                    min_support=grouped_min_support,
                )
            )
        ),
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
    )


def finalize_profile_training_report(
    *,
    profile_label: str,
    logger: logging.Logger,
    evaluation: TrainingEvaluation,
    ser_metrics: dict[str, object],
    artifact_metadata: dict[str, object],
    persisted_artifacts: PersistedArtifactsLike,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: Sequence[str],
    y_test: Sequence[str],
    provenance: dict[str, object],
    data_controls: dict[str, object],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object]], None],
    report_destination: Path,
) -> dict[str, object]:
    """Logs profile-training outcome and persists the training report."""
    logger.info("%s model trained with %s pooled samples", profile_label, len(x_train))
    logger.info(msg=f"{profile_label} accuracy: {evaluation.accuracy * 100:.2f}%")
    logger.info(msg=f"{profile_label} macro F1 score: {evaluation.macro_f1:.4f}")
    logger.info(msg=f"{profile_label} UAR: {evaluation.uar:.4f}")
    logger.info("%s model saved to %s", profile_label, persisted_artifacts.pickle_path)
    if persisted_artifacts.secure_path is not None:
        logger.info(
            "%s secure model saved to %s",
            profile_label,
            persisted_artifacts.secure_path,
        )
    report = build_training_report(
        accuracy=evaluation.accuracy,
        macro_f1=evaluation.macro_f1,
        ser_metrics=ser_metrics,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        feature_vector_size=int(x_train.shape[1]),
        labels=[*y_train, *y_test],
        artifacts=persisted_artifacts,
        artifact_metadata=artifact_metadata,
        provenance=provenance,
        data_controls=data_controls,
    )
    persist_training_report(report)
    logger.info("%s training report saved to %s", profile_label, report_destination)
    return report


def run_medium_profile_training(
    *,
    prepared: MediumTrainingPreparation[
        _UtteranceT,
        _SplitMetaWithStrategyT,
        _MetaT,
        _NoiseStatsT,
    ],
    utterances: list[_UtteranceT],
    settings: MediumTrainingSettingsLike,
    logger: logging.Logger,
    profile_label: str,
    backend_id: str,
    profile_id: str,
    pooling_strategy: str,
    create_classifier: Callable[[], _ModelT],
    min_support: int,
    evaluate_predictions: Callable[..., TrainingEvaluation],
    attach_grouped_metrics: Callable[..., dict[str, object]],
    build_model_artifact: Callable[..., dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        PersistedArtifactsLike,
    ],
    build_provenance_metadata: Callable[..., dict[str, object]],
    build_dataset_controls: Callable[[list[_UtteranceT]], dict[str, object]],
    build_medium_noise_controls: Callable[..., dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [_SplitMetaWithStrategyT],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
) -> dict[str, object]:
    """Runs medium-profile training execution and persists report artifacts."""
    logger.info(
        "Medium dataset loaded successfully (train_files=%s, test_files=%s, split=%s).",
        len(prepared.train_utterances),
        len(prepared.test_utterances),
        prepared.split_metadata.split_strategy,
    )
    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=backend_id,
        profile=profile_id,
    )
    execution = execute_default_profile_training(
        create_classifier=create_classifier,
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_test=prepared.x_test,
        y_test=prepared.y_test,
        test_meta=prepared.test_meta,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=lambda model: build_model_artifact(
            model=model,
            feature_vector_size=int(prepared.x_train.shape[1]),
            training_samples=int(prepared.x_train.shape[0]),
            labels=prepared.y_train,
            backend_id=backend_id,
            profile=profile_id,
            feature_dim=int(prepared.x_train.shape[1]),
            frame_size_seconds=settings.medium_runtime.pool_window_size_seconds,
            frame_stride_seconds=settings.medium_runtime.pool_window_stride_seconds,
            pooling_strategy=pooling_strategy,
            backend_model_id=prepared.model_id,
            torch_device=prepared.runtime_device,
            torch_dtype=prepared.runtime_dtype,
            provenance=provenance,
        ),
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
    )
    return finalize_profile_training_report(
        profile_label=profile_label,
        logger=logger,
        evaluation=execution.evaluation,
        ser_metrics=execution.ser_metrics,
        artifact_metadata=execution.artifact_metadata,
        persisted_artifacts=execution.persisted_artifacts,
        x_train=prepared.x_train,
        x_test=prepared.x_test,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        provenance=provenance,
        data_controls={
            "dataset": build_dataset_controls(utterances),
            "medium_noise_controls": build_medium_noise_controls(
                min_window_std=settings.medium_training.min_window_std,
                max_windows_per_clip=settings.medium_training.max_windows_per_clip,
                train_stats=prepared.train_noise_stats,
                test_stats=prepared.test_noise_stats,
            ),
            "medium_grouped_evaluation": build_grouped_evaluation_controls(prepared.split_metadata),
        },
        build_training_report=build_training_report,
        persist_training_report=lambda report: persist_training_report(
            report,
            settings.models.training_report_file,
        ),
        report_destination=settings.models.training_report_file,
    )


def run_accurate_profile_training(
    *,
    prepared: AccurateTrainingPreparation[
        _UtteranceT,
        _SplitMetaWithStrategyT,
        _MetaT,
    ],
    utterances: list[_UtteranceT],
    settings: object,
    logger: logging.Logger,
    profile_label: str,
    backend_id: str,
    profile_id: str,
    pooling_strategy: str,
    frame_size_seconds: float,
    frame_stride_seconds: float,
    create_classifier: Callable[[], _ModelT],
    min_support: int,
    evaluate_predictions: Callable[..., TrainingEvaluation],
    attach_grouped_metrics: Callable[..., dict[str, object]],
    build_model_artifact: Callable[..., dict[str, object]],
    extract_artifact_metadata: Callable[[dict[str, object]], dict[str, object]],
    persist_model_artifacts: Callable[
        [_ModelT, dict[str, object]],
        PersistedArtifactsLike,
    ],
    build_provenance_metadata: Callable[..., dict[str, object]],
    build_dataset_controls: Callable[[list[_UtteranceT]], dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [_SplitMetaWithStrategyT],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
    report_destination: Path,
) -> dict[str, object]:
    """Runs accurate-profile training execution and persists report artifacts."""
    logger.info(
        "%s dataset loaded successfully (train_files=%s, test_files=%s, split=%s).",
        profile_label,
        len(prepared.train_utterances),
        len(prepared.test_utterances),
        prepared.split_metadata.split_strategy,
    )
    provenance = build_provenance_metadata(
        settings=settings,
        backend_id=backend_id,
        profile=profile_id,
    )
    execution = execute_default_profile_training(
        create_classifier=create_classifier,
        x_train=prepared.x_train,
        y_train=prepared.y_train,
        x_test=prepared.x_test,
        y_test=prepared.y_test,
        test_meta=prepared.test_meta,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=lambda model: build_model_artifact(
            model=model,
            feature_vector_size=int(prepared.x_train.shape[1]),
            training_samples=int(prepared.x_train.shape[0]),
            labels=prepared.y_train,
            backend_id=backend_id,
            profile=profile_id,
            feature_dim=int(prepared.x_train.shape[1]),
            frame_size_seconds=frame_size_seconds,
            frame_stride_seconds=frame_stride_seconds,
            pooling_strategy=pooling_strategy,
            backend_model_id=prepared.model_id,
            torch_device=prepared.runtime_device,
            torch_dtype=prepared.runtime_dtype,
            provenance=provenance,
        ),
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
    )
    return finalize_profile_training_report(
        profile_label=profile_label,
        logger=logger,
        evaluation=execution.evaluation,
        ser_metrics=execution.ser_metrics,
        artifact_metadata=execution.artifact_metadata,
        persisted_artifacts=execution.persisted_artifacts,
        x_train=prepared.x_train,
        x_test=prepared.x_test,
        y_train=prepared.y_train,
        y_test=prepared.y_test,
        provenance=provenance,
        data_controls={
            "dataset": build_dataset_controls(utterances),
            "accurate_grouped_evaluation": build_grouped_evaluation_controls(
                prepared.split_metadata
            ),
        },
        build_training_report=build_training_report,
        persist_training_report=lambda report: persist_training_report(
            report,
            report_destination,
        ),
        report_destination=report_destination,
    )
