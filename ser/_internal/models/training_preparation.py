"""Training preparation, evaluation, and metadata helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ser.models.training_types import (
    AccurateTrainingPreparation,
    MediumTrainingPreparation,
    TrainingEvaluation,
)
from ser.train.metrics import compute_grouped_ser_metrics_by_sample, compute_ser_metrics

_UtteranceT = TypeVar("_UtteranceT")
_SplitMetaT = TypeVar("_SplitMetaT")
_MetaT = TypeVar("_MetaT")
_NoiseStatsT = TypeVar("_NoiseStatsT")
_BackendT = TypeVar("_BackendT")
_CacheT = TypeVar("_CacheT")


def evaluate_training_predictions(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> TrainingEvaluation:
    """Computes core SER metrics and validates the required UAR payload."""

    y_true_labels = [str(item) for item in y_true]
    y_pred_labels = [str(item) for item in y_pred]
    accuracy: float = float(accuracy_score(y_true=y_true_labels, y_pred=y_pred_labels))
    macro_f1: float = float(f1_score(y_true_labels, y_pred_labels, average="macro"))
    ser_metrics = compute_ser_metrics(y_true=y_true_labels, y_pred=y_pred_labels)
    uar = ser_metrics.get("uar")
    if not isinstance(uar, float):
        raise RuntimeError("SER metrics payload missing float 'uar'.")
    return TrainingEvaluation(
        accuracy=accuracy,
        macro_f1=macro_f1,
        uar=uar,
        ser_metrics=ser_metrics,
    )


def attach_grouped_metrics(
    *,
    ser_metrics: dict[str, object],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sample_ids: Sequence[str],
    corpus_ids: Sequence[str],
    language_ids: Sequence[str],
    min_support: int,
) -> dict[str, object]:
    """Adds corpus/language grouped metrics to one evaluation payload."""

    y_true_labels = [str(item) for item in y_true]
    y_pred_labels = [str(item) for item in y_pred]
    sample_id_labels = [str(item) for item in sample_ids]
    corpus_group_ids = [str(item) for item in corpus_ids]
    language_group_ids = [str(item) for item in language_ids]
    ser_metrics["group_metrics"] = {
        "by_corpus": compute_grouped_ser_metrics_by_sample(
            y_true=y_true_labels,
            y_pred=y_pred_labels,
            sample_ids=sample_id_labels,
            group_ids=corpus_group_ids,
            min_support=min_support,
        ),
        "by_language": compute_grouped_ser_metrics_by_sample(
            y_true=y_true_labels,
            y_pred=y_pred_labels,
            sample_ids=sample_id_labels,
            group_ids=language_group_ids,
            min_support=min_support,
        ),
    }
    return ser_metrics


def extract_normalized_artifact_metadata(
    artifact: dict[str, object],
    *,
    normalize_metadata: Callable[[dict[str, object]], dict[str, object]],
) -> dict[str, object]:
    """Extracts artifact metadata payload and applies one normalization callback."""

    artifact_metadata_obj = artifact.get("metadata")
    if not isinstance(artifact_metadata_obj, dict):
        raise RuntimeError("Model artifact metadata is missing before persistence.")
    return normalize_metadata(artifact_metadata_obj)


def prepare_medium_training_features(
    *,
    utterances: list[_UtteranceT],
    split_utterances: Callable[
        [list[_UtteranceT]],
        tuple[list[_UtteranceT], list[_UtteranceT], _SplitMetaT],
    ],
    resolve_model_id: Callable[[], str],
    resolve_runtime_selectors: Callable[[], tuple[str, str]],
    build_backend: Callable[[str, str, str], _BackendT],
    build_cache: Callable[[], _CacheT],
    build_feature_dataset: Callable[
        [list[_UtteranceT], _BackendT, _CacheT, str],
        tuple[np.ndarray, list[str], list[_MetaT], _NoiseStatsT],
    ],
) -> MediumTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT, _NoiseStatsT]:
    """Builds medium-profile feature matrices and runtime metadata for training."""

    train_utterances, test_utterances, split_metadata = split_utterances(utterances)
    model_id = resolve_model_id()
    runtime_device, runtime_dtype = resolve_runtime_selectors()
    backend = build_backend(model_id, runtime_device, runtime_dtype)
    cache = build_cache()
    x_train, y_train, _train_meta, train_noise_stats = build_feature_dataset(
        train_utterances,
        backend,
        cache,
        model_id,
    )
    x_test, y_test, test_meta, test_noise_stats = build_feature_dataset(
        test_utterances,
        backend,
        cache,
        model_id,
    )
    return MediumTrainingPreparation(
        train_utterances=train_utterances,
        test_utterances=test_utterances,
        split_metadata=split_metadata,
        model_id=model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        test_meta=test_meta,
        train_noise_stats=train_noise_stats,
        test_noise_stats=test_noise_stats,
    )


def prepare_accurate_training_features(
    *,
    utterances: list[_UtteranceT],
    split_utterances: Callable[
        [list[_UtteranceT]],
        tuple[list[_UtteranceT], list[_UtteranceT], _SplitMetaT],
    ],
    resolve_model_id: Callable[[], str],
    resolve_runtime_selectors: Callable[[], tuple[str, str]],
    build_backend: Callable[[str, str, str], _BackendT],
    build_cache: Callable[[], _CacheT],
    build_feature_dataset: Callable[
        [list[_UtteranceT], _BackendT, _CacheT, str],
        tuple[np.ndarray, list[str], list[_MetaT]],
    ],
) -> AccurateTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT]:
    """Builds accurate-profile feature matrices and runtime metadata for training."""

    train_utterances, test_utterances, split_metadata = split_utterances(utterances)
    model_id = resolve_model_id()
    runtime_device, runtime_dtype = resolve_runtime_selectors()
    backend = build_backend(model_id, runtime_device, runtime_dtype)
    cache = build_cache()
    x_train, y_train, _train_meta = build_feature_dataset(
        train_utterances,
        backend,
        cache,
        model_id,
    )
    x_test, y_test, test_meta = build_feature_dataset(
        test_utterances,
        backend,
        cache,
        model_id,
    )
    return AccurateTrainingPreparation(
        train_utterances=train_utterances,
        test_utterances=test_utterances,
        split_metadata=split_metadata,
        model_id=model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        test_meta=test_meta,
    )


def prepare_accurate_training_payload(
    *,
    logger: logging.Logger,
    load_utterances_for_training: Callable[[], list[_UtteranceT] | None],
    ensure_dataset_consents_for_training: Callable[[list[_UtteranceT]], None],
    prepare_training_features: Callable[
        [list[_UtteranceT]],
        AccurateTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT],
    ],
) -> tuple[
    list[_UtteranceT],
    AccurateTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT],
]:
    """Loads utterances, enforces consents, and prepares accurate training payload."""

    utterances = load_utterances_for_training()
    if utterances is None:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    ensure_dataset_consents_for_training(utterances)
    prepared = prepare_training_features(utterances)
    return utterances, prepared


def prepare_medium_training_payload(
    *,
    logger: logging.Logger,
    load_utterances_for_training: Callable[[], list[_UtteranceT] | None],
    ensure_dataset_consents_for_training: Callable[[list[_UtteranceT]], None],
    prepare_training_features: Callable[
        [list[_UtteranceT]],
        MediumTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT, _NoiseStatsT],
    ],
) -> tuple[
    list[_UtteranceT],
    MediumTrainingPreparation[_UtteranceT, _SplitMetaT, _MetaT, _NoiseStatsT],
]:
    """Loads utterances, enforces consents, and prepares medium training payload."""

    utterances = load_utterances_for_training()
    if utterances is None:
        logger.error("Dataset not loaded. Please load the dataset first.")
        raise RuntimeError("Dataset not loaded. Please load the dataset first.")
    ensure_dataset_consents_for_training(utterances)
    prepared = prepare_training_features(utterances)
    return utterances, prepared


__all__ = [
    "attach_grouped_metrics",
    "evaluate_training_predictions",
    "extract_normalized_artifact_metadata",
    "prepare_accurate_training_features",
    "prepare_accurate_training_payload",
    "prepare_medium_training_features",
    "prepare_medium_training_payload",
]
