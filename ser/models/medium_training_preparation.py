"""Medium-profile training preparation helpers for emotion-model entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

from ser.config import AppConfig
from ser.data import EmbeddingCache, Utterance
from ser.models.dataset_splitting import MediumSplitMetadata
from ser.models.training_execution import run_medium_profile_training
from ser.models.training_preparation import (
    prepare_medium_training_features,
    prepare_medium_training_payload,
)
from ser.models.training_types import (
    MediumTrainingPreparation,
    PersistedArtifactsLike,
    TrainingEvaluation,
)
from ser.repr import XLSRBackend

_MetaT = TypeVar("_MetaT")
_NoiseStatsT = TypeVar("_NoiseStatsT")
_ModelT = TypeVar("_ModelT")


class ResolveRuntimeSelectorsForBackend(Protocol):
    """Callable contract for backend-aware runtime selector resolution."""

    def __call__(self, *, settings: AppConfig, backend_id: str) -> tuple[str, str]: ...


type TrainMediumProfileModelCallable = Callable[..., None]


def prepare_medium_xlsr_training(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[[list[Utterance]], None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id: Callable[[], str],
    resolve_runtime_selectors: Callable[[], tuple[str, str]],
    build_backend: Callable[[str, str, str], XLSRBackend],
    build_feature_dataset: Callable[
        [list[Utterance], XLSRBackend, EmbeddingCache, str],
        tuple[np.ndarray, list[str], list[_MetaT], _NoiseStatsT],
    ],
    embedding_cache_path: Path,
) -> tuple[
    list[Utterance],
    MediumTrainingPreparation[
        Utterance,
        MediumSplitMetadata,
        _MetaT,
        _NoiseStatsT,
    ],
]:
    """Loads + prepares medium training payload using XLS-R backend wiring."""
    return prepare_medium_training_payload(
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=ensure_dataset_consents_for_training,
        prepare_training_features=lambda training_utterances: (
            prepare_medium_training_features(
                utterances=training_utterances,
                split_utterances=split_utterances,
                resolve_model_id=resolve_model_id,
                resolve_runtime_selectors=resolve_runtime_selectors,
                build_backend=build_backend,
                build_cache=lambda: EmbeddingCache(embedding_cache_path),
                build_feature_dataset=build_feature_dataset,
            )
        ),
    )


def run_medium_profile_training_from_prepared(
    *,
    prepared: MediumTrainingPreparation[
        Utterance,
        MediumSplitMetadata,
        _MetaT,
        _NoiseStatsT,
    ],
    utterances: list[Utterance],
    settings: AppConfig,
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
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_medium_noise_controls: Callable[..., dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
) -> dict[str, object]:
    """Runs medium-profile training from a prepared payload."""
    return run_medium_profile_training(
        prepared=prepared,
        utterances=utterances,
        settings=settings,
        logger=logger,
        profile_label=profile_label,
        backend_id=backend_id,
        profile_id=profile_id,
        pooling_strategy=pooling_strategy,
        create_classifier=create_classifier,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=build_dataset_controls,
        build_medium_noise_controls=build_medium_noise_controls,
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=build_training_report,
        persist_training_report=persist_training_report,
    )


def train_medium_profile_model(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[..., None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id_for_settings: Callable[[AppConfig], str],
    resolve_runtime_selectors_for_settings: Callable[[AppConfig], tuple[str, str]],
    build_backend: Callable[[str, str, str, AppConfig], XLSRBackend],
    build_feature_dataset: Callable[
        ...,
        tuple[np.ndarray, list[str], list[_MetaT], _NoiseStatsT],
    ],
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
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_medium_noise_controls: Callable[..., dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
    profile_label: str,
    backend_id: str,
    profile_id: str,
    pooling_strategy: str,
    embedding_cache_name: str = "medium_embeddings",
) -> None:
    """Runs medium-profile training from settings using delegated hooks."""
    utterances, prepared = prepare_medium_xlsr_training(
        settings=settings,
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=lambda training_utterances: (
            ensure_dataset_consents_for_training(utterances=training_utterances)
        ),
        split_utterances=split_utterances,
        resolve_model_id=lambda: resolve_model_id_for_settings(settings),
        resolve_runtime_selectors=lambda: resolve_runtime_selectors_for_settings(settings),
        build_backend=lambda model_id, runtime_device, runtime_dtype: build_backend(
            model_id,
            runtime_device,
            runtime_dtype,
            settings,
        ),
        build_feature_dataset=lambda partition, backend, cache, model_id: (
            build_feature_dataset(
                utterances=partition,
                backend=backend,
                cache=cache,
                model_id=model_id,
            )
        ),
        embedding_cache_path=settings.tmp_folder / embedding_cache_name,
    )
    _ = run_medium_profile_training_from_prepared(
        prepared=prepared,
        utterances=utterances,
        settings=settings,
        logger=logger,
        profile_label=profile_label,
        backend_id=backend_id,
        profile_id=profile_id,
        pooling_strategy=pooling_strategy,
        create_classifier=create_classifier,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=build_dataset_controls,
        build_medium_noise_controls=build_medium_noise_controls,
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=build_training_report,
        persist_training_report=persist_training_report,
    )


def train_medium_profile_entrypoint(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    train_profile_model: TrainMediumProfileModelCallable,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[..., None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id_for_settings: Callable[[AppConfig], str],
    resolve_runtime_selectors_for_backend: ResolveRuntimeSelectorsForBackend,
    build_backend_for_settings: Callable[[str, str, str, AppConfig], XLSRBackend],
    build_feature_dataset: Callable[
        ...,
        tuple[np.ndarray, list[str], list[_MetaT], _NoiseStatsT],
    ],
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
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_medium_noise_controls: Callable[..., dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
    profile_label: str,
    backend_id: str,
    profile_id: str,
    pooling_strategy: str,
    embedding_cache_name: str = "medium_embeddings",
) -> None:
    """Runs medium training entrypoint wiring while preserving helper contract keys."""
    train_profile_model(
        settings=settings,
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=ensure_dataset_consents_for_training,
        split_utterances=split_utterances,
        resolve_model_id_for_settings=resolve_model_id_for_settings,
        resolve_runtime_selectors_for_settings=lambda active_settings: (
            resolve_runtime_selectors_for_backend(
                settings=active_settings,
                backend_id=backend_id,
            )
        ),
        build_backend=lambda model_id, runtime_device, runtime_dtype, active_settings: (
            build_backend_for_settings(
                model_id,
                runtime_device,
                runtime_dtype,
                active_settings,
            )
        ),
        build_feature_dataset=build_feature_dataset,
        create_classifier=create_classifier,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=build_dataset_controls,
        build_medium_noise_controls=build_medium_noise_controls,
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=build_training_report,
        persist_training_report=persist_training_report,
        profile_label=profile_label,
        backend_id=backend_id,
        profile_id=profile_id,
        pooling_strategy=pooling_strategy,
        embedding_cache_name=embedding_cache_name,
    )


__all__ = [
    "prepare_medium_xlsr_training",
    "run_medium_profile_training_from_prepared",
    "train_medium_profile_entrypoint",
    "train_medium_profile_model",
]
