"""Accurate-profile training preparation helpers for emotion-model entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from ser.config import AppConfig
from ser.data import EmbeddingCache, Utterance
from ser.models.dataset_splitting import MediumSplitMetadata
from ser.models.training_orchestration import (
    AccurateTrainingPreparation,
    PersistedArtifactsLike,
    TrainingEvaluation,
    prepare_accurate_training_features,
    prepare_accurate_training_payload,
    run_accurate_profile_training,
)
from ser.repr import Emotion2VecBackend, WhisperBackend

_MetaT = TypeVar("_MetaT")
_ModelT = TypeVar("_ModelT")


class PreparedAccurateTrainingRunner(Protocol):
    """Callable contract for running one prepared accurate-profile training pass."""

    def __call__(
        self,
        *,
        prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, Any],
        utterances: list[Utterance],
        settings: AppConfig,
        profile_label: str,
        backend_id: str,
        profile_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
    ) -> None: ...


class ResolveRuntimeSelectorsForBackend(Protocol):
    """Callable contract for backend-aware runtime selector resolution."""

    def __call__(self, *, settings: AppConfig, backend_id: str) -> tuple[str, str]: ...


type BuildAccurateFeatureDatasetForBackend = Callable[
    ..., tuple[np.ndarray, list[str], list[Any]]
]
type TrainAccurateProfileModelCallable = Callable[..., None]


def train_accurate_whisper_profile_model(
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
    build_backend: Callable[[str, str, str, AppConfig], WhisperBackend],
    build_feature_dataset: Callable[
        ...,
        tuple[np.ndarray, list[str], list[_MetaT]],
    ],
    run_prepared_training: Callable[
        [
            AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
            list[Utterance],
            AppConfig,
        ],
        None,
    ],
    embedding_cache_name: str = "accurate_embeddings",
) -> None:
    """Runs accurate training from settings using delegated hooks."""
    utterances, prepared = prepare_accurate_whisper_training(
        settings=settings,
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=lambda training_utterances: (
            ensure_dataset_consents_for_training(utterances=training_utterances)
        ),
        split_utterances=split_utterances,
        resolve_model_id=lambda: resolve_model_id_for_settings(settings),
        resolve_runtime_selectors=lambda: resolve_runtime_selectors_for_settings(
            settings
        ),
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
    run_prepared_training(prepared, utterances, settings)


def train_accurate_research_profile_model(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    parse_allowed_restricted_backends_env: Callable[[], object],
    load_persisted_backend_consents: Callable[..., object],
    ensure_backend_access: Callable[..., object],
    restricted_backend_id: str,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[..., None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id_for_settings: Callable[[AppConfig], str],
    resolve_runtime_selectors_for_settings: Callable[[AppConfig], tuple[str, str]],
    build_backend: Callable[[str, str, str, AppConfig], Emotion2VecBackend],
    build_feature_dataset: Callable[
        ...,
        tuple[np.ndarray, list[str], list[_MetaT]],
    ],
    run_prepared_training: Callable[
        [
            AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
            list[Utterance],
            AppConfig,
        ],
        None,
    ],
    embedding_cache_name: str = "accurate_research_embeddings",
) -> None:
    """Runs accurate-research training from settings using delegated hooks."""
    allowed_restricted_backends = parse_allowed_restricted_backends_env()
    persisted_consents = load_persisted_backend_consents(settings=settings)
    ensure_backend_access(
        backend_id=restricted_backend_id,
        restricted_backends_enabled=settings.runtime_flags.restricted_backends,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    utterances, prepared = prepare_accurate_research_training(
        settings=settings,
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=lambda training_utterances: (
            ensure_dataset_consents_for_training(utterances=training_utterances)
        ),
        split_utterances=split_utterances,
        resolve_model_id=lambda: resolve_model_id_for_settings(settings),
        resolve_runtime_selectors=lambda: resolve_runtime_selectors_for_settings(
            settings
        ),
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
    run_prepared_training(prepared, utterances, settings)


def prepare_accurate_whisper_training(
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
    build_backend: Callable[[str, str, str], WhisperBackend],
    build_feature_dataset: Callable[
        [list[Utterance], WhisperBackend, EmbeddingCache, str],
        tuple[np.ndarray, list[str], list[_MetaT]],
    ],
    embedding_cache_path: Path,
) -> tuple[
    list[Utterance],
    AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
]:
    """Loads + prepares accurate training payload using Whisper backend wiring."""
    return prepare_accurate_training_payload(
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=ensure_dataset_consents_for_training,
        prepare_training_features=lambda training_utterances: (
            prepare_accurate_training_features(
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


def prepare_accurate_research_training(
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
    build_backend: Callable[[str, str, str], Emotion2VecBackend],
    build_feature_dataset: Callable[
        [list[Utterance], Emotion2VecBackend, EmbeddingCache, str],
        tuple[np.ndarray, list[str], list[_MetaT]],
    ],
    embedding_cache_path: Path,
) -> tuple[
    list[Utterance],
    AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
]:
    """Loads + prepares accurate-research payload using emotion2vec backend wiring."""
    return prepare_accurate_training_payload(
        logger=logger,
        load_utterances_for_training=load_utterances_for_training,
        ensure_dataset_consents_for_training=ensure_dataset_consents_for_training,
        prepare_training_features=lambda training_utterances: (
            prepare_accurate_training_features(
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


def run_accurate_profile_training_from_prepared(
    *,
    prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
    utterances: list[Utterance],
    settings: AppConfig,
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
    build_dataset_controls: Callable[[list[Utterance]], dict[str, object]],
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
    report_destination: Path,
) -> dict[str, object]:
    """Runs accurate-profile training from a prepared payload."""
    return run_accurate_profile_training(
        prepared=prepared,
        utterances=utterances,
        settings=settings,
        logger=logger,
        profile_label=profile_label,
        backend_id=backend_id,
        profile_id=profile_id,
        pooling_strategy=pooling_strategy,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        create_classifier=create_classifier,
        min_support=min_support,
        evaluate_predictions=evaluate_predictions,
        attach_grouped_metrics=attach_grouped_metrics,
        build_model_artifact=build_model_artifact,
        extract_artifact_metadata=extract_artifact_metadata,
        persist_model_artifacts=persist_model_artifacts,
        build_provenance_metadata=build_provenance_metadata,
        build_dataset_controls=build_dataset_controls,
        build_grouped_evaluation_controls=build_grouped_evaluation_controls,
        build_training_report=build_training_report,
        persist_training_report=persist_training_report,
        report_destination=report_destination,
    )


def build_prepared_accurate_training_runner(
    *,
    logger: logging.Logger,
    create_classifier: Callable[[], _ModelT],
    min_support_resolver: Callable[[], int],
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
    build_grouped_evaluation_controls: Callable[
        [MediumSplitMetadata],
        dict[str, object],
    ],
    build_training_report: Callable[..., dict[str, object]],
    persist_training_report: Callable[[dict[str, object], Path], None],
) -> PreparedAccurateTrainingRunner:
    """Builds a prepared-training runner bound to shared accurate hooks."""

    def _run(
        *,
        prepared: AccurateTrainingPreparation[Utterance, MediumSplitMetadata, Any],
        utterances: list[Utterance],
        settings: AppConfig,
        profile_label: str,
        backend_id: str,
        profile_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
    ) -> None:
        _ = run_accurate_profile_training_from_prepared(
            prepared=prepared,
            utterances=utterances,
            settings=settings,
            logger=logger,
            profile_label=profile_label,
            backend_id=backend_id,
            profile_id=profile_id,
            pooling_strategy="mean_std",
            frame_size_seconds=frame_size_seconds,
            frame_stride_seconds=frame_stride_seconds,
            create_classifier=create_classifier,
            min_support=min_support_resolver(),
            evaluate_predictions=evaluate_predictions,
            attach_grouped_metrics=attach_grouped_metrics,
            build_model_artifact=build_model_artifact,
            extract_artifact_metadata=extract_artifact_metadata,
            persist_model_artifacts=persist_model_artifacts,
            build_provenance_metadata=build_provenance_metadata,
            build_dataset_controls=build_dataset_controls,
            build_grouped_evaluation_controls=build_grouped_evaluation_controls,
            build_training_report=build_training_report,
            persist_training_report=persist_training_report,
            report_destination=settings.models.training_report_file,
        )

    return _run


def train_accurate_whisper_profile_entrypoint(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    train_profile_model: TrainAccurateProfileModelCallable,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[..., None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id_for_settings: Callable[[AppConfig], str],
    resolve_runtime_selectors_for_backend: ResolveRuntimeSelectorsForBackend,
    build_backend_for_settings: Callable[[str, str, str, AppConfig], WhisperBackend],
    build_feature_dataset_for_backend: BuildAccurateFeatureDatasetForBackend,
    run_prepared_accurate_profile_training: PreparedAccurateTrainingRunner,
    backend_id: str,
    profile_id: str,
    profile_label: str,
    frame_size_seconds: float,
    frame_stride_seconds: float,
) -> None:
    """Runs accurate-whisper entrypoint wiring while preserving helper contract keys."""
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
        build_feature_dataset=lambda utterances, backend, cache, model_id: (
            build_feature_dataset_for_backend(
                utterances=utterances,
                backend=backend,
                cache=cache,
                model_id=model_id,
                backend_id=backend_id,
            )
        ),
        run_prepared_training=lambda prepared, utterances, active_settings: (
            run_prepared_accurate_profile_training(
                prepared=prepared,
                utterances=utterances,
                settings=active_settings,
                profile_label=profile_label,
                backend_id=backend_id,
                profile_id=profile_id,
                frame_size_seconds=frame_size_seconds,
                frame_stride_seconds=frame_stride_seconds,
            )
        ),
    )


def train_accurate_research_profile_entrypoint(
    *,
    settings: AppConfig,
    logger: logging.Logger,
    train_profile_model: TrainAccurateProfileModelCallable,
    parse_allowed_restricted_backends_env: Callable[[], object],
    load_persisted_backend_consents: Callable[..., object],
    ensure_backend_access: Callable[..., object],
    restricted_backend_id: str,
    load_utterances_for_training: Callable[[], list[Utterance] | None],
    ensure_dataset_consents_for_training: Callable[..., None],
    split_utterances: Callable[
        [list[Utterance]],
        tuple[list[Utterance], list[Utterance], MediumSplitMetadata],
    ],
    resolve_model_id_for_settings: Callable[[AppConfig], str],
    resolve_runtime_selectors_for_backend: ResolveRuntimeSelectorsForBackend,
    build_backend_for_settings: Callable[[str, str, AppConfig], Emotion2VecBackend],
    build_feature_dataset_for_backend: BuildAccurateFeatureDatasetForBackend,
    run_prepared_accurate_profile_training: PreparedAccurateTrainingRunner,
    backend_id: str,
    profile_id: str,
    profile_label: str,
    frame_size_seconds: float,
    frame_stride_seconds: float,
) -> None:
    """Runs accurate-research entrypoint wiring while preserving helper contract keys."""
    train_profile_model(
        settings=settings,
        logger=logger,
        parse_allowed_restricted_backends_env=parse_allowed_restricted_backends_env,
        load_persisted_backend_consents=load_persisted_backend_consents,
        ensure_backend_access=ensure_backend_access,
        restricted_backend_id=restricted_backend_id,
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
        build_backend=lambda model_id, runtime_device, _runtime_dtype, active_settings: (
            build_backend_for_settings(
                model_id,
                runtime_device,
                active_settings,
            )
        ),
        build_feature_dataset=lambda utterances, backend, cache, model_id: (
            build_feature_dataset_for_backend(
                utterances=utterances,
                backend=backend,
                cache=cache,
                model_id=model_id,
                backend_id=backend_id,
            )
        ),
        run_prepared_training=lambda prepared, utterances, active_settings: (
            run_prepared_accurate_profile_training(
                prepared=prepared,
                utterances=utterances,
                settings=active_settings,
                profile_label=profile_label,
                backend_id=backend_id,
                profile_id=profile_id,
                frame_size_seconds=frame_size_seconds,
                frame_stride_seconds=frame_stride_seconds,
            )
        ),
    )


__all__ = [
    "build_prepared_accurate_training_runner",
    "prepare_accurate_research_training",
    "prepare_accurate_whisper_training",
    "PreparedAccurateTrainingRunner",
    "run_accurate_profile_training_from_prepared",
    "train_accurate_research_profile_entrypoint",
    "train_accurate_whisper_profile_model",
    "train_accurate_whisper_profile_entrypoint",
    "train_accurate_research_profile_model",
]
