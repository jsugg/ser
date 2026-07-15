"""Medium-profile training preparation helpers for emotion-model entrypoints."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol, TypeVar, cast

import numpy as np

from ser._internal.data import EmbeddingCache, Utterance
from ser._internal.models.dataset_splitting import (
    MediumSplitMetadata,
    medium_split_metadata_from_mapping,
)
from ser._internal.models.medium_noise_controls import (
    MediumNoiseControlStats,
    merge_medium_noise_stats,
)
from ser._internal.models.training_execution import run_medium_profile_training
from ser._internal.models.training_orchestration import (
    canonical_train_partition,
    current_training_state,
    prepare_until_quarantine_stable,
    publish_prepared_features,
    read_prepared_feature_payload,
    record_dropped_windows,
    reuse_checked_backend,
    training_meta_sample_ids,
    validate_operation_plan,
)
from ser._internal.models.training_preparation import (
    prepare_medium_training_features,
    prepare_medium_training_payload,
)
from ser._internal.models.training_readiness import TrainingMode
from ser._internal.models.training_support import WindowMeta
from ser._internal.models.training_types import (
    MediumTrainingPreparation,
    PersistedArtifactsLike,
    TrainingEvaluation,
)
from ser._internal.repr import XLSRBackend
from ser.config import AppConfig

_MetaT = TypeVar("_MetaT")
_NoiseStatsT = TypeVar("_NoiseStatsT")
_ModelT = TypeVar("_ModelT")


class ResolveRuntimeSelectorsForBackend(Protocol):
    """Callable contract for backend-aware runtime selector resolution."""

    def __call__(self, *, settings: AppConfig, backend_id: str) -> tuple[str, str]: ...


type TrainMediumProfileModelCallable = Callable[..., None]


def _partition_noise_stats(sample_ids: set[str]) -> MediumNoiseControlStats:
    """Aggregates exact per-sample medium noise stats for one canonical partition."""
    aggregate = MediumNoiseControlStats()
    by_sample = current_training_state().medium_noise_stats_by_sample
    missing = sample_ids.difference(by_sample)
    if missing:
        raise RuntimeError(
            f"Medium noise provenance is missing {len(missing)} canonical sample(s)."
        )
    for sample_id in sorted(sample_ids):
        stats = by_sample[sample_id]
        if not isinstance(stats, MediumNoiseControlStats):
            raise TypeError("Medium noise provenance contains an invalid statistics payload.")
        aggregate = merge_medium_noise_stats(aggregate, stats)
    return aggregate


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
        prepare_training_features=lambda training_utterances: prepare_medium_training_features(
            utterances=training_utterances,
            split_utterances=split_utterances,
            resolve_model_id=resolve_model_id,
            resolve_runtime_selectors=resolve_runtime_selectors,
            build_backend=build_backend,
            build_cache=lambda: EmbeddingCache(embedding_cache_path),
            build_feature_dataset=build_feature_dataset,
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
    resolved_model_id = resolve_model_id_for_settings(settings)
    resolved_device, resolved_dtype = resolve_runtime_selectors_for_settings(settings)
    plan = validate_operation_plan(
        settings=settings,
        backend_id=backend_id,
        model_id=resolved_model_id,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    if plan is not None:
        payload = read_prepared_feature_payload(plan)
        metadata = payload.metadata
        by_id = {item.sample_id: item for item in current_training_state().utterances}
        train_ids = cast(list[str], metadata.get("train_sample_ids", []))
        test_ids = cast(list[str], metadata.get("test_sample_ids", []))
        split_raw = cast(dict[str, object], metadata.get("split_metadata", {}))
        train_stats_raw = cast(dict[str, int], metadata.get("train_noise_stats", {}))
        test_stats_raw = cast(dict[str, int], metadata.get("test_noise_stats", {}))
        test_meta_raw = cast(list[dict[str, str]], metadata.get("test_meta", []))
        train_meta_raw = cast(list[dict[str, str]], metadata.get("train_meta", []))
        utterances = list(current_training_state().utterances)
        prepared = cast(
            MediumTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT, _NoiseStatsT],
            MediumTrainingPreparation(
                train_utterances=[by_id[item] for item in train_ids],
                test_utterances=[by_id[item] for item in test_ids],
                split_metadata=medium_split_metadata_from_mapping(split_raw),
                model_id=resolved_model_id,
                runtime_device=resolved_device,
                runtime_dtype=resolved_dtype,
                x_train=payload.x_train,
                y_train=payload.y_train,
                train_meta=[WindowMeta(**item) for item in train_meta_raw],
                x_test=payload.x_test,
                y_test=payload.y_test,
                test_meta=[WindowMeta(**item) for item in test_meta_raw],
                train_noise_stats=MediumNoiseControlStats(**train_stats_raw),
                test_noise_stats=MediumNoiseControlStats(**test_stats_raw),
            ),
        )
    else:
        utterances, prepared = prepare_until_quarantine_stable(
            settings=settings,
            prepare=lambda: prepare_medium_xlsr_training(
                settings=settings,
                logger=logger,
                load_utterances_for_training=lambda: list(current_training_state().utterances),
                ensure_dataset_consents_for_training=lambda training_utterances: (
                    ensure_dataset_consents_for_training(utterances=training_utterances)
                ),
                split_utterances=split_utterances,
                resolve_model_id=lambda: resolved_model_id,
                resolve_runtime_selectors=lambda: (resolved_device, resolved_dtype),
                build_backend=lambda model_id, runtime_device, runtime_dtype: cast(
                    XLSRBackend,
                    reuse_checked_backend(
                        backend_id=backend_id,
                        model_id=model_id,
                        device=runtime_device,
                        dtype=runtime_dtype,
                        build=lambda: build_backend(
                            model_id,
                            runtime_device,
                            runtime_dtype,
                            settings,
                        ),
                    ),
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
            ),
        )
    canonical_prepared = prepared
    dev_noise_stats: MediumNoiseControlStats | None = None
    if (
        plan is None
        and current_training_state().readiness is not None
        and current_training_state().utterances
    ):
        filtered_x, filtered_y, filtered_meta, train_utterances = canonical_train_partition(
            settings=settings,
            x_train=np.asarray(prepared.x_train, dtype=np.float64),
            y_train=prepared.y_train,
            train_metadata=prepared.train_meta,
            sample_id=lambda item: training_meta_sample_ids([item])[0],
        )
        canonical_train_ids = {item.sample_id for item in train_utterances}
        dev_ids = {
            item.sample_id
            for item in prepared.train_utterances
            if item.sample_id not in canonical_train_ids
        }
        canonical_prepared = replace(
            prepared,
            train_utterances=train_utterances,
            x_train=filtered_x,
            y_train=filtered_y,
            train_meta=filtered_meta,
            train_noise_stats=cast(
                _NoiseStatsT,
                _partition_noise_stats(canonical_train_ids),
            ),
        )
        dev_noise_stats = _partition_noise_stats(dev_ids)
    dropped_windows = 0
    run_noise_stats: tuple[object, ...] = (
        getattr(canonical_prepared, "train_noise_stats", None),
        getattr(prepared, "test_noise_stats", None),
    )
    if current_training_state().operation.mode is TrainingMode.PREPARE_ONLY:
        run_noise_stats = (*run_noise_stats, dev_noise_stats)
    for noise_stats in run_noise_stats:
        if isinstance(noise_stats, MediumNoiseControlStats):
            dropped_windows += noise_stats.dropped_low_std_windows
            dropped_windows += noise_stats.dropped_cap_windows
    record_dropped_windows(dropped_windows)
    if current_training_state().operation.mode is TrainingMode.PREPARE_ONLY:
        publish_prepared_features(
            settings=settings,
            backend_id=backend_id,
            model_id=prepared.model_id,
            device=prepared.runtime_device,
            dtype=prepared.runtime_dtype,
            utterances=utterances,
            x_train=np.asarray(prepared.x_train, dtype=np.float64),
            x_test=np.asarray(prepared.x_test, dtype=np.float64),
            y_train=prepared.y_train,
            y_test=prepared.y_test,
            metadata={
                "train_sample_ids": [item.sample_id for item in prepared.train_utterances],
                "test_sample_ids": [item.sample_id for item in prepared.test_utterances],
                "split_metadata": prepared.split_metadata,
                "test_meta": prepared.test_meta,
                "train_meta": prepared.train_meta,
                "train_noise_stats": canonical_prepared.train_noise_stats,
                "dev_noise_stats": dev_noise_stats,
                "test_noise_stats": prepared.test_noise_stats,
            },
            cache_namespace=embedding_cache_name,
            windowing_policy={
                "pool_window_size_seconds": settings.medium_runtime.pool_window_size_seconds,
                "pool_window_stride_seconds": settings.medium_runtime.pool_window_stride_seconds,
                "pooling_strategy": pooling_strategy,
            },
            noise_statistics={
                "train": canonical_prepared.train_noise_stats,
                "dev": dev_noise_stats,
                "test": prepared.test_noise_stats,
            },
        )
        return
    _ = run_medium_profile_training_from_prepared(
        prepared=canonical_prepared,
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
