"""Accurate-profile training preparation helpers for entrypoint owners."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np

from ser._internal.data import EmbeddingCache, Utterance
from ser._internal.models.accurate_training_execution import (
    PreparedAccurateTrainingRunner,
)
from ser._internal.models.dataset_splitting import (
    MediumSplitMetadata,
    medium_split_metadata_from_mapping,
)
from ser._internal.models.profile_runtime import (
    ACCURATE_BACKEND_ID,
    ACCURATE_RESEARCH_BACKEND_ID,
)
from ser._internal.models.training_orchestration import (
    canonical_train_partition,
    current_training_state,
    prepare_until_quarantine_stable,
    publish_prepared_features,
    read_prepared_feature_payload,
    reuse_checked_backend,
    training_meta_sample_ids,
    validate_operation_plan,
)
from ser._internal.models.training_preparation import (
    prepare_accurate_training_features,
    prepare_accurate_training_payload,
)
from ser._internal.models.training_readiness import TrainingMode
from ser._internal.models.training_support import WindowMeta
from ser._internal.models.training_types import (
    AccurateTrainingPreparation,
)
from ser._internal.repr import Emotion2VecBackend, WhisperBackend
from ser.config import AppConfig

_MetaT = TypeVar("_MetaT")


class ResolveRuntimeSelectorsForBackend(Protocol):
    """Callable contract for backend-aware runtime selector resolution."""

    def __call__(self, *, settings: AppConfig, backend_id: str) -> tuple[str, str]: ...


type BuildAccurateFeatureDatasetForBackend = Callable[..., tuple[np.ndarray, list[str], list[Any]]]
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
    resolved_model_id = resolve_model_id_for_settings(settings)
    resolved_device, resolved_dtype = resolve_runtime_selectors_for_settings(settings)
    plan = validate_operation_plan(
        settings=settings,
        backend_id=ACCURATE_BACKEND_ID,
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
        test_meta_raw = cast(list[dict[str, str]], metadata.get("test_meta", []))
        train_meta_raw = cast(list[dict[str, str]], metadata.get("train_meta", []))
        utterances = list(current_training_state().utterances)
        prepared = cast(
            AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
            AccurateTrainingPreparation(
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
            ),
        )
    else:
        utterances, prepared = prepare_until_quarantine_stable(
            settings=settings,
            prepare=lambda: prepare_accurate_whisper_training(
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
                    WhisperBackend,
                    reuse_checked_backend(
                        backend_id=ACCURATE_BACKEND_ID,
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
    if current_training_state().operation.mode is TrainingMode.PREPARE_ONLY:
        publish_prepared_features(
            settings=settings,
            backend_id=ACCURATE_BACKEND_ID,
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
            },
            cache_namespace=embedding_cache_name,
            windowing_policy={
                "pool_window_size_seconds": settings.accurate_runtime.pool_window_size_seconds,
                "pool_window_stride_seconds": settings.accurate_runtime.pool_window_stride_seconds,
                "pooling_strategy": "mean_std",
            },
            noise_statistics={},
        )
        return
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
        prepared = replace(
            prepared,
            train_utterances=train_utterances,
            x_train=filtered_x,
            y_train=filtered_y,
            train_meta=filtered_meta,
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
    resolved_model_id = resolve_model_id_for_settings(settings)
    resolved_device, resolved_dtype = resolve_runtime_selectors_for_settings(settings)
    plan = validate_operation_plan(
        settings=settings,
        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
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
        test_meta_raw = cast(list[dict[str, str]], metadata.get("test_meta", []))
        train_meta_raw = cast(list[dict[str, str]], metadata.get("train_meta", []))
        utterances = list(current_training_state().utterances)
        prepared = cast(
            AccurateTrainingPreparation[Utterance, MediumSplitMetadata, _MetaT],
            AccurateTrainingPreparation(
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
            ),
        )
    else:
        utterances, prepared = prepare_until_quarantine_stable(
            settings=settings,
            prepare=lambda: prepare_accurate_research_training(
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
                    Emotion2VecBackend,
                    reuse_checked_backend(
                        backend_id=ACCURATE_RESEARCH_BACKEND_ID,
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
    if current_training_state().operation.mode is TrainingMode.PREPARE_ONLY:
        publish_prepared_features(
            settings=settings,
            backend_id=ACCURATE_RESEARCH_BACKEND_ID,
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
            },
            cache_namespace=embedding_cache_name,
            windowing_policy={
                "pool_window_size_seconds": (
                    settings.accurate_research_runtime.pool_window_size_seconds
                ),
                "pool_window_stride_seconds": (
                    settings.accurate_research_runtime.pool_window_stride_seconds
                ),
                "pooling_strategy": "mean_std",
            },
            noise_statistics={},
        )
        return
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
        prepared = replace(
            prepared,
            train_utterances=train_utterances,
            x_train=filtered_x,
            y_train=filtered_y,
            train_meta=filtered_meta,
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
        prepare_training_features=lambda training_utterances: prepare_accurate_training_features(
            utterances=training_utterances,
            split_utterances=split_utterances,
            resolve_model_id=resolve_model_id,
            resolve_runtime_selectors=resolve_runtime_selectors,
            build_backend=build_backend,
            build_cache=lambda: EmbeddingCache(embedding_cache_path),
            build_feature_dataset=build_feature_dataset,
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
        prepare_training_features=lambda training_utterances: prepare_accurate_training_features(
            utterances=training_utterances,
            split_utterances=split_utterances,
            resolve_model_id=resolve_model_id,
            resolve_runtime_selectors=resolve_runtime_selectors,
            build_backend=build_backend,
            build_cache=lambda: EmbeddingCache(embedding_cache_path),
            build_feature_dataset=build_feature_dataset,
        ),
    )


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
    "prepare_accurate_research_training",
    "prepare_accurate_whisper_training",
    "train_accurate_research_profile_entrypoint",
    "train_accurate_whisper_profile_model",
    "train_accurate_whisper_profile_entrypoint",
    "train_accurate_research_profile_model",
]
