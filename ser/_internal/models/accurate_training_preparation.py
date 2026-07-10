"""Accurate-profile training preparation helpers for entrypoint owners."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from ser.config import AppConfig
from ser.data import EmbeddingCache, Utterance
from ser.models.accurate_training_execution import (
    PreparedAccurateTrainingRunner,
)
from ser.models.dataset_splitting import MediumSplitMetadata
from ser.models.training_preparation import (
    prepare_accurate_training_features,
    prepare_accurate_training_payload,
)
from ser.models.training_types import (
    AccurateTrainingPreparation,
)
from ser.repr import Emotion2VecBackend, WhisperBackend

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
    utterances, prepared = prepare_accurate_whisper_training(
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
