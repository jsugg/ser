"""Accurate-profile encoding and dataset-build orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import partial
from typing import TypeVar

import numpy as np

from ser.config import AppConfig
from ser.data import EmbeddingCache
from ser.models.feature_runtime_encoding import (
    encode_sequence_with_cache,
    resolve_accurate_runtime_config,
    resolve_effective_model_id,
)
from ser.models.profile_runtime import (
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
)
from ser.models.profile_training_preparation import (
    UtteranceLike,
)
from ser.models.profile_training_preparation import (
    build_accurate_feature_dataset as _build_prepared_accurate_feature_dataset,
)
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import EncodedSequence, FeatureBackend, PoolingWindow
from ser.utils.audio_utils import read_audio_file

_UtteranceT = TypeVar("_UtteranceT", bound=UtteranceLike)
_MetaT = TypeVar("_MetaT")


def _build_pooling_windows(
    encoded: EncodedSequence,
    window_size_seconds: float,
    window_stride_seconds: float,
) -> list[PoolingWindow]:
    """Builds temporal pooling windows for one encoded sequence."""
    return temporal_pooling_windows(
        encoded,
        window_size_seconds=window_size_seconds,
        window_stride_seconds=window_stride_seconds,
    )


def encode_accurate_sequence(
    *,
    audio_path: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    backend: FeatureBackend,
    cache: EmbeddingCache,
    model_id: str | None,
    backend_id: str,
    settings: AppConfig,
    accurate_backend_id: str,
    accurate_research_backend_id: str,
    logger: logging.Logger,
) -> EncodedSequence:
    """Encodes one accurate-profile sequence with deterministic cache keys."""
    runtime_config = resolve_accurate_runtime_config(
        settings=settings,
        backend_id=backend_id,
        accurate_backend_id=accurate_backend_id,
        accurate_research_backend_id=accurate_research_backend_id,
    )
    default_model_id = (
        resolve_accurate_model_id(settings)
        if backend_id == accurate_backend_id
        else resolve_accurate_research_model_id(settings)
    )
    resolved_model_id = resolve_effective_model_id(
        model_id,
        default_model_id=default_model_id,
    )
    return encode_sequence_with_cache(
        audio_path=audio_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        backend=backend,
        cache=cache,
        backend_id=backend_id,
        model_id=resolved_model_id,
        frame_size_seconds=runtime_config.pool_window_size_seconds,
        frame_stride_seconds=runtime_config.pool_window_stride_seconds,
        log_prefix=f"Accurate backend {backend_id}",
        logger=logger,
        read_audio=partial(
            read_audio_file,
            audio_read_config=settings.audio_read,
        ),
    )


def build_accurate_feature_dataset(
    *,
    utterances: Sequence[_UtteranceT],
    backend: FeatureBackend,
    cache: EmbeddingCache,
    model_id: str | None,
    backend_id: str,
    settings: AppConfig,
    accurate_backend_id: str,
    accurate_research_backend_id: str,
    logger: logging.Logger,
    window_meta_factory: Callable[[str, str, str], _MetaT],
    build_pooling_windows: Callable[
        [EncodedSequence, float, float],
        Sequence[PoolingWindow],
    ] = _build_pooling_windows,
    pool_features: Callable[
        [EncodedSequence, Sequence[PoolingWindow]],
        np.ndarray,
    ] = mean_std_pool,
    encode_sequence: Callable[..., EncodedSequence] = encode_accurate_sequence,
) -> tuple[np.ndarray, list[str], list[_MetaT]]:
    """Builds an accurate-profile pooled feature matrix from utterance inputs."""
    runtime_config = resolve_accurate_runtime_config(
        settings=settings,
        backend_id=backend_id,
        accurate_backend_id=accurate_backend_id,
        accurate_research_backend_id=accurate_research_backend_id,
    )

    def _encode_utterance(utterance: _UtteranceT) -> EncodedSequence:
        return encode_sequence(
            audio_path=str(utterance.audio_path),
            start_seconds=utterance.start_seconds,
            duration_seconds=utterance.duration_seconds,
            backend=backend,
            cache=cache,
            model_id=model_id,
            backend_id=backend_id,
            settings=settings,
            accurate_backend_id=accurate_backend_id,
            accurate_research_backend_id=accurate_research_backend_id,
            logger=logger,
        )

    return _build_prepared_accurate_feature_dataset(
        utterances=utterances,
        encode_sequence=_encode_utterance,
        window_size_seconds=runtime_config.pool_window_size_seconds,
        window_stride_seconds=runtime_config.pool_window_stride_seconds,
        build_pooling_windows=build_pooling_windows,
        pool_features=pool_features,
        window_meta_factory=window_meta_factory,
    )


__all__ = [
    "build_accurate_feature_dataset",
    "encode_accurate_sequence",
]
