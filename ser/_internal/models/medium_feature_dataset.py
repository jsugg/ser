"""Medium-profile encoding and dataset-build orchestration helpers."""

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
    resolve_effective_model_id,
)
from ser.models.medium_noise_controls import MediumNoiseControlStats
from ser.models.profile_runtime import resolve_medium_model_id
from ser.models.profile_training_preparation import (
    UtteranceLike,
)
from ser.models.profile_training_preparation import (
    build_medium_feature_dataset as _build_prepared_medium_feature_dataset,
)
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import EncodedSequence, PoolingWindow, XLSRBackend
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


def encode_medium_sequence(
    *,
    audio_path: str,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
    backend: XLSRBackend,
    cache: EmbeddingCache,
    model_id: str | None,
    settings: AppConfig,
    backend_id: str,
    logger: logging.Logger,
) -> EncodedSequence:
    """Encodes one medium-profile sequence with deterministic cache keys."""
    resolved_model_id = resolve_effective_model_id(
        model_id,
        default_model_id=resolve_medium_model_id(settings),
    )
    return encode_sequence_with_cache(
        audio_path=audio_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        backend=backend,
        cache=cache,
        backend_id=backend_id,
        model_id=resolved_model_id,
        frame_size_seconds=settings.medium_runtime.pool_window_size_seconds,
        frame_stride_seconds=settings.medium_runtime.pool_window_stride_seconds,
        log_prefix="Medium",
        logger=logger,
        read_audio=partial(
            read_audio_file,
            audio_read_config=settings.audio_read,
        ),
    )


def build_medium_feature_dataset(
    *,
    utterances: Sequence[_UtteranceT],
    backend: XLSRBackend,
    cache: EmbeddingCache,
    model_id: str | None,
    settings: AppConfig,
    apply_noise_controls: Callable[
        [np.ndarray],
        tuple[np.ndarray, MediumNoiseControlStats],
    ],
    merge_noise_stats: Callable[
        [MediumNoiseControlStats, MediumNoiseControlStats],
        MediumNoiseControlStats,
    ],
    window_meta_factory: Callable[[str, str, str], _MetaT],
    build_pooling_windows: Callable[
        [EncodedSequence, float, float],
        Sequence[PoolingWindow],
    ] = _build_pooling_windows,
    pool_features: Callable[
        [EncodedSequence, Sequence[PoolingWindow]],
        np.ndarray,
    ] = mean_std_pool,
    encode_sequence: Callable[..., EncodedSequence] = encode_medium_sequence,
) -> tuple[np.ndarray, list[str], list[_MetaT], MediumNoiseControlStats]:
    """Builds a medium-profile pooled feature matrix from utterance inputs."""
    runtime_config = settings.medium_runtime

    def _encode_utterance(utterance: _UtteranceT) -> EncodedSequence:
        return encode_sequence(
            audio_path=str(utterance.audio_path),
            start_seconds=utterance.start_seconds,
            duration_seconds=utterance.duration_seconds,
            backend=backend,
            cache=cache,
            model_id=model_id,
        )

    return _build_prepared_medium_feature_dataset(
        utterances=utterances,
        encode_sequence=_encode_utterance,
        window_size_seconds=runtime_config.pool_window_size_seconds,
        window_stride_seconds=runtime_config.pool_window_stride_seconds,
        build_pooling_windows=build_pooling_windows,
        pool_features=pool_features,
        apply_noise_controls=apply_noise_controls,
        merge_noise_stats=merge_noise_stats,
        initial_noise_stats=MediumNoiseControlStats(),
        window_meta_factory=window_meta_factory,
    )


__all__ = ["build_medium_feature_dataset", "encode_medium_sequence"]
