"""Shared medium/accurate training dataset preparation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar

import numpy as np

from ser._internal.repr import EncodedSequence, PoolingWindow

_MetaT = TypeVar("_MetaT")
_StatsT = TypeVar("_StatsT")
_UtteranceT = TypeVar("_UtteranceT", bound="UtteranceLike")


class UtteranceLike(Protocol):
    """Minimal utterance contract required by training preparation helpers."""

    @property
    def audio_path(self) -> Path: ...

    @property
    def start_seconds(self) -> float | None: ...

    @property
    def duration_seconds(self) -> float | None: ...

    @property
    def label(self) -> str | None: ...

    @property
    def language(self) -> str | None: ...

    @property
    def sample_id(self) -> str: ...

    @property
    def corpus(self) -> str: ...


def build_medium_feature_dataset(
    *,
    utterances: Sequence[_UtteranceT],
    encode_sequence: Callable[[_UtteranceT], EncodedSequence],
    window_size_seconds: float,
    window_stride_seconds: float,
    build_pooling_windows: Callable[
        [EncodedSequence, float, float],
        Sequence[PoolingWindow],
    ],
    pool_features: Callable[[EncodedSequence, Sequence[PoolingWindow]], np.ndarray],
    apply_noise_controls: Callable[[np.ndarray], tuple[np.ndarray, _StatsT]],
    merge_noise_stats: Callable[[_StatsT, _StatsT], _StatsT],
    initial_noise_stats: _StatsT,
    window_meta_factory: Callable[[str, str, str], _MetaT],
    handle_sample_failure: Callable[[_UtteranceT, Exception], bool] | None = None,
) -> tuple[np.ndarray, list[str], list[_MetaT], _StatsT]:
    """Builds a medium-profile feature matrix from utterance-level embeddings."""
    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[_MetaT] = []
    aggregate_stats = initial_noise_stats

    from ser._internal.models.training_orchestration import (  # noqa: TID251
        record_medium_noise_stats,
        record_preparation_progress,
    )

    total = len(utterances)
    for processed, utterance in enumerate(utterances, start=1):
        try:
            encoded = encode_sequence(utterance)
        except Exception as error:
            if handle_sample_failure is not None and handle_sample_failure(utterance, error):
                record_preparation_progress(
                    processed=processed,
                    total=total,
                    sample_id=utterance.sample_id,
                )
                continue
            raise
        windows = build_pooling_windows(
            encoded,
            window_size_seconds,
            window_stride_seconds,
        )
        pooled = pool_features(encoded, windows)
        filtered, stats = apply_noise_controls(np.asarray(pooled, dtype=np.float64))
        feature_blocks.append(filtered)
        row_count = int(filtered.shape[0])
        if utterance.label is None:
            raise ValueError(f"Utterance {utterance.sample_id!r} has no primary emotion target.")
        labels.extend([utterance.label] * row_count)
        language = utterance.language or "unknown"
        meta.extend(
            [window_meta_factory(utterance.sample_id, utterance.corpus, language)] * row_count
        )
        aggregate_stats = merge_noise_stats(aggregate_stats, stats)
        record_medium_noise_stats(sample_id=utterance.sample_id, stats=stats)
        record_preparation_progress(
            processed=processed,
            total=total,
            sample_id=utterance.sample_id,
        )

    if not feature_blocks:
        raise RuntimeError("Medium training produced no feature vectors.")
    feature_matrix = np.vstack(feature_blocks).astype(np.float64, copy=False)
    if int(feature_matrix.shape[0]) != len(labels) or int(feature_matrix.shape[0]) != len(meta):
        raise RuntimeError("Medium feature/label row mismatch during dataset build.")
    return feature_matrix, labels, meta, aggregate_stats


def build_accurate_feature_dataset(
    *,
    utterances: Sequence[_UtteranceT],
    encode_sequence: Callable[[_UtteranceT], EncodedSequence],
    window_size_seconds: float,
    window_stride_seconds: float,
    build_pooling_windows: Callable[
        [EncodedSequence, float, float],
        Sequence[PoolingWindow],
    ],
    pool_features: Callable[[EncodedSequence, Sequence[PoolingWindow]], np.ndarray],
    window_meta_factory: Callable[[str, str, str], _MetaT],
    handle_sample_failure: Callable[[_UtteranceT, Exception], bool] | None = None,
) -> tuple[np.ndarray, list[str], list[_MetaT]]:
    """Builds an accurate-profile feature matrix from utterance embeddings."""
    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[_MetaT] = []

    from ser._internal.models.training_orchestration import (  # noqa: TID251
        record_preparation_progress,
    )

    total = len(utterances)
    for processed, utterance in enumerate(utterances, start=1):
        try:
            encoded = encode_sequence(utterance)
        except Exception as error:
            if handle_sample_failure is not None and handle_sample_failure(utterance, error):
                record_preparation_progress(
                    processed=processed,
                    total=total,
                    sample_id=utterance.sample_id,
                )
                continue
            raise
        windows = build_pooling_windows(
            encoded,
            window_size_seconds,
            window_stride_seconds,
        )
        pooled = pool_features(encoded, windows)
        feature_blocks.append(np.asarray(pooled, dtype=np.float64))
        row_count = int(pooled.shape[0])
        if utterance.label is None:
            raise ValueError(f"Utterance {utterance.sample_id!r} has no primary emotion target.")
        labels.extend([utterance.label] * row_count)
        language = utterance.language or "unknown"
        meta.extend(
            [window_meta_factory(utterance.sample_id, utterance.corpus, language)] * row_count
        )
        record_preparation_progress(
            processed=processed,
            total=total,
            sample_id=utterance.sample_id,
        )

    if not feature_blocks:
        raise RuntimeError("Accurate training produced no feature vectors.")
    feature_matrix = np.vstack(feature_blocks).astype(np.float64, copy=False)
    if int(feature_matrix.shape[0]) != len(labels) or int(feature_matrix.shape[0]) != len(meta):
        raise RuntimeError("Accurate feature/label row mismatch during dataset build.")
    return feature_matrix, labels, meta
