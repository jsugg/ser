"""Speaker-grouped evaluation helpers for SER experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

type FeatureMatrix = NDArray[np.float64]
type IndexArray = NDArray[np.int64]
type FoldIndices = tuple[IndexArray, IndexArray]


@dataclass(frozen=True)
class GroupedSplit:
    """Train/test split output with explicit sample indices."""

    x_train: FeatureMatrix
    x_test: FeatureMatrix
    y_train: list[str]
    y_test: list[str]
    train_indices: IndexArray
    test_indices: IndexArray


def extract_ravdess_speaker_id(file_name: str) -> str | None:
    """Extracts actor ID from a RAVDESS-style audio filename."""
    normalized_name = Path(file_name).name
    parts = normalized_name.split("-")
    if len(parts) < 7:
        return None
    speaker_id = parts[6].split(".")[0].strip()
    return speaker_id or None


def _validate_grouped_inputs(
    features: FeatureMatrix, labels: Sequence[str], speaker_ids: Sequence[str]
) -> None:
    """Validates grouped-split input arrays and sequence lengths."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    sample_count = int(features.shape[0])
    if sample_count == 0:
        raise ValueError("features must contain at least one sample.")
    if len(labels) != sample_count:
        raise ValueError("labels length must match number of feature rows.")
    if len(speaker_ids) != sample_count:
        raise ValueError("speaker_ids length must match number of feature rows.")
    if len(set(speaker_ids)) < 2:
        raise ValueError("At least two distinct speaker IDs are required.")


def grouped_train_test_split(
    features: FeatureMatrix,
    labels: Sequence[str],
    speaker_ids: Sequence[str],
    *,
    test_size: float,
    random_state: int,
) -> GroupedSplit:
    """Builds a train/test split with disjoint speaker groups.

    Args:
        features: Feature matrix with shape `(n_samples, n_features)`.
        labels: Emotion label per sample.
        speaker_ids: Speaker identifier per sample.
        test_size: Fraction in `(0, 1)` reserved for test split.
        random_state: Deterministic random seed.

    Returns:
        Grouped train/test split with selected indices.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    _validate_grouped_inputs(features, labels, speaker_ids)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(features, labels, groups=speaker_ids))
    train_indices = np.asarray(train_idx, dtype=np.int64)
    test_indices = np.asarray(test_idx, dtype=np.int64)

    return GroupedSplit(
        x_train=np.asarray(features[train_indices], dtype=np.float64),
        x_test=np.asarray(features[test_indices], dtype=np.float64),
        y_train=[str(labels[index]) for index in train_indices.tolist()],
        y_test=[str(labels[index]) for index in test_indices.tolist()],
        train_indices=train_indices,
        test_indices=test_indices,
    )


def speaker_independent_cv(
    features: FeatureMatrix,
    labels: Sequence[str],
    speaker_ids: Sequence[str],
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[FoldIndices, ...]:
    """Builds stratified speaker-group cross-validation folds.

    Args:
        features: Feature matrix with shape `(n_samples, n_features)`.
        labels: Emotion label per sample.
        speaker_ids: Speaker identifier per sample.
        n_splits: Number of folds (must be at least two).
        random_state: Deterministic random seed.

    Returns:
        A tuple of `(train_indices, test_indices)` fold pairs.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be greater than or equal to 2.")
    _validate_grouped_inputs(features, labels, speaker_ids)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds: list[FoldIndices] = []
    for train_idx, test_idx in splitter.split(features, labels, groups=speaker_ids):
        folds.append(
            (
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(test_idx, dtype=np.int64),
            )
        )
    return tuple(folds)
