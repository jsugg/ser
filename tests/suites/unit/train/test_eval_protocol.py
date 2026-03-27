"""Tests for speaker-grouped evaluation helpers."""

import numpy as np
import pytest

from ser.train.eval import (
    extract_ravdess_speaker_id,
    grouped_train_test_split,
    speaker_independent_cv,
)


def _fixture_grouped_samples() -> tuple[np.ndarray, list[str], list[str]]:
    """Returns deterministic sample matrix + labels + speaker IDs."""
    features = np.asarray([[float(index), float(index + 1)] for index in range(12)])
    labels = ["happy", "sad"] * 6
    speaker_ids = [speaker for speaker in ("01", "02", "03", "04", "05", "06") for _ in range(2)]
    return features, labels, speaker_ids


def test_extract_ravdess_speaker_id_parses_actor_token() -> None:
    """Speaker ID should be parsed from valid RAVDESS naming conventions."""
    assert extract_ravdess_speaker_id("03-01-05-01-02-01-24.wav") == "24"
    assert extract_ravdess_speaker_id("/tmp/03-01-02-01-01-01-09.wav") == "09"


def test_extract_ravdess_speaker_id_returns_none_for_invalid_name() -> None:
    """Invalid names should not produce speaker IDs."""
    assert extract_ravdess_speaker_id("invalid.wav") is None
    assert extract_ravdess_speaker_id("03-01-05.wav") is None


def test_grouped_train_test_split_keeps_speakers_disjoint() -> None:
    """Train and test partitions should never share speaker IDs."""
    features, labels, speaker_ids = _fixture_grouped_samples()

    split = grouped_train_test_split(
        features=features,
        labels=labels,
        speaker_ids=speaker_ids,
        test_size=0.34,
        random_state=7,
    )

    train_speakers = {speaker_ids[index] for index in split.train_indices.tolist()}
    test_speakers = {speaker_ids[index] for index in split.test_indices.tolist()}
    assert train_speakers.isdisjoint(test_speakers)
    assert split.x_train.shape[0] + split.x_test.shape[0] == features.shape[0]
    assert len(split.y_train) + len(split.y_test) == len(labels)


def test_grouped_train_test_split_validates_length_mismatches() -> None:
    """Mismatched label/speaker lengths should fail fast."""
    features, labels, speaker_ids = _fixture_grouped_samples()

    with pytest.raises(ValueError, match="labels length"):
        grouped_train_test_split(
            features=features,
            labels=labels[:-1],
            speaker_ids=speaker_ids,
            test_size=0.25,
            random_state=42,
        )

    with pytest.raises(ValueError, match="speaker_ids length"):
        grouped_train_test_split(
            features=features,
            labels=labels,
            speaker_ids=speaker_ids[:-1],
            test_size=0.25,
            random_state=42,
        )


def test_speaker_independent_cv_produces_disjoint_group_folds() -> None:
    """Each CV fold must keep train/test speaker groups disjoint."""
    features, labels, speaker_ids = _fixture_grouped_samples()

    folds = speaker_independent_cv(
        features=features,
        labels=labels,
        speaker_ids=speaker_ids,
        n_splits=3,
        random_state=42,
    )

    assert len(folds) == 3
    observed_test_indices: set[int] = set()
    for train_idx, test_idx in folds:
        train_speakers = {speaker_ids[index] for index in train_idx.tolist()}
        test_speakers = {speaker_ids[index] for index in test_idx.tolist()}
        assert train_speakers.isdisjoint(test_speakers)
        observed_test_indices.update(test_idx.tolist())

    assert observed_test_indices == set(range(features.shape[0]))
