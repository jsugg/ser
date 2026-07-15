"""Tests for corpus/class balanced utterance sampling."""

from __future__ import annotations

import math

from ser._internal.models.utterance_sampling import (
    UtteranceSamplingItem,
    select_training_windows,
    utterance_sampling_distribution,
)


def test_distribution_uses_sqrt_corpus_and_inverse_sqrt_class_weights() -> None:
    """Expected hierarchical mass matches the declared sampling policy."""
    items = [
        UtteranceSamplingItem("a:happy:1", "a", "happy", 10),
        UtteranceSamplingItem("a:happy:2", "a", "happy", 20),
        UtteranceSamplingItem("a:sad:1", "a", "sad", 1),
        UtteranceSamplingItem("b:sad:1", "b", "sad", 1),
    ]
    rows = utterance_sampling_distribution(items)
    corpus_a = sum(row.probability for row in rows if row.corpus == "a")
    corpus_b = sum(row.probability for row in rows if row.corpus == "b")
    expected_a = math.sqrt(3) / (math.sqrt(3) + 1)

    assert math.isclose(corpus_a, expected_a)
    assert math.isclose(corpus_b, 1.0 - expected_a)
    happy_mass = sum(row.probability for row in rows if row.corpus == "a" and row.label == "happy")
    sad_mass = sum(row.probability for row in rows if row.corpus == "a" and row.label == "sad")
    assert sad_mass > happy_mass


def test_training_window_selection_is_bounded_seeded_and_epoch_varying() -> None:
    """Long utterances never materialize an unbounded training-window contribution."""
    first = select_training_windows(
        sample_id="corpus:1", window_count=100, max_windows=4, seed=7, epoch=0
    )
    repeated = select_training_windows(
        sample_id="corpus:1", window_count=100, max_windows=4, seed=7, epoch=0
    )
    next_epoch = select_training_windows(
        sample_id="corpus:1", window_count=100, max_windows=4, seed=7, epoch=1
    )

    assert len(first) == 4
    assert first == repeated
    assert first != next_epoch
