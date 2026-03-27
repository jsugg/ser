"""Tests for internal transcription text-metric helpers."""

from __future__ import annotations

import string

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from ser._internal.transcription import text_metrics as helpers
from ser.domain import TranscriptWord

_PROPERTY_TEST_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    database=None,
)
_NORMALIZED_TOKEN = st.text(
    alphabet=string.ascii_lowercase + string.digits,
    min_size=1,
    max_size=8,
)


def test_normalize_words_lowercases_and_strips_punctuation() -> None:
    """Normalization should lowercase, remove punctuation, and drop empty tokens."""

    assert helpers.normalize_words("  Kids, ARE talking... by-the door!  ") == [
        "kids",
        "are",
        "talking",
        "by",
        "the",
        "door",
    ]


def test_levenshtein_distance_counts_edit_operations() -> None:
    """Distance helper should account for substitution and deletion costs."""

    assert helpers.levenshtein_distance(["kids", "are", "talking"], ["kids", "were"]) == 2


def test_compute_word_error_rate_handles_empty_reference() -> None:
    """WER helper should keep the public empty-reference contract."""

    assert helpers.compute_word_error_rate("", "") == 0.0
    assert helpers.compute_word_error_rate("", "unexpected output") == 1.0
    assert helpers.compute_word_error_rate("kids are talking", "kids are") == pytest.approx(
        1.0 / 3.0
    )


def test_transcript_words_to_text_strips_blank_tokens() -> None:
    """Transcript flattening should discard blank word entries."""

    words = [
        TranscriptWord(" kids ", 0.0, 0.2),
        TranscriptWord(" ", 0.2, 0.4),
        TranscriptWord("talking", 0.4, 0.6),
    ]

    assert helpers.transcript_words_to_text(words) == "kids talking"


def test_percentile_uses_nearest_rank_and_empty_default() -> None:
    """Percentile helper should preserve nearest-rank semantics."""

    assert helpers.percentile([], 0.9) == 1.0
    assert helpers.percentile([0.3, 0.1, 0.2, 0.4], 0.5) == 0.2
    assert helpers.percentile([0.3, 0.1, 0.2, 0.4], 0.9) == 0.4


@_PROPERTY_TEST_SETTINGS
@given(st.text())
def test_normalize_words_outputs_lowercase_alphanumeric_tokens(text: str) -> None:
    """Normalization should only emit lowercase alphanumeric comparison tokens."""

    normalized = helpers.normalize_words(text)

    assert all(token == token.lower() and token.isalnum() for token in normalized)


@_PROPERTY_TEST_SETTINGS
@given(
    st.lists(_NORMALIZED_TOKEN, max_size=8),
    st.lists(_NORMALIZED_TOKEN, max_size=8),
)
def test_levenshtein_distance_is_symmetric_and_bounded(
    reference: list[str],
    hypothesis: list[str],
) -> None:
    """Distance helper should preserve metric symmetry and length bounds."""

    distance = helpers.levenshtein_distance(reference, hypothesis)

    assert distance == helpers.levenshtein_distance(hypothesis, reference)
    assert (
        abs(len(reference) - len(hypothesis))
        <= distance
        <= max(
            len(reference),
            len(hypothesis),
        )
    )


@_PROPERTY_TEST_SETTINGS
@given(st.lists(_NORMALIZED_TOKEN, max_size=8))
def test_levenshtein_distance_is_zero_for_identical_sequences(tokens: list[str]) -> None:
    """Distance helper should report no edits for identical token sequences."""

    assert helpers.levenshtein_distance(tokens, tokens) == 0


@_PROPERTY_TEST_SETTINGS
@given(
    st.lists(_NORMALIZED_TOKEN, min_size=1, max_size=8),
    st.lists(_NORMALIZED_TOKEN, max_size=8),
)
def test_compute_word_error_rate_matches_edit_distance_ratio(
    reference: list[str],
    hypothesis: list[str],
) -> None:
    """WER helper should match normalized edit distance over non-empty references."""

    expected = helpers.levenshtein_distance(reference, hypothesis) / float(len(reference))

    assert helpers.compute_word_error_rate(
        " ".join(reference),
        " ".join(hypothesis),
    ) == pytest.approx(expected)
