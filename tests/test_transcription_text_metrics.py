"""Tests for internal transcription text-metric helpers."""

from __future__ import annotations

import pytest

from ser._internal.transcription import text_metrics as helpers
from ser.domain import TranscriptWord


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
