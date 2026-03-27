"""Tests for internal RAVDESS reference helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser._internal.transcription import ravdess_references as helpers

_STATEMENT_TEXT = {
    "01": "kids are talking by the door",
    "02": "dogs are sitting by the door",
}


def test_reference_text_uses_statement_code_lookup() -> None:
    """Reference text should resolve only known statement codes."""

    assert (
        helpers.reference_text(
            Path("03-01-02-01-01-01-01.wav"),
            statement_text=_STATEMENT_TEXT,
        )
        == "kids are talking by the door"
    )
    assert (
        helpers.reference_text(
            Path("03-01-02-01-02-01-01.wav"),
            statement_text=_STATEMENT_TEXT,
        )
        == "dogs are sitting by the door"
    )
    assert helpers.reference_text(Path("invalid.wav"), statement_text=_STATEMENT_TEXT) is None


def test_parse_metadata_extracts_expected_fields() -> None:
    """Metadata parser should extract actor, emotion, and statement values."""

    metadata = helpers.parse_metadata(Path("03-01-06-01-02-01-24.wav"))

    assert metadata == helpers.RavdessMetadata(
        emotion_code="06",
        statement_code="02",
        actor_id="24",
    )
    assert helpers.parse_metadata(Path("invalid.wav")) is None


def test_summarize_subset_coverage_counts_unique_groups() -> None:
    """Coverage summary should count unique actors, emotions, and statements."""

    files = [
        Path("03-01-06-01-02-01-24.wav"),
        Path("03-01-05-01-01-01-24.wav"),
        Path("03-01-05-01-02-01-05.wav"),
        Path("invalid.wav"),
    ]

    assert helpers.summarize_subset_coverage(files) == {
        "actors": 2,
        "emotions": 2,
        "statements": 2,
    }


def test_collect_reference_files_filters_and_stratifies_deterministically() -> None:
    """Collection helper should preserve deterministic stratified sampling."""

    mock_files = [
        f"ser/dataset/ravdess/Actor_{actor:02d}/03-01-02-01-{statement}-01-{actor:02d}.wav"
        for actor in (1, 2)
        for statement in ("01", "02")
        for _ in range(3)
    ] + ["ser/dataset/ravdess/Actor_01/invalid.wav"]

    first = helpers.collect_reference_files(
        glob_pattern="unused",
        statement_text=_STATEMENT_TEXT,
        limit=4,
        sampling_strategy="stratified",
        random_seed=11,
        glob_paths=lambda _pattern, _recursive: mock_files,
    )
    second = helpers.collect_reference_files(
        glob_pattern="unused",
        statement_text=_STATEMENT_TEXT,
        limit=4,
        sampling_strategy="stratified",
        random_seed=11,
        glob_paths=lambda _pattern, _recursive: mock_files,
    )

    assert first == second
    assert len(first) == 4


def test_collect_reference_files_validates_inputs() -> None:
    """Collection helper should fail fast on invalid limit or sampling strategy."""

    with pytest.raises(ValueError, match="limit must be positive"):
        helpers.collect_reference_files(
            glob_pattern="unused",
            statement_text=_STATEMENT_TEXT,
            limit=0,
            sampling_strategy="head",
            random_seed=42,
            glob_paths=lambda _pattern, _recursive: [],
        )

    with pytest.raises(ValueError, match="sampling_strategy"):
        helpers.collect_reference_files(
            glob_pattern="unused",
            statement_text=_STATEMENT_TEXT,
            limit=1,
            sampling_strategy="unknown",
            random_seed=42,
            glob_paths=lambda _pattern, _recursive: [],
        )
