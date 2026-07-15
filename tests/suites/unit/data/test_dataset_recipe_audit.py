"""Tests for explicit task routing and leakage-safe dataset audits."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ser._internal.data.dataset_audit import DatasetAuditError, audit_dataset_recipe
from ser._internal.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance, VadTarget
from ser._internal.data.recipe import CorpusRecipe, DatasetRecipe, route_utterance


def _recipe() -> DatasetRecipe:
    return DatasetRecipe(
        recipe_id="fixture",
        revision="1",
        ontology_version="canonical-eight-v1",
        corpora=(
            CorpusRecipe(
                corpus="acted",
                exact_primary_labels=frozenset({"happy", "sad", "neutral"}),
                approximate_labels=frozenset({"boredom"}),
            ),
            CorpusRecipe(corpus="vad", auxiliary_tasks=("vad",)),
        ),
    )


def _row(
    sample_id: str,
    *,
    label: str | None,
    speaker: str | None,
    digest: str,
    raw_label: str | None = None,
    corpus: str = "acted",
) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample_id,
        corpus=corpus,
        audio_path=Path(f"{sample_id}.wav"),
        label=label,
        raw_label=raw_label,
        vad=VadTarget(0.0, 0.2, -0.1) if corpus == "vad" else None,
        speaker_id=speaker,
        normalized_audio_sha256=digest,
        dataset_revision="fixture-r1",
    )


def test_exact_and_approximate_labels_route_to_isolated_heads() -> None:
    """Approximate labels never train the public primary emotion head."""
    exact = route_utterance(
        _row("acted:exact", label="happy", speaker="acted:s1", digest="1" * 64),
        _recipe(),
    )
    weak = route_utterance(
        _row(
            "acted:weak",
            label="neutral",
            raw_label="boredom",
            speaker="acted:s2",
            digest="2" * 64,
        ),
        _recipe(),
    )

    assert exact.disposition == "accepted"
    assert "primary_emotion" in exact.tasks
    assert weak.disposition == "weak"
    assert "raw_emotion" in weak.tasks
    assert "primary_emotion" not in weak.tasks


def test_audit_is_deterministic_grouped_and_exhaustively_accounted() -> None:
    """Per-corpus grouping is deterministic and every input receives one disposition."""
    rows = [
        _row(
            f"acted:{index}",
            label="happy" if index % 2 else "sad",
            speaker=f"acted:s{index}",
            digest=f"{index:064x}",
        )
        for index in range(1, 7)
    ]
    rows.append(_row("vad:1", label=None, speaker=None, digest="f" * 64, corpus="vad"))

    first = audit_dataset_recipe(rows, recipe=_recipe(), seed=9)
    second = audit_dataset_recipe(list(reversed(rows)), recipe=_recipe(), seed=9)

    assert first.split_ledger_digest == second.split_ledger_digest
    assert sum(first.counters.values()) == len(rows)
    vad_entry = next(entry for entry in first.ledger if entry.sample_id == "vad:1")
    assert vad_entry.split == "ssl_only"
    assert {entry.split for entry in first.ledger if entry.corpus == "acted"} == {
        "train",
        "dev",
        "test",
    }


def test_audit_rejects_duplicate_content_before_splitting() -> None:
    """Identical normalized PCM cannot cross sample or corpus boundaries."""
    left = _row("acted:1", label="happy", speaker="acted:s1", digest="a" * 64)
    right = replace(left, sample_id="acted:2", speaker_id="acted:s2", label="sad")

    with pytest.raises(DatasetAuditError, match="Duplicate normalized audio"):
        audit_dataset_recipe([left, right], recipe=_recipe())


def test_non_strict_audit_quarantines_every_duplicate_deterministically() -> None:
    """Exploratory audits cannot choose a duplicate survivor based on input order."""
    left = _row("acted:1", label="happy", speaker="acted:s1", digest="a" * 64)
    right = replace(left, sample_id="acted:2", speaker_id="acted:s2", label="sad")

    first = audit_dataset_recipe([left, right], recipe=_recipe(), strict=False)
    second = audit_dataset_recipe([right, left], recipe=_recipe(), strict=False)

    assert first.split_ledger_digest == second.split_ledger_digest
    assert {entry.split for entry in first.ledger} == {"quarantined"}


def test_manifest_digest_is_independent_of_install_root() -> None:
    """Moving identical content does not alter the reproducibility digest."""
    rows = [
        _row(
            f"acted:{index}",
            label="happy" if index % 2 else "sad",
            speaker=f"acted:s{index}",
            digest=f"{index:064x}",
        )
        for index in range(1, 7)
    ]
    moved = [replace(row, audio_path=Path("/other/root") / row.audio_path.name) for row in rows]

    first = audit_dataset_recipe(rows, recipe=_recipe(), seed=9)
    second = audit_dataset_recipe(moved, recipe=_recipe(), seed=9)

    assert first.manifest_digest == second.manifest_digest


def test_audit_rejects_official_identity_leakage() -> None:
    """Official split labels do not override speaker/session leakage checks."""
    left = replace(
        _row("acted:1", label="happy", speaker="acted:s1", digest="a" * 64),
        native_split="train",
    )
    right = replace(
        _row("acted:2", label="sad", speaker="acted:s1", digest="b" * 64),
        native_split="test",
    )

    with pytest.raises(DatasetAuditError, match="Split leakage"):
        audit_dataset_recipe([left, right], recipe=_recipe())
