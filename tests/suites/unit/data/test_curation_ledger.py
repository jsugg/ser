"""Tests for the dataset curation ledger and its audit integration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ser._internal.data.curation_ledger import (
    CurationLedger,
    CurationLedgerError,
    LedgerEntry,
    apply_curation_ledger,
    load_builtin_curation_ledger,
    load_curation_ledger,
    normalized_audio_sha256,
    stamp_curated_corpora,
)
from ser._internal.data.dataset_audit import DatasetAuditError, audit_dataset_recipe
from ser._internal.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser._internal.data.recipe import CorpusRecipe, DatasetRecipe
from ser._internal.models.training_readiness import normalized_pcm_digest

_REVISION = "fixture-r1"


def _write_wav(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    sf.write(path, rng.uniform(-0.5, 0.5, 400).astype(np.float32), 16_000)


def _recipe() -> DatasetRecipe:
    return DatasetRecipe(
        recipe_id="fixture",
        revision="1",
        ontology_version="canonical-eight-v1",
        corpora=(
            CorpusRecipe(corpus="acted", exact_primary_labels=frozenset({"happy", "sad", "angry"})),
        ),
    )


def _row(tmp_path: Path, name: str, label: str, speaker: str, digest: str | None) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=f"acted:{name}.wav",
        corpus="acted",
        audio_path=tmp_path / f"{name}.wav",
        label=label,
        speaker_id=f"acted:{speaker}",
        normalized_audio_sha256=digest,
        dataset_revision=_REVISION,
    )


def _entry(row: Utterance, reason: str, kept: str | None = None) -> LedgerEntry:
    assert row.normalized_audio_sha256 is not None
    return LedgerEntry(
        corpus=row.corpus,
        dataset_revision=_REVISION,
        sample_id=row.sample_id,
        normalized_audio_sha256=row.normalized_audio_sha256,
        reason=reason,  # type: ignore[arg-type]
        kept_sample_id=kept,
    )


def _rows_with_duplicates(tmp_path: Path) -> list[Utterance]:
    for name, seed in (("a", 1), ("b", 2), ("c", 3), ("d", 4)):
        _write_wav(tmp_path / f"{name}.wav", seed)
    (tmp_path / "e.wav").write_bytes((tmp_path / "a.wav").read_bytes())
    (tmp_path / "f.wav").write_bytes((tmp_path / "c.wav").read_bytes())
    names = {"a": "happy", "b": "sad", "c": "angry", "d": "happy", "e": "sad", "f": "angry"}
    speakers = {"a": "s1", "b": "s2", "c": "s3", "d": "s4", "e": "s5", "f": "s6"}
    rows = [
        _row(
            tmp_path,
            name,
            label,
            speakers[name],
            normalized_audio_sha256(tmp_path / f"{name}.wav"),
        )
        for name, label in names.items()
    ]
    for index in range(30):
        name = f"u{index}"
        _write_wav(tmp_path / f"{name}.wav", 100 + index)
        label = ("happy", "sad", "angry")[index % 3]
        digest = normalized_audio_sha256(tmp_path / f"{name}.wav")
        rows.append(_row(tmp_path, name, label, f"u{index}", digest))
    return rows


def _ledger(rows: list[Utterance]) -> CurationLedger:
    by_name = {row.sample_id: row for row in rows}
    return CurationLedger(
        ledger_id="fixture",
        version="1",
        entries=(
            _entry(by_name["acted:a.wav"], "conflicting_label_duplicate"),
            _entry(by_name["acted:e.wav"], "conflicting_label_duplicate"),
            _entry(by_name["acted:f.wav"], "exact_repeat_same_label", kept="acted:c.wav"),
        ),
    )


def test_strict_audit_reports_duplicates_without_ledger_and_passes_with_it(
    tmp_path: Path,
) -> None:
    rows = _rows_with_duplicates(tmp_path)

    with pytest.raises(DatasetAuditError, match="Duplicate normalized audio content"):
        audit_dataset_recipe(rows, recipe=_recipe(), strict=True)

    report = audit_dataset_recipe(rows, recipe=_recipe(), strict=True, curation=_ledger(rows))

    assert {item.sample_id: item.reason for item in report.curation_excluded} == {
        "acted:a.wav": "conflicting_label_duplicate",
        "acted:e.wav": "conflicting_label_duplicate",
        "acted:f.wav": "exact_repeat_same_label",
    }
    assert report.counters["curation_excluded"] == 3
    assert {"acted:b.wav", "acted:c.wav", "acted:d.wav"} <= {
        entry.sample_id for entry in report.ledger
    }
    assert len(report.ledger) == len(rows) - 3


def test_hash_mismatch_is_an_error(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    bad = replace(ledger.entries[0], normalized_audio_sha256="0" * 64)
    ledger = replace(ledger, entries=(bad, *ledger.entries[1:]))

    with pytest.raises(CurationLedgerError, match="hash"):
        apply_curation_ledger(rows, ledger)


def test_entry_for_missing_clip_is_an_error(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)

    with pytest.raises(CurationLedgerError, match="does not exist"):
        apply_curation_ledger([row for row in rows if row.sample_id != "acted:f.wav"], ledger)


def test_unused_entry_is_an_error(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    unrelated = _entry(rows[1], "conflicting_label_duplicate")  # b.wav has unique content

    with pytest.raises(CurationLedgerError, match="unused"):
        apply_curation_ledger(rows, replace(ledger, entries=(*ledger.entries, unrelated)))


def test_unlisted_duplicate_stays_a_finding(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    partial = replace(ledger, entries=ledger.entries[:2])

    with pytest.raises(DatasetAuditError, match="Duplicate normalized audio content"):
        audit_dataset_recipe(rows, recipe=_recipe(), strict=True, curation=partial)


def test_conflicting_reason_requires_differing_labels(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    wrong = replace(ledger.entries[2], reason="conflicting_label_duplicate", kept_sample_id=None)

    with pytest.raises(CurationLedgerError, match="label"):
        apply_curation_ledger(rows, replace(ledger, entries=(*ledger.entries[:2], wrong)))


def test_entries_for_other_revisions_are_ignored(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    other = [replace(row, dataset_revision="fixture-r2") for row in rows]

    result = apply_curation_ledger(other, ledger)

    assert result.utterances == tuple(other)
    assert result.excluded == ()


def test_ledger_file_round_trip_rejects_unknown_reason(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger.to_record()), encoding="utf-8")
    assert load_curation_ledger(path) == ledger

    record = ledger.to_record()
    record["entries"][0]["reason"] = "keep-first"  # type: ignore[index]
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(CurationLedgerError, match="reason"):
        load_curation_ledger(path)


def test_builtin_ledger_holds_ids_and_hashes_only() -> None:
    text = json.dumps(load_builtin_curation_ledger().to_record())

    assert "/home/" not in text and "\\" not in text
    assert all(entry.sample_id.split(":", 1)[0] for entry in load_builtin_curation_ledger().entries)


def test_stamp_fills_hash_and_revision_for_curated_corpora_only(tmp_path: Path) -> None:
    _write_wav(tmp_path / "x.wav", 9)
    covered = replace(
        _row(tmp_path, "x", "happy", "s1", None),
        corpus="ravdess",
        sample_id="ravdess:x.wav",
        dataset_revision=None,
    )
    other = replace(covered, corpus="acted", sample_id="acted:x.wav")

    stamped_covered, stamped_other = stamp_curated_corpora([covered, other])

    assert stamped_covered.normalized_audio_sha256 == normalized_pcm_digest(tmp_path / "x.wav")
    assert stamped_covered.dataset_revision is not None
    assert stamped_other == other
