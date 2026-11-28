"""Tests for the dataset curation ledger and its audit integration."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import ser._internal.data.curation_ledger as curation_ledger
from ser._internal.data.curation_ledger import (
    CurationLedger,
    CurationLedgerError,
    LedgerEntry,
    apply_curation_ledger,
    git_blob_sha1,
    git_lfs_pointer_sha1,
    git_tree_sha1,
    load_builtin_curation_ledger,
    load_curation_ledger,
    normalized_audio_sha256,
)
from ser._internal.data.dataset_audit import DatasetAuditError, audit_dataset_recipe
from ser._internal.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser._internal.data.recipe import CorpusRecipe, DatasetRecipe

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
    (tmp_path / "f.wav").unlink()

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
    keeper = _entry(rows[2], "conflicting_label_duplicate")

    with pytest.raises(CurationLedgerError, match="label"):
        apply_curation_ledger(rows, replace(ledger, entries=(*ledger.entries[:2], wrong, keeper)))


def test_entries_for_a_corpus_that_is_not_loaded_are_skipped(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    other = [
        replace(row, corpus="spoken", sample_id=f"spoken:{row.audio_path.name}", speaker_id=None)
        for row in rows
    ]

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


def test_parser_rejects_non_hex_hashes(tmp_path: Path) -> None:
    record = _ledger(_rows_with_duplicates(tmp_path)).to_record()
    record["entries"][0]["normalized_audio_sha256"] = "g" * 64  # type: ignore[index]
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(CurationLedgerError, match="hexadecimal"):
        load_curation_ledger(path)


def test_repeated_exclusion_is_an_error(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)

    with pytest.raises(CurationLedgerError, match="more than once"):
        apply_curation_ledger(rows, replace(ledger, entries=(*ledger.entries, ledger.entries[2])))


def test_named_keeper_that_is_also_excluded_is_an_error(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    (tmp_path / "g.wav").write_bytes((tmp_path / "c.wav").read_bytes())
    by_id = {row.sample_id: row for row in rows}
    rows.append(
        replace(by_id["acted:f.wav"], sample_id="acted:g.wav", audio_path=tmp_path / "g.wav")
    )
    ledger = _ledger(rows)
    keeper_dropped = _entry(by_id["acted:c.wav"], "exact_repeat_same_label", kept="acted:g.wav")

    with pytest.raises(CurationLedgerError, match="kept"):
        apply_curation_ledger(rows, replace(ledger, entries=(*ledger.entries, keeper_dropped)))


def test_changed_media_of_a_listed_clip_fails_despite_recorded_hash(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    _write_wav(tmp_path / "a.wav", 999)

    with pytest.raises(CurationLedgerError, match="changed"):
        audit_dataset_recipe(rows, recipe=_recipe(), strict=True, curation=ledger)


def test_changed_media_of_a_duplicate_peer_fails(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    _write_wav(tmp_path / "c.wav", 999)

    with pytest.raises(CurationLedgerError, match="changed"):
        apply_curation_ledger(rows, ledger)


@pytest.mark.parametrize("ledger_applied", [True, False])
def test_strict_audit_fails_on_changed_media_of_a_row_outside_the_ledger(
    tmp_path: Path, ledger_applied: bool
) -> None:
    rows = _rows_with_duplicates(tmp_path)
    shutil.copyfile(tmp_path / "c.wav", tmp_path / "u0.wav")
    curation = _ledger(rows) if ledger_applied else None
    audited = rows if ledger_applied else rows[6:]

    with pytest.raises(DatasetAuditError, match="'acted:u0.wav' changed"):
        audit_dataset_recipe(audited, recipe=_recipe(), strict=True, curation=curation)


def test_strict_audit_hashes_each_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _rows_with_duplicates(tmp_path)
    curated = [row for row in rows if row.sample_id not in {"acted:a.wav", "acted:e.wav"}]
    calls: list[Path] = []

    def counting(path: Path) -> str:
        calls.append(path)
        return normalized_audio_sha256(path)

    monkeypatch.setattr(curation_ledger, "normalized_audio_sha256", counting)
    audit_dataset_recipe(curated, recipe=_recipe(), strict=True, curation=_ledger(rows))

    assert sorted(calls) == sorted(row.audio_path for row in rows)


def test_strict_curation_rejects_rows_without_revision(tmp_path: Path) -> None:
    rows = [replace(row, dataset_revision=None) for row in _rows_with_duplicates(tmp_path)]

    with pytest.raises(CurationLedgerError, match="rebuild the manifest"):
        audit_dataset_recipe(rows, recipe=_recipe(), strict=True, curation=_ledger(rows))


def test_strict_audit_names_the_rebuild_for_rows_without_revision(tmp_path: Path) -> None:
    rows = [replace(row, dataset_revision=None) for row in _rows_with_duplicates(tmp_path)]

    with pytest.raises(DatasetAuditError, match="rebuild the manifest"):
        audit_dataset_recipe(rows[6:], recipe=_recipe(), strict=True)


def test_partial_conflicting_group_fails(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    without_a = replace(ledger, entries=ledger.entries[1:])

    with pytest.raises(CurationLedgerError, match="conflicting"):
        audit_dataset_recipe(rows, recipe=_recipe(), strict=True, curation=without_a)


def test_entry_for_another_revision_of_a_loaded_corpus_fails(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    moved = replace(ledger.entries[0], dataset_revision="fixture-r2")

    with pytest.raises(CurationLedgerError, match="revision"):
        apply_curation_ledger(rows, replace(ledger, entries=(moved, *ledger.entries[1:])))


def test_entry_for_another_loaded_corpus_does_not_match_the_row(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    for name, seed in (("o1", 501), ("o2", 502)):
        _write_wav(tmp_path / f"{name}.wav", seed)
        digest = normalized_audio_sha256(tmp_path / f"{name}.wav")
        rows.append(
            replace(
                _row(tmp_path, name, "happy", name, digest),
                corpus="other",
                sample_id=f"other:{name}.wav",
                speaker_id=f"other:{name}",
            )
        )
    ledger = _ledger(rows)
    moved = replace(ledger.entries[0], corpus="other")

    with pytest.raises(CurationLedgerError, match="does not exist"):
        apply_curation_ledger(rows, replace(ledger, entries=(moved, *ledger.entries[1:])))


def test_curated_rows_pass_with_excluded_clips_absent(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    ledger = _ledger(rows)
    excluded = {"acted:a.wav", "acted:e.wav", "acted:f.wav"}
    curated = [row for row in rows if row.sample_id not in excluded]

    report = audit_dataset_recipe(curated, recipe=_recipe(), strict=True, curation=ledger)

    assert [item.sample_id for item in report.curation_excluded] == sorted(excluded)
    _write_wav(tmp_path / "c.wav", 999)
    with pytest.raises(CurationLedgerError, match="changed"):
        audit_dataset_recipe(curated, recipe=_recipe(), strict=True, curation=ledger)


def test_missing_keeper_row_fails_even_when_its_file_exists(tmp_path: Path) -> None:
    rows = _rows_with_duplicates(tmp_path)
    dropped = {"acted:a.wav", "acted:e.wav", "acted:f.wav", "acted:c.wav"}
    curated = [row for row in rows if row.sample_id not in dropped]
    assert (tmp_path / "c.wav").is_file()

    with pytest.raises(CurationLedgerError, match="kept clip 'acted:c.wav' is not a row"):
        audit_dataset_recipe(curated, recipe=_recipe(), strict=True, curation=_ledger(rows))


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_tree_hash_matches_git_for_plain_and_lfs_blobs(tmp_path: Path) -> None:
    audio = tmp_path / "audio"
    (audio / "Actor_01").mkdir(parents=True)
    (audio / "Actor_10").mkdir()
    _write_wav(audio / "Actor_01" / "x.wav", 1)
    _write_wav(audio / "Actor_10" / "y.wav", 2)
    _write_wav(audio / "z.wav", 3)
    files = sorted(path for path in audio.rglob("*.wav"))
    pointers = tmp_path / "pointers"
    for path in files:
        target = pointers / path.relative_to(audio)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{digest}\nsize {path.stat().st_size}\n",
            encoding="ascii",
        )

    def git_tree(root: Path) -> str:
        env = {"GIT_DIR": str(tmp_path / f"{root.name}.git"), "GIT_WORK_TREE": str(root)}
        subprocess.run(["git", "init", "-q"], check=True, env=env)
        subprocess.run(["git", "add", "-A"], check=True, env=env)
        result = subprocess.run(
            ["git", "write-tree"], check=True, env=env, capture_output=True, text=True
        )
        return result.stdout.strip()

    relative = {path.relative_to(audio).as_posix(): path for path in files}
    assert git_tree_sha1({rel: git_blob_sha1(path) for rel, path in relative.items()}) == git_tree(
        audio
    )
    assert git_tree_sha1(
        {rel: git_lfs_pointer_sha1(path) for rel, path in relative.items()}
    ) == git_tree(pointers)
