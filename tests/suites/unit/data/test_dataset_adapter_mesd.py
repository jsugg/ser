"""MESD v5 test-set manifest and release accounting."""

import csv
import hashlib
import json
import wave
from pathlib import Path

import pytest

from ser._internal.data.adapters import mesd
from ser._internal.data.adapters.mesd import build_mesd_manifest_jsonl
from ser._internal.data.dataset_audit import audit_dataset_recipe
from ser._internal.data.manifest_jsonl import load_manifest_jsonl
from ser._internal.data.ontology import LabelOntology
from ser._internal.data.recipe import research_recipe_v1


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, source: Path) -> None:
    checksums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(source).as_posix()}\n"
        for path in sorted(source.rglob("*"))
        if path.is_file()
    )
    monkeypatch.setattr(
        mesd, "MESD_SOURCE_INVENTORY_SHA256", hashlib.sha256(checksums.encode()).hexdigest()
    )


def test_mesd_manifest_accounts_for_source_files_without_guessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "raw" / "mesd-v5"
    source.mkdir(parents=True)
    for index, name in enumerate(
        [
            "Anger_C_A_abajo.wav",
            "Disgust_F_A_abajo.wav",
            "Fear_M_A_abajo.wav",
            "Level1/Anger_F_B_L1_abuso.wav",
            "Level2/Anger_F_B_L2_abuso.wav",
            "Neutral_X_A_abajo.wav",
        ]
    ):
        _write_wav(source / name, sample=index)
    (source / "README.txt").write_text("documentation", encoding="utf-8")
    (source / "Fear_M_A_antes.wav").write_bytes((source / "Fear_M_A_abajo.wav").read_bytes())
    _pin_fixture(monkeypatch, source)
    ontology = LabelOntology("test", frozenset({"angry", "disgust", "fearful"}))
    manifest = tmp_path / "manifests" / "mesd.jsonl"

    build_mesd_manifest_jsonl(
        dataset_root=tmp_path,
        ontology=ontology,
        output_path=manifest,
    )

    rows = load_manifest_jsonl(manifest, ontology=ontology, base_dir=tmp_path)
    assert {(row.label, row.raw_label) for row in rows} == {
        ("angry", "Anger"),
        ("disgust", "Disgust"),
        ("fearful", "Fear"),
    }
    assert all(
        row.language == "es"
        and row.speaker_id is None
        and row.session_id is None
        and row.split == "test"
        and row.native_split is None
        and row.dataset_revision == "10.17632/cy34mh68j9.5"
        and row.dataset_license_id == "cc-by-4.0"
        and row.audio_path.is_file()
        and row.normalized_audio_sha256 is not None
        for row in rows
    )
    assert {row.raw_label: row.to_record()["voice_type"] for row in rows} == {
        "Anger": "C",
        "Disgust": "F",
        "Fear": "M",
    }
    audit = audit_dataset_recipe(rows, recipe=research_recipe_v1(), strict=False)
    assert [(row.split, row.group_id) for row in audit.ledger] == [("test", None)] * 3

    with (tmp_path / "mesd-dispositions.csv").open(newline="", encoding="utf-8") as handle:
        ledger = {row["source_path"]: row for row in csv.DictReader(handle)}
    dispositions = {name: row["disposition"] for name, row in ledger.items()}
    assert dispositions == {
        "Anger_C_A_abajo.wav": "used",
        "Disgust_F_A_abajo.wav": "used",
        "Fear_M_A_abajo.wav": "used",
        "Fear_M_A_antes.wav": "duplicate_source_file",
        "Level1/Anger_F_B_L1_abuso.wav": "naturalness_reduced",
        "Level2/Anger_F_B_L2_abuso.wav": "naturalness_reduced",
        "Neutral_X_A_abajo.wav": "invalid_mesd_filename",
        "README.txt": "not_audio",
    }
    assert ledger["Fear_M_A_antes.wav"]["duplicate_of"] == "Fear_M_A_abajo.wav"
    expected_hashes = {
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(source).as_posix()}"
        for path in source.rglob("*")
        if path.is_file()
    }
    assert set((tmp_path / "mesd-SHA256SUMS").read_text().splitlines()) == expected_hashes


def _write_wav(path: Path, sample: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 48_000, 1, "NONE", "not compressed"))
        audio.writeframes(sample.to_bytes(2, byteorder="little"))


def test_mesd_manifest_resolves_after_moving_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "old"
    source = root / "raw/mesd-v5"
    clip = source / "Anger_C_A_abajo.wav"
    _write_wav(clip)
    _pin_fixture(monkeypatch, source)
    manifest = tmp_path / "manifests/mesd.jsonl"
    ontology = LabelOntology("test", frozenset({"angry"}))
    build_mesd_manifest_jsonl(dataset_root=root, ontology=ontology, output_path=manifest)

    relative = clip.relative_to(root)
    assert [Path(json.loads(line)["audio_path"]) for line in manifest.read_text().splitlines()] == [
        relative
    ]
    moved = root.rename(tmp_path / "new")
    rows = load_manifest_jsonl(manifest, ontology=ontology, base_dir=moved)
    assert [row.audio_path for row in rows] == [moved / relative]
    assert rows[0].audio_path.is_file()
    artifacts = [manifest, moved / "mesd-dispositions.csv", moved / "mesd-SHA256SUMS"]
    before = [path.read_bytes() for path in artifacts]
    build_mesd_manifest_jsonl(dataset_root=moved, ontology=ontology, output_path=manifest)
    assert [path.read_bytes() for path in artifacts] == before


@pytest.mark.parametrize("conflict", ["Sadness_F_A_abajo.wav", "Anger_M_A_abajo.wav"])
def test_mesd_excludes_all_copies_of_conflicting_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conflict: str
) -> None:
    source = tmp_path / "raw/mesd-v5"
    for name in ("Anger_F_A_abajo.wav", "Anger_F_A_antes.wav", conflict):
        _write_wav(source / name)
    _write_wav(source / "Fear_M_A_abajo.wav", sample=1)
    _pin_fixture(monkeypatch, source)

    rows = build_mesd_manifest_jsonl(
        dataset_root=tmp_path,
        ontology=LabelOntology("test", frozenset({"angry", "sad", "fearful"})),
        output_path=tmp_path / "mesd.jsonl",
    )

    assert [row.label for row in rows] == ["fearful"]
    with (tmp_path / "mesd-dispositions.csv").open(newline="", encoding="utf-8") as handle:
        assert {row["source_path"]: row["disposition"] for row in csv.DictReader(handle)} == {
            "Anger_F_A_abajo.wav": "conflicting_duplicate",
            "Anger_F_A_antes.wav": "conflicting_duplicate",
            conflict: "conflicting_duplicate",
            "Fear_M_A_abajo.wav": "used",
        }


def test_mesd_rejects_mismatches_between_reduction_folder_and_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "raw/mesd-v5"
    _write_wav(source / "Fear_M_A_abajo.wav")
    invalid = (
        "Level1/Anger_F_B_abuso.wav",
        "Level2/Anger_F_B_L1_abuso.wav",
        "Anger_F_B_L2_abuso.wav",
        "Level1/nested/Anger_F_B_abuso.wav",
    )
    for index, name in enumerate(invalid, start=1):
        _write_wav(source / name, sample=index)
    _pin_fixture(monkeypatch, source)

    build_mesd_manifest_jsonl(
        dataset_root=tmp_path,
        ontology=LabelOntology("test", frozenset({"angry", "fearful"})),
        output_path=tmp_path / "mesd.jsonl",
    )
    with (tmp_path / "mesd-dispositions.csv").open(newline="", encoding="utf-8") as handle:
        ledger = {row["source_path"]: row["disposition"] for row in csv.DictReader(handle)}
    assert all(ledger[name] == "invalid_mesd_filename" for name in invalid)


def test_mesd_rejects_changed_release_before_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "raw/mesd-v5"
    clip = source / "Anger_C_A_abajo.wav"
    _write_wav(clip)
    _pin_fixture(monkeypatch, source)
    _write_wav(clip, sample=1)

    manifest = tmp_path / "mesd.jsonl"
    with pytest.raises(ValueError, match="MESD v5 source inventory SHA-256 mismatch"):
        build_mesd_manifest_jsonl(
            dataset_root=tmp_path,
            ontology=LabelOntology("test", frozenset({"angry"})),
            output_path=manifest,
        )
    assert not manifest.exists()
    assert not (tmp_path / "mesd-dispositions.csv").exists()
    assert not (tmp_path / "mesd-SHA256SUMS").exists()
