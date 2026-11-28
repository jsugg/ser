"""MESD v5 test-set manifest and release accounting."""

import csv
import hashlib
import wave
from pathlib import Path

from ser._internal.data.adapters.mesd import build_mesd_manifest_jsonl
from ser._internal.data.dataset_audit import audit_dataset_recipe
from ser._internal.data.manifest_jsonl import load_manifest_jsonl
from ser._internal.data.ontology import LabelOntology
from ser._internal.data.recipe import research_recipe_v1


def test_mesd_manifest_accounts_for_source_files_without_guessing(tmp_path: Path) -> None:
    source = tmp_path / "raw" / "mesd-v5"
    source.mkdir(parents=True)
    for index, name in enumerate(
        [
            "Anger_C_A_abajo.wav",
            "Disgust_F_A_abajo.wav",
            "Fear_M_A_abajo.wav",
            "Anger_F_B_L1_abuso.wav",
            "Anger_F_B_L2_abuso.wav",
            "Neutral_X_A_abajo.wav",
        ]
    ):
        with wave.open(str(source / name), "wb") as audio:
            audio.setparams((1, 2, 48_000, 1, "NONE", "not compressed"))
            audio.writeframes(index.to_bytes(2, byteorder="little"))
    (source / "README.txt").write_text("documentation", encoding="utf-8")
    (source / "Fear_M_A_antes.wav").write_bytes((source / "Fear_M_A_abajo.wav").read_bytes())
    ontology = LabelOntology("test", frozenset({"angry", "disgust", "fearful"}))
    manifest = tmp_path / "manifests" / "mesd.jsonl"

    build_mesd_manifest_jsonl(
        dataset_root=tmp_path,
        ontology=ontology,
        output_path=manifest,
    )

    rows = load_manifest_jsonl(manifest, ontology=ontology)
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
        "Anger_F_B_L1_abuso.wav": "naturalness_reduced",
        "Anger_F_B_L2_abuso.wav": "naturalness_reduced",
        "Neutral_X_A_abajo.wav": "invalid_mesd_filename",
        "README.txt": "not_audio",
    }
    assert ledger["Fear_M_A_antes.wav"]["duplicate_of"] == "Fear_M_A_abajo.wav"
    expected_hashes = {
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in source.iterdir()
    }
    assert set((tmp_path / "mesd-SHA256SUMS").read_text().splitlines()) == expected_hashes
