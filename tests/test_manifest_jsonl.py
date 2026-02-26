"""Tests for manifest schema and JSONL loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.data.manifest_jsonl import load_manifest_jsonl, write_manifest_jsonl
from ser.data.ontology import LabelOntology, remap_label


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="basic_v1",
        allowed_labels=frozenset({"happy", "sad", "other"}),
    )


def test_manifest_loader_rejects_non_scoped_speaker_id(tmp_path: Path) -> None:
    """speaker_id must be prefixed by corpus id."""
    path = tmp_path / "bad.jsonl"
    path.write_text(
        (
            '{"schema_version": 1, "sample_id": "id:1", "corpus": "ravdess", '
            '"audio_path": "a.wav", "label": "happy", "speaker_id": "42"}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speaker_id must be corpus-scoped"):
        load_manifest_jsonl(path, ontology=_ontology())


def test_manifest_loader_round_trip(tmp_path: Path) -> None:
    """Writer and loader should preserve record essentials."""
    path = tmp_path / "dataset.jsonl"
    utterances = [
        Utterance(
            schema_version=MANIFEST_SCHEMA_VERSION,
            sample_id="ravdess:a.wav",
            corpus="ravdess",
            audio_path=tmp_path / "audio" / "a.wav",
            label="happy",
            speaker_id="ravdess:1",
            language="en",
            split="train",
        ),
        Utterance(
            schema_version=MANIFEST_SCHEMA_VERSION,
            sample_id="ravdess:b.wav",
            corpus="ravdess",
            audio_path=tmp_path / "audio" / "b.wav",
            label="sad",
            speaker_id="ravdess:2",
            language="en",
            split="test",
            start_seconds=1.5,
            duration_seconds=2.0,
        ),
    ]

    write_manifest_jsonl(path, utterances, base_dir=tmp_path)
    loaded = load_manifest_jsonl(path, ontology=_ontology(), base_dir=tmp_path)

    assert [item.sample_id for item in loaded] == ["ravdess:a.wav", "ravdess:b.wav"]
    assert loaded[1].start_seconds == pytest.approx(1.5)
    assert loaded[1].duration_seconds == pytest.approx(2.0)


def test_remap_label_honors_unknown_policy() -> None:
    """Unknown label policy should gate adapter remapping behavior."""
    ontology_drop = LabelOntology(
        ontology_id="drop_v1",
        allowed_labels=frozenset({"happy", "sad", "other"}),
        unknown_label_policy="drop",
    )
    ontology_other = LabelOntology(
        ontology_id="other_v1",
        allowed_labels=frozenset({"happy", "sad", "other"}),
        unknown_label_policy="map_to_other",
    )
    ontology_error = LabelOntology(
        ontology_id="error_v1",
        allowed_labels=frozenset({"happy", "sad", "other"}),
        unknown_label_policy="error",
    )

    assert remap_label(raw_label="03", mapping={"03": "happy"}, ontology=ontology_drop) == "happy"
    assert remap_label(raw_label="99", mapping={"03": "happy"}, ontology=ontology_drop) is None
    assert (
        remap_label(raw_label="99", mapping={"03": "happy"}, ontology=ontology_other)
        == "other"
    )
    with pytest.raises(ValueError, match="Unknown label"):
        remap_label(raw_label="99", mapping={"03": "happy"}, ontology=ontology_error)
