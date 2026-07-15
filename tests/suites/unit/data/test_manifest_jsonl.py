"""Tests for manifest schema and JSONL loading utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ser._internal.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    TargetAnnotation,
    Utterance,
    VadTarget,
)
from ser._internal.data.manifest_jsonl import load_manifest_jsonl, write_manifest_jsonl
from ser._internal.data.ontology import LabelOntology, remap_label


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


def test_manifest_loader_preserves_v1_compatibility(tmp_path: Path) -> None:
    """Explicit v1 records remain loadable after schema v2 is introduced."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        '{"schema_version":1,"sample_id":"legacy:1","corpus":"legacy",'
        '"audio_path":"a.wav","label":"happy"}\n',
        encoding="utf-8",
    )

    loaded = load_manifest_jsonl(path, ontology=_ontology())

    assert loaded[0].schema_version == MANIFEST_SCHEMA_VERSION
    assert loaded[0].label == "happy"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            '{"schema_version":999,"sample_id":"x:1","corpus":"x",'
            '"audio_path":"a.wav","label":"happy"}\n',
            "Unsupported manifest schema version",
        ),
        (
            '{"schema_version":1,"sample_id":"x:1","corpus":"x",' '"audio_path":"a.wav"}\n',
            "schema v1 requires a categorical label",
        ),
    ],
)
def test_manifest_loader_validates_source_schema_before_v2_upgrade(
    tmp_path: Path,
    payload: str,
    error: str,
) -> None:
    """Compatibility upgrades never reinterpret unsupported or malformed source records."""
    path = tmp_path / "source.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_manifest_jsonl(path, ontology=_ontology())


def test_manifest_v2_round_trip_preserves_auxiliary_targets(tmp_path: Path) -> None:
    """V2 records may omit emotion while preserving typed auxiliary annotations."""
    path = tmp_path / "auxiliary.jsonl"
    utterance = Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id="escorpus:1",
        corpus="escorpus",
        audio_path=tmp_path / "a.wav",
        label=None,
        vad=VadTarget(valence=-0.2, arousal=0.4, dominance=0.1),
        transcript="hola",
        annotations=(TargetAnnotation("vad", "annotator_consensus", 0.85),),
        session_id="escorpus:session-1",
        native_split="dev",
        normalized_audio_sha256="a" * 64,
        dataset_revision="2026-07",
    )

    write_manifest_jsonl(path, [utterance], base_dir=tmp_path)
    loaded = load_manifest_jsonl(path, ontology=_ontology(), base_dir=tmp_path)

    assert loaded == [utterance]


@pytest.mark.parametrize(
    "record,error",
    [
        (
            {
                "schema_version": 2,
                "sample_id": "x:1",
                "corpus": "x",
                "audio_path": "a.wav",
                "vad": {"valence": 2.0, "arousal": 0.0, "dominance": 0.0},
            },
            r"within \[-1, 1\]",
        ),
        (
            {
                "schema_version": 2,
                "sample_id": "x:1",
                "corpus": "x",
                "audio_path": "a.wav",
                "language": "en",
                "normalized_audio_sha256": "ABC",
            },
            "64 lowercase hexadecimal",
        ),
    ],
)
def test_manifest_v2_rejects_invalid_target_boundaries(
    tmp_path: Path,
    record: dict[str, object],
    error: str,
) -> None:
    """External target and digest fields fail closed at manifest load time."""
    import json

    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        load_manifest_jsonl(path, ontology=_ontology())


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
    assert remap_label(raw_label="99", mapping={"03": "happy"}, ontology=ontology_other) == "other"
    with pytest.raises(ValueError, match="Unknown label"):
        remap_label(raw_label="99", mapping={"03": "happy"}, ontology=ontology_error)
