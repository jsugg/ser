"""Tests for BIIC-Podcast dataset adapter."""

from __future__ import annotations

from pathlib import Path

from ser.data.adapters.biic_podcast import (
    BIIC_PODCAST_CORPUS_ID,
    BIIC_PODCAST_DATASET_LICENSE_ID,
    BIIC_PODCAST_DATASET_POLICY_ID,
    build_biic_podcast_utterances,
)
from ser.data.ontology import LabelOntology


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset({"happy", "sad"}),
    )


def test_build_biic_podcast_utterances_parses_csv_and_metadata(tmp_path: Path) -> None:
    """BIIC adapter should parse label CSV and emit enriched utterances."""
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "FileName,EmoClass,Split_Set,Speaker,Start,Duration\n"
        "a.wav,2,Train,spk1,0.5,1.0\n"
        "b.wav,1,Test,spk2,1.0,1.5\n",
        encoding="utf-8",
    )

    utterances = build_biic_podcast_utterances(
        dataset_root=tmp_path,
        labels_csv_path=labels_csv,
        audio_base_dir=tmp_path,
        ontology=_ontology(),
        default_language="en",
        max_failed_file_ratio=0.5,
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {BIIC_PODCAST_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        BIIC_PODCAST_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        BIIC_PODCAST_DATASET_LICENSE_ID
    }
