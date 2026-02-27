"""Tests for RAVDESS dataset adapter manifest records."""

from __future__ import annotations

from pathlib import Path

from ser.data.adapters.ravdess import (
    RAVDESS_CORPUS_ID,
    RAVDESS_DATASET_LICENSE_ID,
    RAVDESS_DATASET_POLICY_ID,
    build_ravdess_utterances,
)
from ser.data.ontology import LabelOntology


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset({"happy", "sad"}),
    )


def test_build_ravdess_utterances_sets_policy_and_license(tmp_path: Path) -> None:
    """Adapter should emit dataset policy/license metadata on each utterance."""
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir(parents=True, exist_ok=True)
    (actor_dir / "03-01-03-01-01-01-01.wav").write_bytes(b"fake")
    (actor_dir / "03-01-04-01-01-01-01.wav").write_bytes(b"fake")

    utterances = build_ravdess_utterances(
        dataset_root=tmp_path,
        dataset_glob_pattern=str(tmp_path / "Actor_*" / "*.wav"),
        emotion_code_map={"03": "happy", "04": "sad"},
        default_language="en",
        ontology=_ontology(),
        max_failed_file_ratio=0.5,
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {RAVDESS_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        RAVDESS_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        RAVDESS_DATASET_LICENSE_ID
    }
