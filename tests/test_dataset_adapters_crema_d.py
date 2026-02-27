"""Tests for CREMA-D dataset adapter."""

from __future__ import annotations

from pathlib import Path

from ser.data.adapters.crema_d import (
    CREMA_D_CORPUS_ID,
    CREMA_D_DATASET_LICENSE_ID,
    CREMA_D_DATASET_POLICY_ID,
    build_crema_d_utterances,
)
from ser.data.ontology import LabelOntology


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset({"happy", "sad", "angry"}),
    )


def test_build_crema_d_utterances_maps_codes_and_metadata(tmp_path: Path) -> None:
    """CREMA-D adapter should parse filename code and attach metadata."""
    audio_root = tmp_path / "AudioWAV"
    audio_root.mkdir(parents=True, exist_ok=True)
    (audio_root / "1001_IEO_HAP_LO.wav").write_bytes(b"fake")
    (audio_root / "1002_IEO_SAD_LO.wav").write_bytes(b"fake")

    utterances = build_crema_d_utterances(
        dataset_root=tmp_path,
        dataset_glob_pattern="AudioWAV/**/*.wav",
        emotion_code_map={"HAP": "happy", "SAD": "sad"},
        default_language="en",
        ontology=_ontology(),
        max_failed_file_ratio=0.5,
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {CREMA_D_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {
        CREMA_D_DATASET_POLICY_ID
    }
    assert {item.dataset_license_id for item in utterances} == {
        CREMA_D_DATASET_LICENSE_ID
    }
