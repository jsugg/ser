"""Tests for declarative public dataset catalog entries."""

from __future__ import annotations

from ser._internal.data.catalog.public_datasets import (
    EMODB_2_CORPUS_ID,
    EMODB_2_DATASET_LICENSE_ID,
    EMODB_2_DATASET_POLICY_ID,
    PUBLIC_CSV_MANIFEST_SPECS,
    SPANISH_MEACORPUS_2023_CORPUS_ID,
)
from ser._internal.data.dataset_capabilities import resolve_dataset_capability_profile


def test_escorpus_catalog_distinguishes_source_vad_and_research_terms() -> None:
    profile = resolve_dataset_capability_profile("escorpus-pe")
    assert profile.label_schema == "vad_only"
    assert not profile.supervised_ser_candidate
    assert any("research purposes" in note and "CC BY 4.0" in note for note in profile.notes)


def test_meacorpus_catalog_does_not_imply_v2_can_be_rehydrated() -> None:
    profile = resolve_dataset_capability_profile("spanish-meacorpus-2023")
    assert any(
        "18606423" in note
        and "18246241" in note
        and "neither CSV" in note
        and "speaker IDs" in note
        for note in profile.notes
    )


def test_public_csv_manifest_specs_are_indexed_by_corpus_id() -> None:
    """Catalog keys should match the embedded spec corpus ids."""

    for corpus_id, spec in PUBLIC_CSV_MANIFEST_SPECS.items():
        assert spec.corpus_id == corpus_id


def test_public_csv_manifest_specs_preserve_expected_metadata() -> None:
    """Catalog entries should keep dataset metadata and label mappings stable."""

    emodb = PUBLIC_CSV_MANIFEST_SPECS[EMODB_2_CORPUS_ID]
    assert emodb.dataset_policy_id == EMODB_2_DATASET_POLICY_ID
    assert emodb.dataset_license_id == EMODB_2_DATASET_LICENSE_ID
    assert emodb.label_mapping["boredom"] == "neutral"

    spanish = PUBLIC_CSV_MANIFEST_SPECS[SPANISH_MEACORPUS_2023_CORPUS_ID]
    assert spanish.label_mapping["joy"] == "happy"
    assert spanish.label_mapping["fear"] == "fearful"
