"""Tests for declarative public dataset catalog entries."""

from __future__ import annotations

from ser.data.catalog.public_datasets import (
    EMODB_2_CORPUS_ID,
    EMODB_2_DATASET_LICENSE_ID,
    EMODB_2_DATASET_POLICY_ID,
    PUBLIC_CSV_MANIFEST_SPECS,
    SPANISH_MEACORPUS_2023_CORPUS_ID,
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
