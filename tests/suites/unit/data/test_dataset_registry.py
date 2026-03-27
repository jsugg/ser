"""Tests for dataset registry persistence and lookup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data.dataset_registry import (
    DatasetRegistryEntry,
    load_dataset_registry,
    parse_dataset_registry_options,
    registered_manifest_paths,
    remove_dataset_registry_entry,
    save_dataset_registry,
    upsert_dataset_registry_entry,
)


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(models=SimpleNamespace(folder=tmp_path / "data" / "models")),
    )


def test_registry_round_trip(tmp_path: Path) -> None:
    """Registry save/load should preserve normalized entries."""
    settings = _settings(tmp_path)
    registry = {
        "ravdess": DatasetRegistryEntry(
            dataset_id="ravdess",
            dataset_root=tmp_path / "datasets" / "ravdess",
            manifest_path=tmp_path / "manifests" / "ravdess.jsonl",
            options={"default_language": "en"},
        )
    }

    save_dataset_registry(settings=settings, registry=registry)
    loaded = load_dataset_registry(settings=settings)

    assert "ravdess" in loaded
    assert loaded["ravdess"].dataset_id == "ravdess"
    assert loaded["ravdess"].dataset_root == tmp_path / "datasets" / "ravdess"
    assert loaded["ravdess"].manifest_path == tmp_path / "manifests" / "ravdess.jsonl"
    assert loaded["ravdess"].options == {"default_language": "en"}


def test_registry_upsert_and_registered_manifest_paths(tmp_path: Path) -> None:
    """Upsert should persist entry and path helper should return existing manifests only."""
    settings = _settings(tmp_path)
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="RAVDESS",
        dataset_root=tmp_path / "datasets" / "ravdess",
        manifest_path=manifest_path,
        options={"labels_csv_path": "labels.csv"},
    )
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="crema-d",
        dataset_root=tmp_path / "datasets" / "crema-d",
        manifest_path=tmp_path / "manifests" / "missing.jsonl",
        options={},
    )

    loaded = load_dataset_registry(settings=settings)

    assert set(loaded) == {"ravdess", "crema-d"}
    assert loaded["ravdess"].dataset_id == "ravdess"
    assert loaded["ravdess"].options == {"labels_csv_path": "labels.csv"}
    assert registered_manifest_paths(settings=settings) == (manifest_path,)


def test_parse_dataset_registry_options_normalizes_known_fields() -> None:
    """Typed options parser should normalize known fields and preserve extras."""
    parsed = parse_dataset_registry_options(
        {
            "labels_csv_path": " labels.csv ",
            "audio_base_dir": " audio ",
            "source_repo_id": "org/repo",
            "source_revision": "main",
            "source_commit_sha": "ABCDEF1234",
            "default_language": " en ",
            "custom": "value",
        }
    )

    assert parsed.labels_csv_path == "labels.csv"
    assert parsed.audio_base_dir == "audio"
    assert parsed.source_repo_id == "org/repo"
    assert parsed.source_revision == "main"
    assert parsed.source_commit_sha == "abcdef1234"
    assert parsed.default_language == "en"
    assert ("custom", "value") in parsed.extras
    assert parsed.as_dict()["custom"] == "value"


def test_parse_dataset_registry_options_rejects_invalid_source_id() -> None:
    """Typed options parser should reject invalid source id values."""
    with pytest.raises(ValueError, match="Invalid source_repo_id"):
        parse_dataset_registry_options({"source_repo_id": "invalid"})


def test_parse_dataset_registry_options_rejects_invalid_source_commit_sha() -> None:
    """Typed options parser should reject invalid commit SHA values."""
    with pytest.raises(ValueError, match="Invalid source_commit_sha"):
        parse_dataset_registry_options({"source_commit_sha": "not-a-commit"})


def test_remove_dataset_registry_entry_deletes_existing_entry(tmp_path: Path) -> None:
    """Registry removal should return removed entry and persist deletion."""

    settings = _settings(tmp_path)
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="ravdess",
        dataset_root=tmp_path / "datasets" / "ravdess",
        manifest_path=manifest_path,
        options={},
    )

    removed = remove_dataset_registry_entry(settings=settings, dataset_id="RAVDESS")

    assert removed is not None
    assert removed.dataset_id == "ravdess"
    assert "ravdess" not in load_dataset_registry(settings=settings)


def test_remove_dataset_registry_entry_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    """Registry removal should return None when dataset is not registered."""

    settings = _settings(tmp_path)

    removed = remove_dataset_registry_entry(settings=settings, dataset_id="ravdess")

    assert removed is None
