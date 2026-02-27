"""Tests for dataset registry persistence and lookup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from ser.config import AppConfig
from ser.data.dataset_registry import (
    DatasetRegistryEntry,
    load_dataset_registry,
    registered_manifest_paths,
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
