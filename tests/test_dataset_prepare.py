"""Tests for dataset descriptor resolution and manifest preparation orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data import dataset_prepare as dp
from ser.data.dataset_registry import DatasetRegistryEntry
from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.data.ontology import LabelOntology


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(folder=tmp_path / "data" / "models"),
            dataset=SimpleNamespace(subfolder_prefix="Actor_*", extension="*.wav"),
            emotions={"03": "happy", "04": "sad"},
            data_loader=SimpleNamespace(max_failed_file_ratio=0.1),
            default_language="en",
        ),
    )


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset({"happy", "sad"}),
    )


def _utterance(sample_id: str, audio_path: Path, label: str) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample_id,
        corpus="ravdess",
        audio_path=audio_path,
        label=label,
        speaker_id="ravdess:1",
    )


def test_resolve_dataset_descriptor_rejects_unknown_id() -> None:
    """Unsupported dataset ids should fail with an explicit error."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        dp.resolve_dataset_descriptor("unknown-dataset")


def test_prepare_dataset_manifest_for_ravdess_updates_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RAVDESS preparation should build manifest and upsert registry entry."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    captured: dict[str, object] = {}

    def _build_manifest(**kwargs: object) -> list[Utterance]:
        captured["build_kwargs"] = kwargs
        return [
            _utterance("ravdess:a.wav", dataset_root / "a.wav", "happy"),
            _utterance("ravdess:b.wav", dataset_root / "b.wav", "sad"),
        ]

    monkeypatch.setattr(dp, "build_ravdess_manifest_jsonl", _build_manifest)

    def _capture_registry_kwargs(**kwargs: object) -> None:
        captured["registry_kwargs"] = kwargs

    monkeypatch.setattr(
        dp,
        "upsert_dataset_registry_entry",
        _capture_registry_kwargs,
    )

    built = dp.prepare_dataset_manifest(
        settings=settings,
        dataset_id="ravdess",
        dataset_root=dataset_root,
        ontology=_ontology(),
        manifest_path=manifest_path,
        default_language="en",
    )

    assert built == [manifest_path]
    build_kwargs = captured["build_kwargs"]
    assert isinstance(build_kwargs, dict)
    assert build_kwargs["dataset_root"] == dataset_root
    assert build_kwargs["output_path"] == manifest_path
    registry_kwargs = captured["registry_kwargs"]
    assert isinstance(registry_kwargs, dict)
    assert registry_kwargs["dataset_id"] == "ravdess"
    assert registry_kwargs["dataset_root"] == dataset_root
    assert registry_kwargs["manifest_path"] == manifest_path


def test_prepare_from_registry_entry_passes_optional_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry rebuild should pass labels/audio/default_language options through."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(tmp_path / "labels.csv"),
            "audio_base_dir": str(tmp_path / "audio"),
            "default_language": "en",
        },
    )
    captured: dict[str, object] = {}

    def _prepare_dataset_manifest(**kwargs: object) -> list[Path]:
        captured["kwargs"] = kwargs
        return [manifest_path]

    monkeypatch.setattr(
        dp,
        "prepare_dataset_manifest",
        _prepare_dataset_manifest,
    )

    built = dp.prepare_from_registry_entry(
        settings=settings,
        entry=entry,
        ontology=_ontology(),
    )

    assert built == [manifest_path]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["dataset_id"] == "msp-podcast"
    assert kwargs["dataset_root"] == dataset_root
    assert kwargs["manifest_path"] == manifest_path
    assert kwargs["labels_csv_path"] == tmp_path / "labels.csv"
    assert kwargs["audio_base_dir"] == tmp_path / "audio"
    assert kwargs["default_language"] == "en"
