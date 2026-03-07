"""Contract tests for dataset-controls helper projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from ser.models import dataset_controls


@dataclass(frozen=True)
class _Utterance:
    corpus: str
    language: str | None


@dataclass(frozen=True)
class _DatasetConfig:
    manifest_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _Settings:
    dataset: _DatasetConfig


def test_build_dataset_controls_prefers_registry_mode_when_manifest_empty() -> None:
    """Helper should promote mode to registry when registry manifests are present."""
    utterances = [
        _Utterance(corpus="ravdess", language="en"),
        _Utterance(corpus="crema-d", language=None),
    ]

    controls = dataset_controls.build_dataset_controls(
        utterances=utterances,
        manifest_paths=(),
        resolve_registry_manifest_paths=lambda: (
            Path("manifests/crema-d.jsonl"),
            Path("manifests/ravdess.jsonl"),
            Path("manifests/ravdess.jsonl"),
        ),
    )

    assert controls["mode"] == "registry"
    assert controls["manifest_paths"] == [
        "manifests/crema-d.jsonl",
        "manifests/ravdess.jsonl",
    ]
    assert controls["utterance_count"] == 2
    assert controls["corpus_counts"] == {"ravdess": 1, "crema-d": 1}
    assert controls["language_counts"] == {"en": 1, "unknown": 1}


def test_build_dataset_controls_keeps_manifest_mode_when_paths_configured() -> None:
    """Helper should keep manifest mode and avoid registry lookup when configured."""
    utterances = [
        _Utterance(corpus="emodb", language="de"),
    ]

    controls = dataset_controls.build_dataset_controls(
        utterances=utterances,
        manifest_paths=(Path("manifests/emodb.jsonl"),),
        resolve_registry_manifest_paths=lambda: (_ for _ in ()).throw(
            AssertionError("registry lookup should not run in manifest mode")
        ),
    )

    assert controls["mode"] == "manifest"
    assert controls["manifest_paths"] == ["manifests/emodb.jsonl"]
    assert controls["utterance_count"] == 1
    assert controls["corpus_counts"] == {"emodb": 1}
    assert controls["language_counts"] == {"de": 1}


def test_resolve_registry_manifest_paths_returns_sorted_unique_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry resolver should emit deterministic sorted unique manifest paths."""
    import ser.data.dataset_registry as dataset_registry

    monkeypatch.setattr(
        dataset_registry,
        "load_dataset_registry",
        lambda settings: {"loaded": bool(settings)},
    )
    monkeypatch.setattr(
        dataset_registry,
        "registered_manifest_paths",
        lambda settings: (
            Path("manifests/ravdess.jsonl"),
            Path("manifests/crema-d.jsonl"),
            Path("manifests/ravdess.jsonl"),
        ),
    )

    manifest_paths = dataset_controls.resolve_registry_manifest_paths(settings=object())

    assert manifest_paths == (
        Path("manifests/crema-d.jsonl"),
        Path("manifests/ravdess.jsonl"),
    )


def test_resolve_registry_manifest_paths_fails_closed_on_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry resolver should return empty tuple when lookup raises."""
    import ser.data.dataset_registry as dataset_registry

    def _raise_lookup_error(*, settings: object) -> object:
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(dataset_registry, "load_dataset_registry", _raise_lookup_error)

    assert dataset_controls.resolve_registry_manifest_paths(settings=object()) == ()


def test_build_dataset_controls_for_settings_uses_current_settings() -> None:
    """Settings-aware helper should project configured manifests and skip registry fallback."""
    utterances = [_Utterance(corpus="ravdess", language="en")]

    controls = dataset_controls.build_dataset_controls_for_settings(
        utterances,
        read_settings=lambda: _Settings(
            dataset=_DatasetConfig(
                manifest_paths=(Path("manifests/ravdess.jsonl"),),
            )
        ),
        resolve_registry_manifest_paths_for_settings=lambda **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("registry lookup should not run in manifest mode")),
    )

    assert controls["mode"] == "manifest"
    assert controls["manifest_paths"] == ["manifests/ravdess.jsonl"]
