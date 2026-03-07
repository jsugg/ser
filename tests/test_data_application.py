"""Tests for dataset application services."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data import application as data_application
from ser.data.dataset_capabilities import list_dataset_capability_profiles
from ser.data.dataset_consents import load_persisted_dataset_consents
from ser.data.dataset_prepare import SUPPORTED_DATASETS
from ser.data.dataset_registry import upsert_dataset_registry_entry


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(folder=tmp_path / "data" / "models"),
            default_language="en",
            emotions={"03": "happy", "04": "sad"},
            data_loader=SimpleNamespace(max_failed_file_ratio=0.1),
            dataset=SimpleNamespace(subfolder_prefix="Actor_*", extension="*.wav"),
        ),
    )


def test_compute_dataset_descriptor_missing_consents_for_msp(tmp_path: Path) -> None:
    """MSP should report missing restricted policy/license without persisted consent."""
    settings = _settings(tmp_path)

    status = data_application.compute_dataset_descriptor_missing_consents(
        settings=settings,
        dataset_id="msp-podcast",
    )

    assert status.descriptor.dataset_id == "msp-podcast"
    assert status.missing_policy_consents == ("academic_only",)
    assert status.missing_license_consents == ("msp-academic-license",)


def test_persist_missing_dataset_descriptor_consents(tmp_path: Path) -> None:
    """Persist helper should write missing policy/license entries to consent store."""
    settings = _settings(tmp_path)

    data_application.persist_missing_dataset_descriptor_consents(
        settings=settings,
        missing_policy_consents=("academic_only",),
        missing_license_consents=("msp-academic-license",),
        source="unit-test",
    )

    consents = load_persisted_dataset_consents(settings=settings)
    assert "academic_only" in consents.policy_consents
    assert "msp-academic-license" in consents.license_consents


def test_run_dataset_prepare_workflow_rejects_source_override_with_skip_download(
    tmp_path: Path,
) -> None:
    """Skip-download workflow should reject source pin overrides."""
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="skip_download=True"):
        data_application.run_dataset_prepare_workflow(
            settings=settings,
            dataset_id="msp-podcast",
            skip_download=True,
            source_repo_id="org/repo",
            labels_csv_path=tmp_path / "labels.csv",
        )


def test_run_dataset_prepare_workflow_propagates_source_pin_to_manifest_prepare(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Workflow should pass resolved source pin from download into manifest preparation."""
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    dataset_root = tmp_path / "datasets" / "msp"

    monkeypatch.setattr(
        data_application,
        "resolve_label_ontology",
        lambda _settings: "ontology-stub",
    )

    def _capture_download(**kwargs: object) -> tuple[str, str]:
        captured["download_kwargs"] = kwargs
        return ("org/repo", "rev-1")

    def _capture_prepare(**kwargs: object) -> list[Path]:
        captured["prepare_kwargs"] = kwargs
        return [manifest_path]

    monkeypatch.setattr(data_application, "download_dataset", _capture_download)
    monkeypatch.setattr(data_application, "prepare_dataset_manifest", _capture_prepare)

    result = data_application.run_dataset_prepare_workflow(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        labels_csv_path=tmp_path / "labels.csv",
        source_repo_id="org/repo",
        source_revision="rev-1",
    )

    assert result.downloaded is True
    assert result.source_repo_id == "org/repo"
    assert result.source_revision == "rev-1"
    prepare_kwargs = captured["prepare_kwargs"]
    assert isinstance(prepare_kwargs, dict)
    assert prepare_kwargs["source_repo_id"] == "org/repo"
    assert prepare_kwargs["source_revision"] == "rev-1"


def test_collect_dataset_registry_snapshot_returns_typed_entries_and_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Snapshot collector should return typed entry rows and mapped issues."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp-podcast.jsonl"
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234567890",
        },
    )
    monkeypatch.setattr(
        data_application,
        "collect_dataset_registry_health_issues",
        lambda **kwargs: (
            SimpleNamespace(
                dataset_id="msp-podcast",
                code="source_provenance_mismatch",
                message="Mismatch",
            ),
        ),
    )

    snapshot = data_application.collect_dataset_registry_snapshot(settings=settings)

    assert len(snapshot.entries) == 1
    entry = snapshot.entries[0]
    assert entry.dataset_id == "msp-podcast"
    assert entry.source_repo_id == "org/repo"
    assert entry.source_revision == "rev-1"
    assert entry.source_commit_sha == "abcdef1234567890"
    assert len(snapshot.issues) == 1
    issue = snapshot.issues[0]
    assert issue.dataset_id == "msp-podcast"
    assert issue.code == "source_provenance_mismatch"


def test_build_dataset_registry_snapshot_json_payload_serializes_paths(
    tmp_path: Path,
) -> None:
    """JSON payload builder should serialize path fields to strings."""
    snapshot = data_application.DatasetRegistrySnapshot(
        entries=(
            data_application.DatasetRegistrySnapshotEntry(
                dataset_id="msp-podcast",
                dataset_root=tmp_path / "datasets" / "msp-podcast",
                manifest_path=tmp_path / "manifests" / "msp.jsonl",
                options={"source_repo_id": "org/repo"},
                source_repo_id="org/repo",
                source_revision="rev-1",
                source_commit_sha="abcdef1234567890",
            ),
        ),
        issues=(
            data_application.DatasetRegistrySnapshotIssue(
                dataset_id="msp-podcast",
                code="source_provenance_mismatch",
                message="Mismatch",
            ),
        ),
    )

    payload = data_application.build_dataset_registry_snapshot_json_payload(snapshot)

    entries = payload["entries"]
    assert isinstance(entries, list)
    assert entries[0]["dataset_id"] == "msp-podcast"
    assert isinstance(entries[0]["dataset_root"], str)
    assert entries[0]["source_commit_sha"] == "abcdef1234567890"
    assert payload["issues"] == [
        {
            "dataset_id": "msp-podcast",
            "code": "source_provenance_mismatch",
            "message": "Mismatch",
        }
    ]


def test_collect_dataset_capability_snapshot_defaults_to_installed_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Capability snapshot should return only installed datasets by default."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    dataset_root.mkdir(parents=True, exist_ok=True)
    audio_path = dataset_root / "sample.wav"
    audio_path.write_bytes(b"audio-bytes")
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"audio_path":"sample.wav","corpus":"ravdess","label":"happy","sample_id":"x","schema_version":1}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        data_application,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: data_application.DatasetRegistrySnapshot(
            entries=(
                data_application.DatasetRegistrySnapshotEntry(
                    dataset_id="ravdess",
                    dataset_root=dataset_root,
                    manifest_path=manifest_path,
                    options={},
                    source_repo_id=None,
                    source_revision=None,
                    source_commit_sha=None,
                ),
            ),
            issues=(),
        ),
    )

    rows = data_application.collect_dataset_capability_snapshot(settings=settings)

    assert len(rows) == 1
    assert rows[0].dataset_id == "ravdess"
    assert rows[0].installed is True
    assert rows[0].ssl_candidate is True


def test_build_dataset_capability_snapshot_json_payload_serializes_optional_paths(
    tmp_path: Path,
) -> None:
    """Capability JSON payload should serialize optional path fields safely."""
    payload = data_application.build_dataset_capability_snapshot_json_payload(
        (
            data_application.DatasetCapabilitySnapshotEntry(
                dataset_id="spanish-meacorpus-2023",
                display_name="Spanish MEACorpus 2023",
                registered=False,
                installed=False,
                manifest_exists=False,
                dataset_root=None,
                manifest_path=None,
                referenced_audio_files=0,
                present_audio_files=0,
                nonempty_audio_files=0,
                dataset_size_bytes=0,
                source_url="https://zenodo.org/records/18606423",
                policy_id="noncommercial",
                license_id="cc-by-nc-4.0",
                modalities=("audio", "text", "metadata"),
                label_schema="emotion_mapped_to_canonical",
                has_label_mapping=True,
                supervised_ser_candidate=True,
                ssl_candidate=True,
                multimodal_candidate=True,
                mergeable_with_emotion_ontology=True,
                recommended_uses=("audio_text_emotion_fusion",),
                notes=("metadata only",),
            ),
        )
    )

    entries = payload["entries"]
    assert isinstance(entries, list)
    assert entries[0]["dataset_id"] == "spanish-meacorpus-2023"
    assert entries[0]["registered"] is False
    assert entries[0]["dataset_root"] is None
    assert entries[0]["manifest_path"] is None
    assert entries[0]["dataset_size_bytes"] == 0


def test_dataset_capability_profiles_cover_all_supported_dataset_ids() -> None:
    """Capability catalog should stay aligned with supported dataset registry."""

    profile_ids = {profile.dataset_id for profile in list_dataset_capability_profiles()}
    assert profile_ids == set(SUPPORTED_DATASETS)


def test_collect_dataset_capability_snapshot_marks_zero_byte_audio_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registered datasets with only zero-byte referenced audio are not installed."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "biic-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    audio_path = dataset_root / "placeholder.wav"
    audio_path.write_bytes(b"")
    manifest_path = tmp_path / "manifests" / "biic-podcast.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '{"audio_path":"placeholder.wav","corpus":"biic-podcast","label":"happy","sample_id":"x","schema_version":1}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        data_application,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: data_application.DatasetRegistrySnapshot(
            entries=(
                data_application.DatasetRegistrySnapshotEntry(
                    dataset_id="biic-podcast",
                    dataset_root=dataset_root,
                    manifest_path=manifest_path,
                    options={},
                    source_repo_id=None,
                    source_revision=None,
                    source_commit_sha=None,
                ),
            ),
            issues=(),
        ),
    )

    rows = data_application.collect_dataset_capability_snapshot(
        settings=settings,
        include_uninstalled=True,
    )

    row = next(item for item in rows if item.dataset_id == "biic-podcast")
    assert row.registered is True
    assert row.installed is False
    assert row.dataset_size_bytes == 0


def test_run_dataset_uninstall_workflow_removes_registry_and_files(
    tmp_path: Path,
) -> None:
    """Uninstall workflow should remove registry entry and local files by default."""

    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.wav").write_bytes(b"audio")
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="ravdess",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={},
    )

    result = data_application.run_dataset_uninstall_workflow(
        settings=settings,
        dataset_id="ravdess",
    )

    assert result.removed_from_registry is True
    assert result.removed_manifest_paths == (manifest_path,)
    assert result.removed_dataset_roots == (dataset_root,)
    snapshot = data_application.collect_dataset_registry_snapshot(settings=settings)
    assert snapshot.entries == ()
    assert manifest_path.exists() is False
    assert dataset_root.exists() is False


def test_run_dataset_uninstall_workflow_keep_files(tmp_path: Path) -> None:
    """Uninstall workflow should preserve files when remove_files is disabled."""

    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sample.wav").write_bytes(b"audio")
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="ravdess",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={},
    )

    result = data_application.run_dataset_uninstall_workflow(
        settings=settings,
        dataset_id="ravdess",
        remove_files=False,
    )

    assert result.removed_from_registry is True
    assert result.removed_manifest_paths == ()
    assert result.removed_dataset_roots == ()
    assert manifest_path.exists() is True
    assert dataset_root.exists() is True


def test_run_dataset_uninstall_workflow_returns_not_registered(tmp_path: Path) -> None:
    """Uninstall workflow should return not-registered result when missing."""

    settings = _settings(tmp_path)

    result = data_application.run_dataset_uninstall_workflow(
        settings=settings,
        dataset_id="ravdess",
    )

    assert result.descriptor.dataset_id == "ravdess"
    assert result.removed_from_registry is False
    assert result.removed_manifest_paths == ()
    assert result.removed_dataset_roots == ()


def test_run_dataset_uninstall_workflow_removes_default_paths_when_registry_path_differs(
    tmp_path: Path,
) -> None:
    """Uninstall should clean default dataset root even with stale custom registry paths."""

    settings = _settings(tmp_path)
    stale_dataset_root = tmp_path / "stale" / "msp-podcast"
    stale_dataset_root.mkdir(parents=True, exist_ok=True)
    stale_manifest_path = tmp_path / "stale" / "msp-podcast.jsonl"
    stale_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stale_manifest_path.write_text("{}", encoding="utf-8")
    default_root = data_application.default_dataset_root(settings, "msp-podcast")
    default_root.mkdir(parents=True, exist_ok=True)
    (default_root / "partial.bin").write_bytes(b"partial")
    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=stale_dataset_root,
        manifest_path=stale_manifest_path,
        options={},
    )

    result = data_application.run_dataset_uninstall_workflow(
        settings=settings,
        dataset_id="msp-podcast",
    )

    assert result.removed_from_registry is True
    assert default_root in result.removed_dataset_roots
    assert stale_dataset_root in result.removed_dataset_roots
    assert default_root.exists() is False
