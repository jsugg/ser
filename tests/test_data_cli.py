"""Tests for dataset CLI helper commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data import cli as data_cli
from ser.data.application import (
    DatasetCapabilitySnapshotEntry,
    DatasetPrepareWorkflowResult,
    DatasetRegistrySnapshot,
    DatasetRegistrySnapshotEntry,
    DatasetRegistrySnapshotIssue,
    DatasetUninstallWorkflowResult,
)
from ser.data.dataset_consents import PersistedDatasetConsents
from ser.data.dataset_prepare import DatasetDescriptor


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(folder=tmp_path / "data" / "models"),
            default_language="en",
        ),
    )


def _workflow_result(
    *,
    dataset_id: str,
    dataset_root: Path,
    manifest_path: Path,
    manifest_paths: tuple[Path, ...],
    downloaded: bool,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
    source_commit_sha: str | None = None,
) -> DatasetPrepareWorkflowResult:
    descriptor = DatasetDescriptor(
        dataset_id=dataset_id,
        display_name=dataset_id.upper(),
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://example.invalid/source",
        requires_manual_download=False,
    )
    return DatasetPrepareWorkflowResult(
        descriptor=descriptor,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        manifest_paths=manifest_paths,
        downloaded=downloaded,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_commit_sha=source_commit_sha,
    )


def _registry_snapshot(
    *,
    entries: tuple[DatasetRegistrySnapshotEntry, ...] = (),
    issues: tuple[DatasetRegistrySnapshotIssue, ...] = (),
) -> DatasetRegistrySnapshot:
    return DatasetRegistrySnapshot(entries=entries, issues=issues)


def test_run_configure_command_show(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Configure --show should print persisted dataset consents."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        data_cli,
        "load_persisted_dataset_consents",
        lambda settings: PersistedDatasetConsents(
            policy_consents={"academic_only": "unit_test"},
            license_consents={"msp-academic-license": "unit_test"},
        ),
    )

    code = data_cli.run_configure_command(["--show"], settings=settings)

    captured = capsys.readouterr()
    assert code == 0
    assert "academic_only" in captured.out
    assert "msp-academic-license" in captured.out


def test_run_configure_command_requires_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mutating configure commands should require --persist."""
    settings = _settings(tmp_path)

    code = data_cli.run_configure_command(
        ["--accept-dataset-policy", "academic_only"],
        settings=settings,
    )

    assert code == 2


def test_run_data_download_command_prepares_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Data download command should resolve descriptor and invoke manifest preparation."""
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}
    manifest_path = tmp_path / "data" / "manifests" / "ravdess.jsonl"
    dataset_root = tmp_path / "data" / "datasets" / "ravdess"

    monkeypatch.setattr(
        data_cli,
        "_ensure_descriptor_consents",
        lambda **kwargs: captured.__setitem__("consent_kwargs", kwargs),
    )

    def _capture_workflow_kwargs(**kwargs: object) -> DatasetPrepareWorkflowResult:
        captured["workflow_kwargs"] = kwargs
        return _workflow_result(
            dataset_id="ravdess",
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            manifest_paths=(manifest_path,),
            downloaded=True,
        )

    monkeypatch.setattr(
        data_cli,
        "run_dataset_prepare_workflow",
        _capture_workflow_kwargs,
    )

    code = data_cli.run_data_command(
        ["download", "--dataset", "ravdess", "--accept-license"],
        settings=settings,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "Wrote manifest(s):" in output
    consent_kwargs = captured["consent_kwargs"]
    assert isinstance(consent_kwargs, dict)
    assert consent_kwargs["dataset_id"] == "ravdess"
    assert consent_kwargs["accept_license_flag"] is True
    workflow_kwargs = captured["workflow_kwargs"]
    assert isinstance(workflow_kwargs, dict)
    assert workflow_kwargs["dataset_id"] == "ravdess"
    assert workflow_kwargs["skip_download"] is False
    assert workflow_kwargs["manifest_path"] is None


def test_run_data_download_command_returns_2_on_manifest_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Known validation failures should return data-command usage error code."""
    settings = _settings(tmp_path)

    monkeypatch.setattr(data_cli, "_ensure_descriptor_consents", lambda **kwargs: None)

    def _raise_manifest_error(**kwargs: object) -> DatasetPrepareWorkflowResult:
        raise ValueError("MSP-Podcast manifest build requires labels CSV.")

    monkeypatch.setattr(data_cli, "run_dataset_prepare_workflow", _raise_manifest_error)

    code = data_cli.run_data_command(
        ["download", "--dataset", "msp-podcast"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "requires labels CSV" in captured.err


def test_run_data_download_command_returns_1_on_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Operational runtime failures should return internal error code."""
    settings = _settings(tmp_path)

    monkeypatch.setattr(data_cli, "_ensure_descriptor_consents", lambda **kwargs: None)

    def _raise_download_error(**kwargs: object) -> DatasetPrepareWorkflowResult:
        raise RuntimeError("MSP mirror validation failed.")

    monkeypatch.setattr(data_cli, "run_dataset_prepare_workflow", _raise_download_error)

    code = data_cli.run_data_command(
        ["download", "--dataset", "msp-podcast"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "mirror validation failed" in captured.err


def test_run_data_download_command_passes_source_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MSP source override flags should pass through to download orchestrator."""
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(data_cli, "_ensure_descriptor_consents", lambda **kwargs: None)

    def _capture_workflow_kwargs(**kwargs: object) -> DatasetPrepareWorkflowResult:
        captured["workflow_kwargs"] = kwargs
        manifest_path = tmp_path / "manifests" / "msp.jsonl"
        return _workflow_result(
            dataset_id="msp-podcast",
            dataset_root=tmp_path / "datasets" / "msp-podcast",
            manifest_path=manifest_path,
            manifest_paths=(manifest_path,),
            downloaded=True,
            source_repo_id="org/repo",
            source_revision="rev-1",
        )

    monkeypatch.setattr(
        data_cli, "run_dataset_prepare_workflow", _capture_workflow_kwargs
    )

    code = data_cli.run_data_command(
        [
            "download",
            "--dataset",
            "msp-podcast",
            "--source",
            "org/repo",
            "--source-revision",
            "rev-1",
        ],
        settings=settings,
    )

    assert code == 0
    workflow_kwargs = captured["workflow_kwargs"]
    assert isinstance(workflow_kwargs, dict)
    assert workflow_kwargs["source_repo_id"] == "org/repo"
    assert workflow_kwargs["source_revision"] == "rev-1"


def test_run_data_download_command_rejects_source_overrides_with_skip_download(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Source overrides with skipped acquisition should fail fast."""
    settings = _settings(tmp_path)

    code = data_cli.run_data_command(
        [
            "download",
            "--dataset",
            "msp-podcast",
            "--skip-download",
            "--source",
            "org/repo",
        ],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "cannot be used with --skip-download" in captured.err


def test_run_data_registry_command_show_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Registry command should handle empty registry cleanly."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        data_cli,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: _registry_snapshot(),
    )

    code = data_cli.run_data_command(["registry", "--show"], settings=settings)

    captured = capsys.readouterr()
    assert code == 0
    assert "Dataset registry is empty." in captured.out


def test_run_data_registry_command_prints_source_pin(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Registry command should print source pin provenance for MSP entries."""
    settings = _settings(tmp_path)
    snapshot = _registry_snapshot(
        entries=(
            DatasetRegistrySnapshotEntry(
                dataset_id="msp-podcast",
                dataset_root=tmp_path / "datasets" / "msp-podcast",
                manifest_path=tmp_path / "manifests" / "msp.jsonl",
                options={
                    "source_repo_id": "org/repo",
                    "source_revision": "rev-1",
                },
                source_repo_id="org/repo",
                source_revision="rev-1",
            ),
        ),
    )

    monkeypatch.setattr(
        data_cli,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: snapshot,
    )

    code = data_cli.run_data_command(["registry", "--show"], settings=settings)

    captured = capsys.readouterr()
    assert code == 0
    assert "- msp-podcast" in captured.out
    assert "source_pin: org/repo@rev-1" in captured.out


def test_run_data_registry_command_json_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Registry command should support JSON output for tooling."""
    settings = _settings(tmp_path)
    snapshot = _registry_snapshot(
        entries=(
            DatasetRegistrySnapshotEntry(
                dataset_id="msp-podcast",
                dataset_root=tmp_path / "datasets" / "msp-podcast",
                manifest_path=tmp_path / "manifests" / "msp.jsonl",
                options={
                    "source_repo_id": "org/repo",
                    "source_revision": "rev-1",
                },
                source_repo_id="org/repo",
                source_revision="rev-1",
            ),
        ),
    )

    monkeypatch.setattr(
        data_cli,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: snapshot,
    )

    code = data_cli.run_data_command(
        ["registry", "--format", "json"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert '"dataset_id": "msp-podcast"' in captured.out
    assert '"source_repo_id": "org/repo"' in captured.out
    assert '"issues": []' in captured.out


def test_run_data_registry_command_strict_fails_on_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry strict mode should fail when health issues exist."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        data_cli,
        "collect_dataset_registry_snapshot",
        lambda **kwargs: _registry_snapshot(
            issues=(
                DatasetRegistrySnapshotIssue(
                    dataset_id="msp-podcast",
                    code="source_provenance_mismatch",
                    message="mismatch",
                ),
            )
        ),
    )

    code = data_cli.run_data_command(["registry", "--strict"], settings=settings)

    assert code == 2


def test_run_data_catalog_command_defaults_to_installed_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Catalog command should print installed dataset capabilities by default."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        data_cli,
        "collect_dataset_capability_snapshot",
        lambda **kwargs: (
            DatasetCapabilitySnapshotEntry(
                dataset_id="ravdess",
                display_name="RAVDESS",
                registered=True,
                installed=True,
                manifest_exists=True,
                dataset_root=tmp_path / "datasets" / "ravdess",
                manifest_path=tmp_path / "manifests" / "ravdess.jsonl",
                referenced_audio_files=10,
                present_audio_files=10,
                nonempty_audio_files=10,
                dataset_size_bytes=1024,
                source_url="https://zenodo.org/records/1188976",
                policy_id="noncommercial",
                license_id="cc-by-nc-sa-4.0",
                modalities=("audio",),
                label_schema="emotion_8_class",
                has_label_mapping=True,
                supervised_ser_candidate=True,
                ssl_candidate=True,
                multimodal_candidate=False,
                mergeable_with_emotion_ontology=True,
                recommended_uses=("supervised_ser_training",),
                notes=("test-note",),
            ),
        ),
    )

    code = data_cli.run_data_command(["catalog"], settings=settings)

    captured = capsys.readouterr()
    assert code == 0
    assert "- ravdess (RAVDESS)" in captured.out
    assert "dataset_size: 1024 bytes" in captured.out
    assert "recommended_uses: supervised_ser_training" in captured.out


def test_run_data_catalog_command_json_format(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Catalog command should support JSON output for automation."""
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        data_cli,
        "collect_dataset_capability_snapshot",
        lambda **kwargs: (),
    )

    code = data_cli.run_data_command(
        ["catalog", "--format", "json"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert '"entries": []' in captured.out


def test_run_data_uninstall_command_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Uninstall command should remove one registered dataset."""

    settings = _settings(tmp_path)
    descriptor = DatasetDescriptor(
        dataset_id="ravdess",
        display_name="RAVDESS",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://zenodo.org/records/1188976",
        requires_manual_download=False,
    )
    monkeypatch.setattr(
        data_cli,
        "run_dataset_uninstall_workflow",
        lambda **kwargs: DatasetUninstallWorkflowResult(
            descriptor=descriptor,
            removed_from_registry=True,
            removed_manifest_paths=(tmp_path / "manifests" / "ravdess.jsonl",),
            removed_dataset_roots=(tmp_path / "datasets" / "ravdess",),
        ),
    )

    code = data_cli.run_data_command(
        ["uninstall", "--dataset", "ravdess"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Uninstalled dataset `ravdess`" in captured.out
    assert "removed_manifest" in captured.out
    assert "removed_dataset_root" in captured.out


def test_run_data_uninstall_command_returns_2_when_not_registered(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Uninstall command should return 2 when dataset is not registered."""

    settings = _settings(tmp_path)
    descriptor = DatasetDescriptor(
        dataset_id="ravdess",
        display_name="RAVDESS",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://zenodo.org/records/1188976",
        requires_manual_download=False,
    )
    monkeypatch.setattr(
        data_cli,
        "run_dataset_uninstall_workflow",
        lambda **kwargs: DatasetUninstallWorkflowResult(
            descriptor=descriptor,
            removed_from_registry=False,
            removed_manifest_paths=(),
            removed_dataset_roots=(),
        ),
    )

    code = data_cli.run_data_command(
        ["uninstall", "--dataset", "ravdess"],
        settings=settings,
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "is not registered" in captured.err
