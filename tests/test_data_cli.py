"""Tests for dataset CLI helper commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data import cli as data_cli
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


def test_run_configure_command_show(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Configure --show should print persisted dataset consents."""
    monkeypatch.setattr(data_cli, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        data_cli,
        "load_persisted_dataset_consents",
        lambda settings: PersistedDatasetConsents(
            policy_consents={"academic_only": "unit_test"},
            license_consents={"msp-academic-license": "unit_test"},
        ),
    )

    code = data_cli.run_configure_command(["--show"])

    captured = capsys.readouterr()
    assert code == 0
    assert "academic_only" in captured.out
    assert "msp-academic-license" in captured.out


def test_run_configure_command_requires_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mutating configure commands should require --persist."""
    monkeypatch.setattr(data_cli, "get_settings", lambda: _settings(tmp_path))

    code = data_cli.run_configure_command(["--accept-dataset-policy", "academic_only"])

    assert code == 2


def test_run_data_download_command_prepares_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Data download command should resolve descriptor and invoke manifest preparation."""
    settings = _settings(tmp_path)
    descriptor = DatasetDescriptor(
        dataset_id="ravdess",
        display_name="RAVDESS",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://example.invalid/ravdess",
        requires_manual_download=True,
    )
    captured: dict[str, object] = {}
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"

    monkeypatch.setattr(data_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        data_cli,
        "resolve_dataset_descriptor",
        lambda dataset_id: descriptor,
    )
    monkeypatch.setattr(
        data_cli,
        "_ensure_descriptor_consents",
        lambda **kwargs: captured.__setitem__("consent_kwargs", kwargs),
    )
    monkeypatch.setattr(
        data_cli,
        "_resolve_label_ontology",
        lambda _settings: "ontology-stub",
    )

    def _capture_prepare_kwargs(**kwargs: object) -> list[Path]:
        captured["prepare_kwargs"] = kwargs
        return [manifest_path]

    monkeypatch.setattr(
        data_cli,
        "prepare_dataset_manifest",
        _capture_prepare_kwargs,
    )
    monkeypatch.setattr(
        data_cli,
        "download_dataset",
        lambda **kwargs: captured.__setitem__("download_kwargs", kwargs),
    )

    code = data_cli.run_data_command(
        ["download", "--dataset", "ravdess", "--accept-license"]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "Wrote manifest(s):" in output
    consent_kwargs = captured["consent_kwargs"]
    assert isinstance(consent_kwargs, dict)
    assert consent_kwargs["dataset_id"] == "ravdess"
    assert consent_kwargs["accept_license_flag"] is True
    prepare_kwargs = captured["prepare_kwargs"]
    assert isinstance(prepare_kwargs, dict)
    assert prepare_kwargs["dataset_id"] == "ravdess"
    assert prepare_kwargs["ontology"] == "ontology-stub"
    assert (
        prepare_kwargs["manifest_path"]
        == settings.models.folder.parent / "manifests" / "ravdess.jsonl"
    )
