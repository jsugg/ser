"""Tests for dataset strategy registry and contract behavior."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest
import soundfile as sf

import ser._internal.data.dataset_prepare as dataset_prepare
import ser._internal.data.strategies.default as default_strategies
from ser._internal.data.strategies import (
    CremaDDatasetStrategy,
    DatasetStrategyRegistry,
    Emodb2DatasetStrategy,
    PreparedManifestResult,
    RavdessDatasetStrategy,
    SpanishMeacorpus2023DatasetStrategy,
    build_default_dataset_strategies,
)


def _write_crema_wav(dataset_root: Path) -> Path:
    """Writes one minimal valid CREMA-D WAV fixture."""
    audio_path = dataset_root / "AudioWAV" / "1001_IEO_HAP_LO.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(audio_path, np.asarray([0.0, 0.25], dtype=np.float32), 16_000)
    return audio_path


def _download_crema_d(strategy: CremaDDatasetStrategy, dataset_root: Path) -> None:
    """Invokes the CREMA-D strategy with its static descriptor."""
    strategy.download(
        descriptor=dataset_prepare.SUPPORTED_DATASETS["crema-d"],
        dataset_root=dataset_root,
        source_repo_id=None,
        source_revision=None,
    )


def test_build_default_dataset_strategies_contains_expected_ids() -> None:
    """Default strategy factory should register every supported dataset id."""
    strategies = build_default_dataset_strategies()
    assert set(strategies) == {
        "ravdess",
        "crema-d",
        "msp-podcast",
        "emodb-2.0",
        "escorpus-pe",
        "mesd",
        "oreau-french-esd",
        "jl-corpus",
        "cafe",
        "asvp-esd",
        "emov-db",
        "pavoque",
        "att-hack",
        "coraa-ser",
        "spanish-meacorpus-2023",
        "biic-podcast",
    }


def test_default_strategies_expose_required_contract_members() -> None:
    """Each strategy instance should expose the expected contract members."""
    strategies = build_default_dataset_strategies()
    for strategy in strategies.values():
        assert isinstance(strategy.supports_source_overrides, bool)
        assert callable(strategy.download)
        assert callable(strategy.prepare_manifest)


def test_default_strategy_factory_preserves_public_strategy_types() -> None:
    """Factory output should preserve the public strategy class surface."""

    strategies = build_default_dataset_strategies()

    assert isinstance(strategies["ravdess"], RavdessDatasetStrategy)
    assert isinstance(strategies["emodb-2.0"], Emodb2DatasetStrategy)
    assert isinstance(
        strategies["spanish-meacorpus-2023"],
        SpanishMeacorpus2023DatasetStrategy,
    )


def test_source_override_support_is_only_enabled_for_msp() -> None:
    """Only MSP strategy should support source override pins."""
    strategies = build_default_dataset_strategies()
    assert strategies["msp-podcast"].supports_source_overrides is True
    assert strategies["ravdess"].supports_source_overrides is False
    assert strategies["crema-d"].supports_source_overrides is False
    assert strategies["emodb-2.0"].supports_source_overrides is False
    assert strategies["escorpus-pe"].supports_source_overrides is False
    assert strategies["mesd"].supports_source_overrides is False
    assert strategies["oreau-french-esd"].supports_source_overrides is False
    assert strategies["jl-corpus"].supports_source_overrides is False
    assert strategies["cafe"].supports_source_overrides is False
    assert strategies["asvp-esd"].supports_source_overrides is False
    assert strategies["emov-db"].supports_source_overrides is False
    assert strategies["pavoque"].supports_source_overrides is False
    assert strategies["att-hack"].supports_source_overrides is False
    assert strategies["coraa-ser"].supports_source_overrides is False
    assert strategies["spanish-meacorpus-2023"].supports_source_overrides is False
    assert strategies["biic-podcast"].supports_source_overrides is False


def test_crema_d_download_requires_git_lfs_before_clone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing Git LFS should fail before acquisition mutates the target."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        default_strategies.shutil,
        "which",
        lambda executable: "/fake/git" if executable == "git" else None,
    )
    monkeypatch.setattr(
        default_strategies.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )
    dataset_root = tmp_path / "crema-d"

    with pytest.raises(RuntimeError, match=r"git-lfs.*brew install git-lfs"):
        _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert commands == []
    assert dataset_root.exists()
    assert not any(dataset_root.iterdir())


def test_crema_d_lfs_pull_failure_is_fatal_and_cleans_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed LFS transfer must not publish a partial checkout."""
    staging_paths: list[Path] = []
    monkeypatch.setattr(default_strategies.shutil, "which", lambda name: f"/fake/{name}")

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "clone":
            staging_root = Path(command[-1])
            staging_paths.append(staging_root)
            (staging_root / ".git").mkdir(parents=True)
        if command[1:] == ["lfs", "pull"]:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
                stderr="quota exceeded",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(default_strategies.subprocess, "run", _run)
    dataset_root = tmp_path / "crema-d"

    with pytest.raises(RuntimeError, match="Git LFS pull failed: quota exceeded"):
        _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert dataset_root.exists()
    assert not any(dataset_root.iterdir())
    assert staging_paths and all(not path.exists() for path in staging_paths)


def test_crema_d_pointer_checkout_is_not_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful commands still require validated audio before atomic publication."""
    staging_paths: list[Path] = []
    monkeypatch.setattr(default_strategies.shutil, "which", lambda name: f"/fake/{name}")

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "clone":
            staging_root = Path(command[-1])
            staging_paths.append(staging_root)
            audio_path = staging_root / "AudioWAV" / "1077_WSI_ANG_XX.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\nsize 85462\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(default_strategies.subprocess, "run", _run)
    dataset_root = tmp_path / "crema-d"

    with pytest.raises(RuntimeError, match="unmaterialized Git LFS pointer"):
        _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert dataset_root.exists()
    assert not any(dataset_root.iterdir())
    assert staging_paths and all(not path.exists() for path in staging_paths)


def test_crema_d_download_publishes_only_validated_staging_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A validated checkout should atomically replace the pre-created empty root."""
    monkeypatch.setattr(default_strategies.shutil, "which", lambda name: f"/fake/{name}")

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "clone":
            staging_root = Path(command[-1])
            (staging_root / ".git").mkdir(parents=True)
            _write_crema_wav(staging_root)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(default_strategies.subprocess, "run", _run)
    dataset_root = tmp_path / "crema-d"

    _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert (dataset_root / ".git").is_dir()
    assert (dataset_root / "AudioWAV" / "1001_IEO_HAP_LO.wav").is_file()
    assert not list(tmp_path.glob(".crema-d.staging-*"))


def test_crema_d_existing_non_checkout_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invalid user data outside a Git checkout must never be deleted or overwritten."""
    dataset_root = tmp_path / "crema-d"
    pointer_path = dataset_root / "AudioWAV" / "1077_WSI_ANG_XX.wav"
    pointer_path.parent.mkdir(parents=True)
    pointer_bytes = b"version https://git-lfs.github.com/spec/v1\nsize 85462\n"
    pointer_path.write_bytes(pointer_bytes)
    monkeypatch.setattr(
        default_strategies.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess should not run"),
    )

    with pytest.raises(RuntimeError, match="Move it aside and retry"):
        _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert pointer_path.read_bytes() == pointer_bytes


def test_crema_d_existing_git_checkout_is_repaired_and_revalidated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compatible partial checkout should run explicit LFS repair commands."""
    dataset_root = tmp_path / "crema-d"
    (dataset_root / ".git").mkdir(parents=True)
    pointer_path = dataset_root / "AudioWAV" / "1077_WSI_ANG_XX.wav"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_bytes(b"version https://git-lfs.github.com/spec/v1\nsize 85462\n")
    commands: list[list[str]] = []
    monkeypatch.setattr(default_strategies.shutil, "which", lambda name: f"/fake/{name}")

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1:] == ["lfs", "checkout"]:
            _write_crema_wav(dataset_root)
            pointer_path.unlink()
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(default_strategies.subprocess, "run", _run)

    _download_crema_d(CremaDDatasetStrategy(), dataset_root)

    assert [command[1:] for command in commands] == [
        ["lfs", "install", "--local"],
        ["lfs", "pull"],
        ["lfs", "checkout"],
    ]
    assert (dataset_root / ".git").is_dir()
    assert (dataset_root / "AudioWAV" / "1001_IEO_HAP_LO.wav").is_file()


def test_dataset_strategy_registry_rejects_missing_supported_ids() -> None:
    """Registry construction should fail when supported ids are missing mappings."""
    with pytest.raises(ValueError, match="Missing strategy ids"):
        DatasetStrategyRegistry.from_mapping(
            strategies={"ravdess": build_default_dataset_strategies()["ravdess"]},
            supported_dataset_ids=("ravdess", "msp-podcast"),
        )


def test_dataset_strategy_registry_rejects_unknown_extra_ids() -> None:
    """Registry construction should fail when unknown extra ids are provided."""
    strategies = build_default_dataset_strategies()
    with pytest.raises(ValueError, match="Unknown strategy ids"):
        DatasetStrategyRegistry.from_mapping(
            strategies=strategies,
            supported_dataset_ids=("ravdess",),
        )


def test_dataset_strategy_registry_resolve_returns_expected_strategy() -> None:
    """Registry resolution should return the registered strategy instance."""
    strategies = build_default_dataset_strategies()
    registry = DatasetStrategyRegistry.from_mapping(
        strategies=strategies,
        supported_dataset_ids=tuple(strategies),
    )

    strategy = registry.resolve("msp-podcast")

    assert strategy is strategies["msp-podcast"]


def test_supported_datasets_and_default_strategy_factory_stay_in_sync() -> None:
    """Supported dataset descriptors must match default strategy registration ids."""
    strategies = build_default_dataset_strategies()
    supported_ids = set(dataset_prepare.SUPPORTED_DATASETS)

    assert set(strategies) == supported_ids
    assert set(dataset_prepare._DATASET_STRATEGY_REGISTRY.as_mapping()) == supported_ids


def test_dataset_strategy_registry_reports_missing_and_unknown_ids_together() -> None:
    """Registry error should include both missing and unknown id diagnostics."""

    class _FakeStrategy:
        supports_source_overrides = False

        def download(
            self,
            *,
            descriptor: object,
            dataset_root: Path,
            source_repo_id: str | None,
            source_revision: str | None,
        ) -> tuple[str | None, str | None]:
            del descriptor, dataset_root, source_repo_id, source_revision
            return (None, None)

        def prepare_manifest(self, **kwargs: object) -> PreparedManifestResult:
            del kwargs
            return PreparedManifestResult(manifest_paths=(), options={})

    strategies = {
        "ravdess": build_default_dataset_strategies()["ravdess"],
        "oops": _FakeStrategy(),
    }
    with pytest.raises(
        ValueError,
        match=r"Missing strategy ids: msp-podcast\..*Unknown strategy ids: oops\.",
    ):
        DatasetStrategyRegistry.from_mapping(
            strategies=strategies,
            supported_dataset_ids=("ravdess", "msp-podcast"),
        )


def test_dataset_strategy_registry_as_mapping_is_immutable() -> None:
    """Registry mapping view should be immutable and read-only."""
    strategies = build_default_dataset_strategies()
    registry = DatasetStrategyRegistry.from_mapping(
        strategies=strategies,
        supported_dataset_ids=tuple(strategies),
    )

    mapping = registry.as_mapping()
    assert isinstance(mapping, MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, object], mapping)["new-id"] = strategies["ravdess"]
