"""Tests for dataset descriptor resolution and manifest preparation orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data import dataset_prepare as dp
from ser.data.dataset_registry import DatasetRegistryEntry
from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.data.ontology import LabelOntology
from ser.data.strategies.base import DatasetStrategyRegistry, PreparedManifestResult


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


def test_supported_datasets_have_registered_strategies() -> None:
    """Every supported dataset id should resolve to one concrete strategy."""
    for dataset_id in dp.SUPPORTED_DATASETS:
        strategy = dp._resolve_dataset_strategy(dataset_id)
        assert strategy is not None
    assert dp._resolve_dataset_strategy("msp-podcast").supports_source_overrides is True
    assert dp._resolve_dataset_strategy("ravdess").supports_source_overrides is False


def test_build_dataset_strategy_registry_with_explicit_consistent_mapping() -> None:
    """Explicit strategy mapping should build one registry when ids are consistent."""
    strategies = dp.build_default_dataset_strategies()

    registry = dp._build_dataset_strategy_registry(
        strategies=strategies,
        supported_dataset_ids=tuple(strategies),
    )

    assert registry.resolve("msp-podcast") is strategies["msp-podcast"]


def test_build_dataset_strategy_registry_surfaces_context_on_integrity_error() -> None:
    """Registry builder should surface contextual integrity errors on id mismatches."""

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

    with pytest.raises(
        ValueError,
        match=(
            r"Dataset strategy registry initialization failed\..*"
            r"Missing strategy ids: msp-podcast\."
        ),
    ):
        dp._build_dataset_strategy_registry(
            strategies={"ravdess": _FakeStrategy()},
            supported_dataset_ids=("ravdess", "msp-podcast"),
        )


def test_download_dataset_surfaces_context_when_strategy_registry_is_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Download workflow should surface contextual registry-resolution errors."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    empty_registry = DatasetStrategyRegistry.from_mapping(
        strategies={},
        supported_dataset_ids=(),
    )
    monkeypatch.setattr(dp, "_DATASET_STRATEGY_REGISTRY", empty_registry)

    with pytest.raises(
        ValueError,
        match=(
            r"Dataset strategy resolution failed for dataset_id='ravdess'\..*"
            r"No strategy registered for dataset 'ravdess'\."
        ),
    ):
        dp.download_dataset(
            settings=settings,
            dataset_id="ravdess",
            dataset_root=dataset_root,
        )


def test_prepare_dataset_manifest_surfaces_context_when_registry_is_inconsistent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Manifest workflow should surface contextual registry-resolution errors."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    empty_registry = DatasetStrategyRegistry.from_mapping(
        strategies={},
        supported_dataset_ids=(),
    )
    monkeypatch.setattr(dp, "_DATASET_STRATEGY_REGISTRY", empty_registry)

    with pytest.raises(
        ValueError,
        match=(
            r"Dataset strategy resolution failed for dataset_id='ravdess'\..*"
            r"No strategy registered for dataset 'ravdess'\."
        ),
    ):
        dp.prepare_dataset_manifest(
            settings=settings,
            dataset_id="ravdess",
            dataset_root=dataset_root,
            ontology=_ontology(),
            manifest_path=manifest_path,
        )


def test_download_dataset_logs_strategy_phase_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Download path should emit one structured success strategy log."""

    class _FakeDownloadStrategy:
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
            raise AssertionError("unreachable")

    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    monkeypatch.setattr(
        dp, "_resolve_dataset_strategy", lambda dataset_id: _FakeDownloadStrategy()
    )

    with caplog.at_level("INFO", logger=dp.logger.name):
        result = dp.download_dataset(
            settings=settings,
            dataset_id="ravdess",
            dataset_root=dataset_root,
        )

    assert result == (None, None)
    assert any(
        "phase=download outcome=success dataset_id=ravdess strategy=_FakeDownloadStrategy"
        in record.message
        for record in caplog.records
    )


def test_download_dataset_logs_strategy_phase_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Download path should emit one structured failure strategy log."""

    class _FakeDownloadFailureStrategy:
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
            raise RuntimeError("boom")

        def prepare_manifest(self, **kwargs: object) -> PreparedManifestResult:
            raise AssertionError("unreachable")

    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    monkeypatch.setattr(
        dp,
        "_resolve_dataset_strategy",
        lambda dataset_id: _FakeDownloadFailureStrategy(),
    )

    with caplog.at_level("INFO", logger=dp.logger.name):
        with pytest.raises(RuntimeError, match="boom"):
            dp.download_dataset(
                settings=settings,
                dataset_id="ravdess",
                dataset_root=dataset_root,
            )

    assert any(
        "phase=download outcome=failure dataset_id=ravdess strategy=_FakeDownloadFailureStrategy"
        in record.message
        and "error=RuntimeError" in record.message
        for record in caplog.records
    )


def test_prepare_manifest_logs_strategy_phase_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Prepare path should emit one structured success strategy log."""

    class _FakePrepareStrategy:
        supports_source_overrides = False

        def download(self, **kwargs: object) -> tuple[str | None, str | None]:
            raise AssertionError("unreachable")

        def prepare_manifest(
            self,
            *,
            settings: AppConfig,
            descriptor: object,
            dataset_root: Path,
            ontology: LabelOntology,
            manifest_path: Path,
            language: str,
            labels_csv_path: Path | None,
            audio_base_dir: Path | None,
            options: dict[str, str],
        ) -> PreparedManifestResult:
            del (
                settings,
                descriptor,
                dataset_root,
                ontology,
                language,
                labels_csv_path,
                audio_base_dir,
            )
            return PreparedManifestResult(
                manifest_paths=(manifest_path,), options=options
            )

    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    manifest_path = tmp_path / "manifests" / "ravdess.jsonl"
    monkeypatch.setattr(
        dp, "_resolve_dataset_strategy", lambda dataset_id: _FakePrepareStrategy()
    )
    monkeypatch.setattr(dp, "upsert_dataset_registry_entry", lambda **kwargs: None)

    with caplog.at_level("INFO", logger=dp.logger.name):
        built = dp.prepare_dataset_manifest(
            settings=settings,
            dataset_id="ravdess",
            dataset_root=dataset_root,
            ontology=_ontology(),
            manifest_path=manifest_path,
        )

    assert built == [manifest_path]
    assert any(
        "phase=prepare_manifest outcome=success dataset_id=ravdess strategy=_FakePrepareStrategy"
        in record.message
        for record in caplog.records
    )


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

    monkeypatch.setattr(
        "ser.data.strategies.default.build_ravdess_manifest_jsonl",
        _build_manifest,
    )

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
    labels_csv_path = tmp_path / "labels.csv"
    audio_base_dir = tmp_path / "audio"
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_csv_path.write_text("FileName,emotion\nx.wav,happy\n", encoding="utf-8")
    audio_base_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/repo",
                    "revision": "rev-1",
                    "commit_sha": "abcdef1234",
                }
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(labels_csv_path),
            "audio_base_dir": str(audio_base_dir),
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234",
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
    assert kwargs["labels_csv_path"] == labels_csv_path
    assert kwargs["audio_base_dir"] == audio_base_dir
    assert kwargs["source_repo_id"] == "org/repo"
    assert kwargs["source_revision"] == "rev-1"
    assert kwargs["source_commit_sha"] == "abcdef1234"
    assert kwargs["default_language"] == "en"


def test_download_dataset_msp_podcast_uses_hf_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MSP-Podcast download should delegate to mirror acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    captured: dict[str, object] = {}

    def _prepare_msp_podcast_from_hf_mirror(
        *,
        dataset_root: Path,
        repo_id: str,
        revision: str,
    ) -> object:
        captured["dataset_root"] = dataset_root
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            audio_dir=dataset_root / "audio",
            rows_seen=10,
            labels_written=10,
        )

    monkeypatch.setattr(
        "ser.data.strategies.default.prepare_msp_podcast_from_hf_mirror",
        _prepare_msp_podcast_from_hf_mirror,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        source_repo_id="org/repo",
        source_revision="rev-1",
    )

    assert captured["dataset_root"] == dataset_root
    assert captured["repo_id"] == "org/repo"
    assert captured["revision"] == "rev-1"


def test_download_dataset_ravdess_uses_zenodo_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """RAVDESS download should delegate to Zenodo acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"
    captured: dict[str, object] = {}

    def _prepare_ravdess_from_zenodo(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            files_seen=42,
            source_manifest_path=dataset_root / "source_manifest.json",
        )

    monkeypatch.setattr(
        "ser.data.strategies.default.prepare_ravdess_from_zenodo",
        _prepare_ravdess_from_zenodo,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="ravdess",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_emodb_uses_zenodo_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """EmoDB 2.0 download should delegate to Zenodo acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "emodb-2.0"
    captured: dict[str, object] = {}

    def _prepare_emodb_2_from_zenodo(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=10,
            labels_written=10,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("emodb-2.0"),
        "_download_preparer",
        _prepare_emodb_2_from_zenodo,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="emodb-2.0",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_cafe_uses_zenodo_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CaFE download should delegate to Zenodo acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "cafe"
    captured: dict[str, object] = {}

    def _prepare_cafe_from_zenodo(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("cafe"),
        "_download_preparer",
        _prepare_cafe_from_zenodo,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="cafe",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_asvp_esd_uses_zenodo_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ASVP-ESD download should delegate to Zenodo acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "asvp-esd"
    captured: dict[str, object] = {}

    def _prepare_asvp_esd_from_zenodo(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("asvp-esd"),
        "_download_preparer",
        _prepare_asvp_esd_from_zenodo,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="asvp-esd",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_emov_db_uses_openslr_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """EmoV-DB download should delegate to OpenSLR acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "emov-db"
    captured: dict[str, object] = {}

    def _prepare_emov_db_from_openslr(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("emov-db"),
        "_download_preparer",
        _prepare_emov_db_from_openslr,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="emov-db",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_pavoque_uses_github_release_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PAVOQUE download should delegate to GitHub release acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "pavoque"
    captured: dict[str, object] = {}

    def _prepare_pavoque_from_github_release(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("pavoque"),
        "_download_preparer",
        _prepare_pavoque_from_github_release,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="pavoque",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_att_hack_uses_openslr_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Att-HACK download should delegate to OpenSLR acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "att-hack"
    captured: dict[str, object] = {}

    def _prepare_att_hack_from_openslr(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("att-hack"),
        "_download_preparer",
        _prepare_att_hack_from_openslr,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="att-hack",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_coraa_ser_uses_google_drive_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CORAA SER download should delegate to Google Drive acquisition utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "coraa-ser"
    captured: dict[str, object] = {}

    def _prepare_coraa_ser_from_google_drive(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("coraa-ser"),
        "_download_preparer",
        _prepare_coraa_ser_from_google_drive,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="coraa-ser",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_spanish_meacorpus_uses_zenodo_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Spanish MEACorpus download should delegate to Zenodo metadata utility."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "spanish-meacorpus-2023"
    captured: dict[str, object] = {}

    def _prepare_spanish_meacorpus_2023_from_zenodo(*, dataset_root: Path) -> object:
        captured["dataset_root"] = dataset_root
        return SimpleNamespace(
            dataset_root=dataset_root,
            labels_csv_path=dataset_root / "labels.csv",
            source_manifest_path=dataset_root / "source_manifest.json",
            files_seen=8,
            labels_written=8,
        )

    monkeypatch.setattr(
        dp._resolve_dataset_strategy("spanish-meacorpus-2023"),
        "_download_preparer",
        _prepare_spanish_meacorpus_2023_from_zenodo,
    )

    dp.download_dataset(
        settings=settings,
        dataset_id="spanish-meacorpus-2023",
        dataset_root=dataset_root,
    )

    assert captured["dataset_root"] == dataset_root


def test_download_dataset_rejects_source_override_for_non_msp(tmp_path: Path) -> None:
    """Source overrides should be rejected for non-MSP datasets."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "ravdess"

    with pytest.raises(ValueError, match="supported only for `msp-podcast`"):
        dp.download_dataset(
            settings=settings,
            dataset_id="ravdess",
            dataset_root=dataset_root,
            source_repo_id="org/repo",
        )


def test_prepare_dataset_manifest_msp_uses_generated_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MSP manifest build should infer generated labels/audio defaults."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    labels_csv_path = dataset_root / "labels.csv"
    audio_base_dir = dataset_root / "audio"
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_csv_path.write_text("FileName,emotion\nx.wav,happy\n", encoding="utf-8")
    audio_base_dir.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _build_manifest(**kwargs: object) -> list[Utterance]:
        captured["build_kwargs"] = kwargs
        return [_utterance("msp:x.wav", audio_base_dir / "x.wav", "happy")]

    def _capture_registry_kwargs(**kwargs: object) -> None:
        captured["registry_kwargs"] = kwargs

    monkeypatch.setattr(
        "ser.data.strategies.default.build_msp_podcast_manifest_jsonl",
        _build_manifest,
    )
    monkeypatch.setattr(dp, "upsert_dataset_registry_entry", _capture_registry_kwargs)

    built = dp.prepare_dataset_manifest(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        ontology=_ontology(),
        manifest_path=manifest_path,
        labels_csv_path=None,
        audio_base_dir=None,
        source_repo_id="org/repo",
        source_revision="rev-1",
    )

    assert built == [manifest_path]
    build_kwargs = captured["build_kwargs"]
    assert isinstance(build_kwargs, dict)
    assert build_kwargs["labels_csv_path"] == labels_csv_path
    assert build_kwargs["audio_base_dir"] == audio_base_dir
    registry_kwargs = captured["registry_kwargs"]
    assert isinstance(registry_kwargs, dict)
    options = registry_kwargs["options"]
    assert isinstance(options, dict)
    assert options["labels_csv_path"] == str(labels_csv_path)
    assert options["audio_base_dir"] == str(audio_base_dir)
    assert options["source_repo_id"] == "org/repo"
    assert options["source_revision"] == "rev-1"


def test_prepare_dataset_manifest_msp_persists_source_commit_from_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MSP manifest build should persist commit SHA provenance when available."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    labels_csv_path = dataset_root / "labels.csv"
    audio_base_dir = dataset_root / "audio"
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_csv_path.write_text("FileName,emotion\nx.wav,happy\n", encoding="utf-8")
    audio_base_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/repo",
                    "revision": "rev-1",
                    "commit_sha": "abcdef1234567890",
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _build_manifest(**kwargs: object) -> list[Utterance]:
        captured["build_kwargs"] = kwargs
        return [_utterance("msp:x.wav", audio_base_dir / "x.wav", "happy")]

    def _capture_registry_kwargs(**kwargs: object) -> None:
        captured["registry_kwargs"] = kwargs

    monkeypatch.setattr(
        "ser.data.strategies.default.build_msp_podcast_manifest_jsonl",
        _build_manifest,
    )
    monkeypatch.setattr(dp, "upsert_dataset_registry_entry", _capture_registry_kwargs)

    built = dp.prepare_dataset_manifest(
        settings=settings,
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        ontology=_ontology(),
        manifest_path=manifest_path,
        labels_csv_path=None,
        audio_base_dir=None,
    )

    assert built == [manifest_path]
    registry_kwargs = captured["registry_kwargs"]
    assert isinstance(registry_kwargs, dict)
    options = registry_kwargs["options"]
    assert isinstance(options, dict)
    assert options["source_repo_id"] == "org/repo"
    assert options["source_revision"] == "rev-1"
    assert options["source_commit_sha"] == "abcdef1234567890"


def test_prepare_dataset_manifest_msp_requires_labels_if_not_generated(
    tmp_path: Path,
) -> None:
    """MSP manifest build should fail fast when labels CSV is unavailable."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    dataset_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="requires labels CSV"):
        dp.prepare_dataset_manifest(
            settings=settings,
            dataset_id="msp-podcast",
            dataset_root=dataset_root,
            ontology=_ontology(),
            manifest_path=manifest_path,
            labels_csv_path=None,
            audio_base_dir=None,
        )


def test_prepare_from_registry_entry_detects_msp_source_provenance_mismatch(
    tmp_path: Path,
) -> None:
    """Registry source pin should be consistent with local MSP mirror manifest."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/actual",
                    "revision": "actual-rev",
                }
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=tmp_path / "manifests" / "msp.jsonl",
        options={
            "labels_csv_path": str(dataset_root / "labels.csv"),
            "source_repo_id": "org/expected",
            "source_revision": "expected-rev",
        },
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        dp.prepare_from_registry_entry(
            settings=settings,
            entry=entry,
            ontology=_ontology(),
        )


def test_prepare_from_registry_entry_requires_manifest_for_pinned_source(
    tmp_path: Path,
) -> None:
    """Pinned source provenance should fail closed when mirror manifest is missing."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=tmp_path / "manifests" / "msp.jsonl",
        options={
            "labels_csv_path": str(dataset_root / "labels.csv"),
            "source_repo_id": "org/expected",
            "source_revision": "expected-rev",
        },
    )

    with pytest.raises(ValueError, match="manifest .* is missing"):
        dp.prepare_from_registry_entry(
            settings=settings,
            entry=entry,
            ontology=_ontology(),
        )


def test_prepare_from_registry_entry_rejects_unpinned_registry_with_manifest_source(
    tmp_path: Path,
) -> None:
    """Manifest source metadata requires persisted registry source pin for drift checks."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/actual",
                    "revision": "actual-rev",
                }
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=tmp_path / "manifests" / "msp.jsonl",
        options={"labels_csv_path": str(dataset_root / "labels.csv")},
    )

    with pytest.raises(ValueError, match="missing source pin"):
        dp.prepare_from_registry_entry(
            settings=settings,
            entry=entry,
            ontology=_ontology(),
        )


def test_collect_registry_health_reports_unpinned_manifest_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry health should flag unpinned MSP entries when mirror provenance exists."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/actual",
                    "revision": "actual-rev",
                }
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=tmp_path / "manifests" / "msp.jsonl",
        options={"labels_csv_path": str(dataset_root / "labels.csv")},
    )
    monkeypatch.setattr(
        dp,
        "load_dataset_registry",
        lambda **kwargs: {"msp-podcast": entry},
    )

    issues = dp.collect_dataset_registry_health_issues(settings=settings)

    assert len(issues) == 1
    assert issues[0].dataset_id == "msp-podcast"
    assert issues[0].code == "source_provenance_mismatch"
    assert "missing source pin" in issues[0].message


def test_prepare_from_registry_entry_detects_msp_mirror_artifact_path_mismatch(
    tmp_path: Path,
) -> None:
    """Registry labels/audio options should match MSP mirror manifest artifact paths."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    dataset_root.mkdir(parents=True, exist_ok=True)
    mirror_labels_path = dataset_root / "labels.csv"
    mirror_audio_dir = dataset_root / "audio"
    mirror_labels_path.write_text("FileName,emotion\nx.wav,happy\n", encoding="utf-8")
    mirror_audio_dir.mkdir(parents=True, exist_ok=True)
    mismatched_labels_path = tmp_path / "custom_labels.csv"
    mismatched_labels_path.write_text(
        "FileName,emotion\ny.wav,sad\n",
        encoding="utf-8",
    )
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/repo",
                    "revision": "rev-1",
                    "commit_sha": "abcdef1234",
                },
                "artifacts": {
                    "labels_csv": str(mirror_labels_path),
                    "audio_dir": str(mirror_audio_dir),
                },
                "stats": {"labels_written": 1},
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(mismatched_labels_path),
            "audio_base_dir": str(mirror_audio_dir),
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234",
        },
    )

    with pytest.raises(ValueError, match="labels_csv_path .* does not match mirror"):
        dp.prepare_from_registry_entry(
            settings=settings,
            entry=entry,
            ontology=_ontology(),
        )


def test_prepare_from_registry_entry_detects_msp_mirror_labels_stats_drift(
    tmp_path: Path,
) -> None:
    """MSP manifest labels_written stats should match labels.csv row count."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    dataset_root.mkdir(parents=True, exist_ok=True)
    labels_csv_path = dataset_root / "labels.csv"
    audio_dir = dataset_root / "audio"
    labels_csv_path.write_text(
        "FileName,emotion\nx.wav,happy\ny.wav,sad\n",
        encoding="utf-8",
    )
    audio_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/repo",
                    "revision": "rev-1",
                    "commit_sha": "abcdef1234",
                },
                "artifacts": {
                    "labels_csv": str(labels_csv_path),
                    "audio_dir": str(audio_dir),
                },
                "stats": {"labels_written": 1},
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(labels_csv_path),
            "audio_base_dir": str(audio_dir),
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234",
        },
    )

    with pytest.raises(ValueError, match="row count .* does not match"):
        dp.prepare_from_registry_entry(
            settings=settings,
            entry=entry,
            ontology=_ontology(),
        )


def test_collect_registry_health_reports_msp_artifact_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry health should report MSP mirror artifact mismatch issues."""
    settings = _settings(tmp_path)
    dataset_root = tmp_path / "datasets" / "msp-podcast"
    dataset_root.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifests" / "msp.jsonl"
    labels_csv_path = dataset_root / "labels.csv"
    labels_csv_path.write_text("FileName,emotion\nx.wav,happy\n", encoding="utf-8")
    (dataset_root / "manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "repo_id": "org/repo",
                    "revision": "rev-1",
                    "commit_sha": "abcdef1234",
                },
                "artifacts": {
                    "labels_csv": str(labels_csv_path),
                    "audio_dir": str(dataset_root / "audio"),
                },
                "stats": {"labels_written": 1},
            }
        ),
        encoding="utf-8",
    )
    entry = DatasetRegistryEntry(
        dataset_id="msp-podcast",
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options={
            "labels_csv_path": str(labels_csv_path),
            "audio_base_dir": str(tmp_path / "missing-audio-dir"),
            "source_repo_id": "org/repo",
            "source_revision": "rev-1",
            "source_commit_sha": "abcdef1234",
        },
    )
    monkeypatch.setattr(
        dp,
        "load_dataset_registry",
        lambda **kwargs: {"msp-podcast": entry},
    )

    issues = dp.collect_dataset_registry_health_issues(settings=settings)

    assert len(issues) == 1
    assert issues[0].dataset_id == "msp-podcast"
    assert issues[0].code == "mirror_artifact_mismatch"
    assert "audio_base_dir" in issues[0].message
