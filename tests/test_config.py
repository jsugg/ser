"""Tests for typed configuration loading and environment refresh."""

from collections.abc import Generator
from pathlib import Path

import pytest

import ser.config as config


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_reload_settings_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should be reflected in loaded settings."""
    monkeypatch.setenv("DATASET_FOLDER", "custom/dataset")
    monkeypatch.setenv("DEFAULT_LANGUAGE", "es")
    monkeypatch.setenv("SER_CACHE_DIR", "custom/cache")
    monkeypatch.setenv("SER_DATA_DIR", "custom/data")
    monkeypatch.setenv("SER_TMP_DIR", "custom/tmp")
    monkeypatch.setenv("SER_MODELS_DIR", "custom/models")
    monkeypatch.setenv("SER_TRANSCRIPTS_DIR", "custom/transcripts")
    monkeypatch.setenv("SER_MODEL_FILE_NAME", "custom_model.pkl")
    monkeypatch.setenv("SER_SECURE_MODEL_FILE_NAME", "custom_model.skops")
    monkeypatch.setenv("SER_TRAINING_REPORT_FILE_NAME", "custom_training_report.json")
    monkeypatch.setenv("SER_MAX_WORKERS", "6")
    monkeypatch.setenv("SER_MAX_FAILED_FILE_RATIO", "0.45")
    monkeypatch.setenv("SER_TEST_SIZE", "0.3")
    monkeypatch.setenv("SER_RANDOM_STATE", "7")
    monkeypatch.setenv("WHISPER_MODEL", "base")
    monkeypatch.setenv("WHISPER_DEMUCS", "false")
    monkeypatch.setenv("WHISPER_VAD", "true")
    monkeypatch.setattr(config.os, "cpu_count", lambda: 4)

    settings = config.reload_settings()

    assert settings.dataset.folder == Path("custom/dataset")
    assert settings.default_language == "es"
    assert settings.models.num_cores == 4
    assert settings.tmp_folder == Path("custom/tmp")
    assert settings.models.folder == Path("custom/models")
    assert settings.timeline.folder == Path("custom/transcripts")
    assert settings.models.model_file == Path("custom/models/custom_model.pkl")
    assert settings.models.secure_model_file == Path("custom/models/custom_model.skops")
    assert settings.models.training_report_file == Path(
        "custom/models/custom_training_report.json"
    )
    assert settings.models.whisper_model.name == "base"
    assert settings.data_loader.max_workers == 6
    assert settings.data_loader.max_failed_file_ratio == pytest.approx(0.45)
    assert settings.training.test_size == pytest.approx(0.3)
    assert settings.training.random_state == 7
    assert settings.nn.random_state == 7
    assert settings.transcription.use_demucs is False
    assert settings.transcription.use_vad is True


def test_reload_settings_uses_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cpu_count is unavailable, configuration should default to one core."""
    monkeypatch.setattr(config.os, "cpu_count", lambda: None)

    settings = config.reload_settings()

    assert settings.models.num_cores == 1
