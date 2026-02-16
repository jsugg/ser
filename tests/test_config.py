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
    monkeypatch.setattr(config.os, "cpu_count", lambda: 4)

    settings = config.reload_settings()

    assert settings.dataset.folder == Path("custom/dataset")
    assert settings.default_language == "es"
    assert settings.models.num_cores == 4


def test_reload_settings_uses_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When cpu_count is unavailable, configuration should default to one core."""
    monkeypatch.setattr(config.os, "cpu_count", lambda: None)

    settings = config.reload_settings()

    assert settings.models.num_cores == 1
