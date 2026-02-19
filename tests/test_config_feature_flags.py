"""Tests for runtime feature-flag and schema version configuration."""

from collections.abc import Generator

import pytest

import ser.config as config


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_runtime_flags_and_schema_defaults() -> None:
    """Feature flags and schema versions should default to conservative values."""
    settings = config.reload_settings()

    assert settings.runtime_flags.profile_pipeline is False
    assert settings.runtime_flags.medium_profile is False
    assert settings.runtime_flags.accurate_profile is False
    assert settings.runtime_flags.restricted_backends is False
    assert settings.runtime_flags.new_output_schema is False
    assert settings.schema.output_schema_version == "v1"
    assert settings.schema.artifact_schema_version == "v1"


def test_runtime_flags_and_schema_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment variables should override rollout flags and schema versions."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "1")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "yes")
    monkeypatch.setenv("SER_ENABLE_RESTRICTED_BACKENDS", "on")
    monkeypatch.setenv("SER_ENABLE_NEW_OUTPUT_SCHEMA", "true")
    monkeypatch.setenv("SER_OUTPUT_SCHEMA_VERSION", "v2")
    monkeypatch.setenv("SER_ARTIFACT_SCHEMA_VERSION", "v2")

    settings = config.reload_settings()

    assert settings.runtime_flags.profile_pipeline is True
    assert settings.runtime_flags.medium_profile is True
    assert settings.runtime_flags.accurate_profile is True
    assert settings.runtime_flags.restricted_backends is True
    assert settings.runtime_flags.new_output_schema is True
    assert settings.schema.output_schema_version == "v2"
    assert settings.schema.artifact_schema_version == "v2"
