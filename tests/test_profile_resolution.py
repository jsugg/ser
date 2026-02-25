"""Tests for runtime profile resolution from rollout flags."""

from collections.abc import Generator

import pytest

import ser.config as config
from ser.profiles import (
    available_profiles,
    get_profile_catalog,
    resolve_profile,
    resolve_profile_name,
)


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_available_profiles_have_expected_identifiers() -> None:
    """Profile map should expose all supported runtime profile identifiers."""
    profiles = available_profiles()
    assert set(profiles) == {"fast", "medium", "accurate", "accurate-research"}


def test_profile_catalog_contains_backend_and_flag_metadata() -> None:
    """YAML profile definitions should expose stable backend/runtime mappings."""
    catalog = get_profile_catalog()
    assert catalog["fast"].backend_id == "handcrafted"
    assert catalog["fast"].transcription_defaults.model_name == "turbo"
    assert catalog["fast"].enable_flag is None
    assert catalog["fast"].runtime_defaults.timeout_seconds == pytest.approx(0.0)
    assert catalog["fast"].runtime_defaults.max_timeout_retries == 0
    assert catalog["fast"].runtime_defaults.max_transient_retries == 0
    assert catalog["fast"].runtime_defaults.process_isolation is False
    assert catalog["fast"].runtime_env.timeout_seconds == "SER_FAST_TIMEOUT_SECONDS"
    assert catalog["medium"].backend_id == "hf_xlsr"
    assert catalog["medium"].transcription_defaults.model_name == "turbo"
    assert catalog["medium"].enable_flag == "SER_ENABLE_MEDIUM_PROFILE"
    assert catalog["accurate"].backend_id == "hf_whisper"
    assert catalog["accurate"].transcription_defaults.model_name == "large"
    assert catalog["accurate"].enable_flag == "SER_ENABLE_ACCURATE_PROFILE"
    assert catalog["accurate"].model.env_var == "SER_ACCURATE_MODEL_ID"
    assert catalog["accurate"].runtime_defaults.timeout_seconds == pytest.approx(120.0)
    assert (
        catalog["accurate"].runtime_env.max_timeout_retries
        == "SER_ACCURATE_MAX_TIMEOUT_RETRIES"
    )
    assert catalog["accurate-research"].backend_id == "emotion2vec"
    assert catalog["accurate-research"].transcription_defaults.model_name == "large"
    assert (
        catalog["accurate-research"].enable_flag
        == "SER_ENABLE_ACCURATE_RESEARCH_PROFILE"
    )
    assert "future" not in catalog["medium"].description.lower()
    assert "future" not in catalog["accurate"].description.lower()


def test_resolve_profile_defaults_to_fast() -> None:
    """Default flags should resolve to the fast profile."""
    settings = config.reload_settings()
    assert resolve_profile_name(settings) == "fast"
    assert resolve_profile(settings).name == "fast"


def test_resolve_profile_selects_medium_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium flag should resolve medium profile."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    settings = config.reload_settings()
    assert resolve_profile_name(settings) == "medium"
    assert resolve_profile(settings).name == "medium"


def test_resolve_profile_prefers_accurate_over_medium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate flag should take precedence when both are enabled."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    settings = config.reload_settings()
    assert resolve_profile_name(settings) == "accurate"
    assert resolve_profile(settings).name == "accurate"


def test_resolve_profile_prefers_accurate_research_over_accurate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate-research flag should take precedence over accurate/medium flags."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    settings = config.reload_settings()
    assert resolve_profile_name(settings) == "accurate-research"
    assert resolve_profile(settings).name == "accurate-research"
