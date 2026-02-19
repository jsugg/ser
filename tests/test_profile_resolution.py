"""Tests for runtime profile resolution from rollout flags."""

from collections.abc import Generator

import pytest

import ser.config as config
from ser.profiles import available_profiles, resolve_profile, resolve_profile_name


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_available_profiles_have_expected_identifiers() -> None:
    """Profile map should expose fast/medium/accurate keys."""
    profiles = available_profiles()
    assert set(profiles) == {"fast", "medium", "accurate"}


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
