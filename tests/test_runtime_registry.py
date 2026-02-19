"""Tests for runtime profile capability registry."""

from collections.abc import Generator

import pytest

import ser.config as config
from ser.runtime.registry import (
    UnsupportedProfileError,
    ensure_profile_supported,
    resolve_runtime_capability,
)


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_resolve_runtime_capability_defaults_to_fast() -> None:
    """Default profile should resolve to available handcrafted backend."""
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "fast"
    assert capability.backend_id == "handcrafted"
    assert capability.available is True
    assert capability.message is None


def test_resolve_runtime_capability_marks_medium_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium profile should be present but marked unavailable until implemented."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "medium"
    assert capability.backend_id == "hf_xlsr"
    assert capability.available is False
    assert capability.message is not None


def test_resolve_runtime_capability_marks_accurate_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate profile should be present but marked unavailable until implemented."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "accurate"
    assert capability.backend_id == "hf_whisper"
    assert capability.available is False
    assert capability.message is not None


def test_ensure_profile_supported_raises_for_unavailable_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported profile should fail with actionable error message."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    capability = resolve_runtime_capability(config.reload_settings())
    with pytest.raises(UnsupportedProfileError, match="SER_ENABLE_MEDIUM_PROFILE"):
        ensure_profile_supported(capability)
