"""Tests for runtime profile capability registry."""

from collections.abc import Generator

import pytest

import ser.config as config
import ser.runtime.registry as registry
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
    assert capability.required_modules == ()
    assert capability.missing_modules == ()
    assert capability.implementation_ready is True
    assert capability.message is None


def test_resolve_runtime_capability_marks_medium_unavailable_when_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium profile should report missing optional dependencies when absent."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: ("torch", "transformers"),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "medium"
    assert capability.backend_id == "hf_xlsr"
    assert capability.available is False
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ("torch", "transformers")
    assert capability.implementation_ready is False
    assert capability.message is not None
    assert "optional dependencies not currently available" in capability.message
    assert "SER_ENABLE_MEDIUM_PROFILE" in capability.message


def test_resolve_runtime_capability_marks_medium_unavailable_when_not_implemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium profile should report implementation gap when deps are present."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "medium"
    assert capability.backend_id == "hf_xlsr"
    assert capability.available is False
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is False
    assert capability.message is not None
    assert "dependencies are available" in capability.message
    assert "not implemented yet" in capability.message


def test_resolve_runtime_capability_marks_medium_available_when_hook_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium profile should become available when hook and deps are present."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(
        config.reload_settings(),
        available_backend_hooks=frozenset({"handcrafted", "hf_xlsr"}),
    )
    assert capability.profile == "medium"
    assert capability.backend_id == "hf_xlsr"
    assert capability.available is True
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is True
    assert capability.message is None


def test_resolve_runtime_capability_marks_accurate_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate profile should be present but marked unavailable until implemented."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "accurate"
    assert capability.backend_id == "hf_whisper"
    assert capability.available is False
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is False
    assert capability.message is not None


def test_resolve_runtime_capability_marks_accurate_unavailable_when_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate profile should report missing dependencies when unavailable."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: ("torch", "transformers"),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "accurate"
    assert capability.backend_id == "hf_whisper"
    assert capability.available is False
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ("torch", "transformers")
    assert capability.implementation_ready is False
    assert capability.message is not None
    assert "optional dependencies not currently available" in capability.message
    assert "SER_ENABLE_ACCURATE_PROFILE" in capability.message


def test_resolve_runtime_capability_marks_accurate_available_when_hook_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate profile should become available when hook and deps are present."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(
        config.reload_settings(),
        available_backend_hooks=frozenset({"handcrafted", "hf_whisper"}),
    )
    assert capability.profile == "accurate"
    assert capability.backend_id == "hf_whisper"
    assert capability.available is True
    assert capability.required_modules == ("torch", "transformers")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is True
    assert capability.message is None


def test_resolve_runtime_capability_marks_accurate_research_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate-research profile should fail closed when hook is missing."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    assert capability.profile == "accurate-research"
    assert capability.backend_id == "emotion2vec"
    assert capability.available is False
    assert capability.required_modules == ("torch", "funasr", "modelscope")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is False
    assert capability.message is not None
    assert "SER_ENABLE_ACCURATE_RESEARCH_PROFILE" in capability.message


def test_resolve_runtime_capability_marks_accurate_research_available_when_hook_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate-research profile should become available when hook and deps are present."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    capability = resolve_runtime_capability(
        config.reload_settings(),
        available_backend_hooks=frozenset({"handcrafted", "emotion2vec"}),
    )
    assert capability.profile == "accurate-research"
    assert capability.backend_id == "emotion2vec"
    assert capability.available is True
    assert capability.required_modules == ("torch", "funasr", "modelscope")
    assert capability.missing_modules == ()
    assert capability.implementation_ready is True
    assert capability.message is None


def test_ensure_profile_supported_raises_for_unavailable_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported profile should fail with actionable error message."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        registry,
        "_missing_optional_modules",
        lambda _required_modules: ("transformers",),
    )
    capability = resolve_runtime_capability(config.reload_settings())
    with pytest.raises(UnsupportedProfileError, match="transformers"):
        ensure_profile_supported(capability)
