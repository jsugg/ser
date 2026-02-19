"""Tests for runtime backend hook registry behavior."""

from __future__ import annotations

from collections.abc import Generator

import pytest

import ser.config as config
import ser.runtime.backend_hooks as backend_hooks
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_build_backend_hooks_returns_empty_when_medium_flag_is_disabled() -> None:
    """Hook registry should stay empty when medium profile flag is off."""
    settings = config.reload_settings()
    hooks = backend_hooks.build_backend_hooks(settings)
    assert hooks == {}


def test_build_medium_hook_returns_none_when_dependencies_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium hook should not register when required modules are absent."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        backend_hooks,
        "_missing_optional_modules",
        lambda _required_modules: ("transformers",),
    )
    settings = config.reload_settings()
    assert backend_hooks._build_medium_hook(settings) is None


def test_build_medium_hook_returns_none_when_runner_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium hook should not register until concrete runner implementation exists."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        backend_hooks,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )
    monkeypatch.setattr(
        backend_hooks,
        "_load_medium_inference_runner",
        lambda: None,
    )
    settings = config.reload_settings()
    assert backend_hooks._build_medium_hook(settings) is None
    assert backend_hooks.build_backend_hooks(settings) == {}


def test_build_backend_hooks_registers_medium_when_runner_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium hook should register when flag/deps/runner are all available."""
    monkeypatch.setenv("SER_ENABLE_MEDIUM_PROFILE", "true")
    monkeypatch.setattr(
        backend_hooks,
        "_missing_optional_modules",
        lambda _required_modules: (),
    )

    def fake_runner(
        request: InferenceRequest,
        settings: config.AppConfig,
    ) -> InferenceResult:
        assert settings.runtime_flags.medium_profile is True
        assert request.file_path == "sample.wav"
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        backend_hooks,
        "_load_medium_inference_runner",
        lambda: fake_runner,
    )
    settings = config.reload_settings()
    hooks = backend_hooks.build_backend_hooks(settings)

    assert list(hooks.keys()) == ["hf_xlsr"]
    result = hooks["hf_xlsr"](
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)
    )
    assert result.schema_version == OUTPUT_SCHEMA_VERSION
