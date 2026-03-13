"""Tests for accurate operation-setup adapters."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.runtime import accurate_operation_setup
from ser.runtime.accurate_worker_operation import PreparedAccurateOperation

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _RequestStub:
    file_path: str = "sample.wav"


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    request: _RequestStub
    settings: AppConfig
    expected_backend_id: str
    expected_profile: str
    expected_backend_model_id: str | None


def test_prepare_in_process_operation_delegates_to_worker_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner should delegate in-process setup to accurate worker helpers."""
    captured: dict[str, object] = {}
    expected = object()
    settings = cast(AppConfig, SimpleNamespace())
    runtime_config = cast(ProfileRuntimeConfig, SimpleNamespace())
    request = _RequestStub()

    monkeypatch.setattr(
        accurate_operation_setup.accurate_worker_operation_helpers,
        "prepare_in_process_operation",
        lambda **kwargs: captured.update(kwargs) or expected,
    )

    result = accurate_operation_setup.prepare_in_process_operation(
        request=request,
        settings=settings,
        runtime_config=runtime_config,
        loaded_model=None,
        backend=None,
        load_accurate_model=lambda active_settings: object(),
        validate_loaded_model=lambda active_loaded_model: None,
        read_audio_file=lambda _path: (np.ones(4, dtype=np.float32), 16_000),
        build_backend_for_profile=lambda active_settings: object(),
        model_unavailable_error_factory=RuntimeError,
        model_load_error_factory=ValueError,
    )

    assert result is expected
    assert captured["request"] is request
    assert captured["settings"] is settings
    assert captured["runtime_config"] is runtime_config
    assert captured["model_unavailable_error_factory"] is RuntimeError


def test_prepare_process_operation_delegates_to_worker_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner should delegate process setup to accurate worker helpers."""
    captured: dict[str, object] = {}
    expected = object()
    settings = cast(AppConfig, SimpleNamespace())
    payload = _PayloadStub(
        request=_RequestStub(),
        settings=settings,
        expected_backend_id="hf_whisper",
        expected_profile="accurate",
        expected_backend_model_id="unit-test/whisper",
    )

    monkeypatch.setattr(
        accurate_operation_setup.accurate_worker_operation_helpers,
        "prepare_process_operation",
        lambda **kwargs: captured.update(kwargs) or expected,
    )

    result = accurate_operation_setup.prepare_process_operation(
        payload,
        resolve_runtime_config=lambda active_settings, expected_profile: cast(
            ProfileRuntimeConfig, SimpleNamespace()
        ),
        load_accurate_model=lambda active_payload: object(),
        validate_loaded_model=lambda loaded_model, active_payload: None,
        read_audio_file=lambda _path: (np.ones(4, dtype=np.float32), 16_000),
        build_backend_for_payload=lambda active_payload: object(),
        prepare_accurate_backend_runtime=lambda backend: None,
        model_unavailable_error_factory=RuntimeError,
        model_load_error_factory=ValueError,
    )

    assert result is expected
    assert captured["payload"] is payload
    assert captured["model_unavailable_error_factory"] is RuntimeError
    assert captured["model_load_error_factory"] is ValueError


def test_run_process_operation_delegates_compute_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner should delegate isolated compute phase to accurate worker helpers."""
    captured: dict[str, object] = {}
    expected = object()
    prepared = PreparedAccurateOperation(
        loaded_model=object(),
        backend=object(),
        audio=np.ones(4, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=cast(ProfileRuntimeConfig, SimpleNamespace()),
    )

    monkeypatch.setattr(
        accurate_operation_setup.accurate_worker_operation_helpers,
        "run_process_operation",
        lambda active_prepared, **kwargs: captured.update({"prepared": active_prepared, **kwargs})
        or expected,
    )

    result = accurate_operation_setup.run_process_operation(
        prepared=prepared,
        run_accurate_inference_once=lambda *_args: expected,
    )

    assert result is expected
    assert captured["prepared"] is prepared
    assert callable(captured["run_accurate_inference_once"])
