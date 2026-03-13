"""Tests for medium process-operation adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import ser.config as config
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_process_operation import (
    prepare_process_operation,
    run_process_operation,
)
from ser.runtime.medium_worker_operation import PreparedMediumOperation

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _LoadedModelStub:
    artifact_metadata: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class _RuntimePolicyStub:
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    request: InferenceRequest
    settings: config.AppConfig
    expected_backend_model_id: str


def test_prepare_process_operation_builds_runtime_payload_and_warms_backend() -> None:
    """Process adapter should preserve medium worker setup contracts."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "backend_model_id": settings.models.medium_model_id,
        }
    )
    backend = object()
    calls: dict[str, object] = {}

    prepared = prepare_process_operation(
        payload,
        load_medium_model=lambda _settings, model_id: (
            calls.update({"loaded_model_id": model_id}) or loaded_model
        ),
        ensure_medium_compatible_model=lambda candidate, model_id: calls.update(
            {"validated": (candidate, model_id)}
        ),
        resolve_runtime_policy=lambda _settings: _RuntimePolicyStub(
            device="mps",
            dtype="float16",
        ),
        warn_on_runtime_selector_mismatch=lambda _loaded_model, device, dtype: (
            calls.update({"selectors": (device, dtype)})
        ),
        read_audio_file=lambda _file_path: (np.ones(8, dtype=np.float32), 16_000),
        build_medium_backend=lambda _settings, model_id, device, dtype: (
            calls.update({"backend_inputs": (model_id, device, dtype)}) or backend
        ),
        prepare_medium_backend_runtime=lambda candidate: calls.update(
            {"prepared_backend": candidate}
        ),
        model_unavailable_error_factory=FileNotFoundError,
        model_load_error_factory=RuntimeError,
    )

    assert isinstance(prepared, PreparedMediumOperation)
    assert prepared.loaded_model is loaded_model
    assert prepared.backend is backend
    assert prepared.sample_rate == 16_000
    assert prepared.audio.dtype == np.float32
    assert calls["loaded_model_id"] == settings.models.medium_model_id
    assert calls["validated"] == (loaded_model, settings.models.medium_model_id)
    assert calls["selectors"] == ("mps", "float16")
    assert calls["backend_inputs"] == (
        settings.models.medium_model_id,
        "mps",
        "float16",
    )
    assert calls["prepared_backend"] is backend


def test_prepare_process_operation_maps_missing_model_error() -> None:
    """Missing medium model should map to unavailable error contract."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )

    with pytest.raises(FileNotFoundError, match="missing medium model"):
        _ = prepare_process_operation(
            payload,
            load_medium_model=lambda _settings, _model_id: (_ for _ in ()).throw(
                FileNotFoundError("missing medium model")
            ),
            ensure_medium_compatible_model=lambda _loaded_model, _model_id: None,
            resolve_runtime_policy=lambda _settings: _RuntimePolicyStub(
                device="cpu",
                dtype="float32",
            ),
            warn_on_runtime_selector_mismatch=lambda _loaded_model, _device, _dtype: (None),
            read_audio_file=lambda _file_path: (np.ones(8, dtype=np.float32), 16_000),
            build_medium_backend=lambda _settings, _model_id, _device, _dtype: object(),
            prepare_medium_backend_runtime=lambda _backend: None,
            model_unavailable_error_factory=FileNotFoundError,
            model_load_error_factory=RuntimeError,
        )


def test_run_process_operation_delegates_compute_phase() -> None:
    """Process adapter should delegate the compute phase with prepared payload."""
    settings = config.reload_settings()
    prepared = PreparedMediumOperation(
        loaded_model=object(),
        backend=object(),
        audio=np.ones(8, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=settings.medium_runtime,
    )
    expected = object()

    result = run_process_operation(
        prepared,
        run_medium_inference_once=lambda _loaded_model, _backend, _audio, _sample_rate, _runtime_config: (
            expected
        ),
    )

    assert result is expected
