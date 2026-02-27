"""Tests for accurate-research runtime wrapper behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.repr import FeatureBackend
from ser.runtime.accurate_research_inference import run_accurate_research_inference
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


def test_run_accurate_research_inference_uses_configured_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emotion2Vec backend should initialize with configured research model id."""
    settings = SimpleNamespace(
        models=SimpleNamespace(
            accurate_research_model_id="unit-test/e2v",
            modelscope_cache_root=Path("/tmp/ser-modelscope-cache"),
            huggingface_cache_root=Path("/tmp/ser-hf-cache"),
        ),
        torch_runtime=SimpleNamespace(device="auto", dtype="auto"),
        feature_runtime_policy=SimpleNamespace(for_backend=lambda _backend_id: None),
    )
    captured: dict[str, object] = {}

    class _BackendStub:
        def __init__(
            self,
            *,
            model_id: str,
            device: str,
            modelscope_cache_root: Path,
            huggingface_cache_root: Path,
        ) -> None:
            captured["model_id"] = model_id
            captured["device"] = device
            captured["modelscope_cache_root"] = modelscope_cache_root
            captured["huggingface_cache_root"] = huggingface_cache_root

    monkeypatch.setattr(
        "ser.runtime.accurate_research_inference.Emotion2VecBackend",
        _BackendStub,
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_research_inference.resolve_feature_runtime_policy",
        lambda **_kwargs: SimpleNamespace(
            device="cpu",
            dtype="float32",
            reason="test_policy",
        ),
    )

    def _fake_run(
        request: InferenceRequest,
        settings: object,
        *,
        loaded_model: object | None = None,
        backend: object | None = None,
        enforce_timeout: bool = True,
        allow_retries: bool = True,
        expected_backend_id: str = "",
        expected_profile: str = "",
        expected_backend_model_id: str | None = None,
    ) -> InferenceResult:
        del request, settings, loaded_model, enforce_timeout, allow_retries
        captured["backend"] = backend
        captured["expected_backend_id"] = expected_backend_id
        captured["expected_profile"] = expected_profile
        captured["expected_backend_model_id"] = expected_backend_model_id
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
        )

    monkeypatch.setattr(
        "ser.runtime.accurate_research_inference.run_accurate_inference",
        _fake_run,
    )

    result = run_accurate_research_inference(
        InferenceRequest(file_path="sample.wav", language="en"),
        cast(AppConfig, settings),
    )

    assert result.schema_version == OUTPUT_SCHEMA_VERSION
    assert captured["model_id"] == "unit-test/e2v"
    assert captured["device"] == "cpu"
    assert captured["modelscope_cache_root"] == Path("/tmp/ser-modelscope-cache")
    assert captured["huggingface_cache_root"] == Path("/tmp/ser-hf-cache")
    assert captured["backend"] is not None
    assert captured["expected_backend_id"] == "emotion2vec"
    assert captured["expected_profile"] == "accurate-research"
    assert captured["expected_backend_model_id"] == "unit-test/e2v"


def test_run_accurate_research_inference_uses_injected_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected backend should be forwarded unchanged to shared accurate runner."""
    settings = SimpleNamespace(
        models=SimpleNamespace(accurate_research_model_id="ignored"),
        feature_runtime_policy=SimpleNamespace(for_backend=lambda _backend_id: None),
    )
    backend = object()
    captured: dict[str, object] = {}

    def _fake_run(
        request: InferenceRequest,
        settings: object,
        *,
        loaded_model: object | None = None,
        backend: object | None = None,
        enforce_timeout: bool = True,
        allow_retries: bool = True,
        expected_backend_id: str = "",
        expected_profile: str = "",
        expected_backend_model_id: str | None = None,
    ) -> InferenceResult:
        del request, settings, loaded_model, enforce_timeout, allow_retries
        captured["backend"] = backend
        captured["expected_backend_id"] = expected_backend_id
        captured["expected_profile"] = expected_profile
        captured["expected_backend_model_id"] = expected_backend_model_id
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
        )

    monkeypatch.setattr(
        "ser.runtime.accurate_research_inference.run_accurate_inference",
        _fake_run,
    )

    result = run_accurate_research_inference(
        InferenceRequest(file_path="sample.wav", language="en"),
        cast(AppConfig, settings),
        backend=cast(FeatureBackend, backend),
    )

    assert result.schema_version == OUTPUT_SCHEMA_VERSION
    assert captured["backend"] is backend
    assert captured["expected_backend_id"] == "emotion2vec"
    assert captured["expected_profile"] == "accurate-research"
    assert captured["expected_backend_model_id"] == "ignored"
