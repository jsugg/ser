"""Tests for backend-aware feature runtime selector policy."""

from __future__ import annotations

import pytest

import ser.repr.runtime_policy as runtime_policy
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.utils.torch_inference import TorchRuntime


def _runtime(*, device_spec: str, device_type: str) -> TorchRuntime:
    """Builds one torch runtime stub for selector-policy tests."""
    return TorchRuntime(
        device=object(),
        dtype=object(),
        device_spec=device_spec,
        device_type=device_type,
        dtype_name="float32",
    )


def test_feature_runtime_policy_demotes_xlsr_mps_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XLS-R should avoid MPS runtime by default for stability."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="mps", device_type="mps"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="hf_xlsr",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert resolved.device == "cpu"
    assert resolved.dtype == "float32"
    assert resolved.reason == "hf_xlsr_mps_stability_guard"


def test_feature_runtime_policy_preserves_whisper_requested_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whisper backend should keep caller-provided selectors."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="mps", device_type="mps"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="hf_whisper",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert resolved.device == "auto"
    assert resolved.dtype == "auto"
    assert resolved.reason == "torch_backend_requested_selectors"


def test_feature_runtime_policy_enables_emotion2vec_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emotion2Vec should enable CUDA runtime only when probe resolves CUDA."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="cuda:1", device_type="cuda"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="emotion2vec",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert resolved.device == "cuda:1"
    assert resolved.dtype == "auto"
    assert resolved.reason == "emotion2vec_cuda_enabled"


def test_feature_runtime_policy_defaults_emotion2vec_to_cpu_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emotion2Vec should remain CPU-only when runtime probe is non-CUDA."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="mps", device_type="mps"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="emotion2vec",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert resolved.device == "cpu"
    assert resolved.dtype == "float32"
    assert resolved.reason == "emotion2vec_cpu_default"


def test_feature_runtime_policy_applies_backend_override_for_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend-level overrides should replace requested selectors for Whisper."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="cuda:0", device_type="cuda"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="hf_whisper",
        requested_device="auto",
        requested_dtype="auto",
        backend_override_device="cuda:0",
        backend_override_dtype="float16",
    )

    assert resolved.device == "cuda:0"
    assert resolved.dtype == "float16"
    assert resolved.reason == "torch_backend_requested_selectors"


def test_feature_runtime_policy_keeps_xlsr_mps_guard_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """XLS-R stability guard should still demote MPS even when overrides request MPS."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: _runtime(device_spec="mps", device_type="mps"),
    )

    resolved = resolve_feature_runtime_policy(
        backend_id="hf_xlsr",
        requested_device="cpu",
        requested_dtype="float32",
        backend_override_device="mps",
        backend_override_dtype="float16",
    )

    assert resolved.device == "cpu"
    assert resolved.dtype == "float32"
    assert resolved.reason == "hf_xlsr_mps_stability_guard"
