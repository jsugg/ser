"""Tests for transcription runtime policy resolution."""

from __future__ import annotations

import pytest

import ser.transcript.runtime_policy as runtime_policy
from ser.utils.torch_inference import TorchRuntime


def _runtime(*, device_spec: str, device_type: str) -> TorchRuntime:
    """Builds one deterministic torch runtime stub."""
    return TorchRuntime(
        device=object(),
        dtype=object(),
        device_spec=device_spec,
        device_type=device_type,
        dtype_name="float32",
    )


def test_policy_resolves_explicit_cuda_float16() -> None:
    """Explicit CUDA+float16 should pin precision without fallback candidates."""
    policy = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="stable_whisper",
        requested_device="cuda",
        requested_dtype="float16",
    )

    assert policy.device_type in {"cpu", "cuda"}
    if policy.device_type == "cuda":
        assert policy.precision_candidates == ("float16",)
    else:
        assert policy.precision_candidates == ("float32",)


def test_policy_prefers_fp16_then_fp32_on_low_memory_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-memory MPS auto mode should prioritize fp16 for fit-first behavior."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda **_kwargs: _runtime(device_spec="mps", device_type="mps"),
    )
    monkeypatch.setattr(
        runtime_policy,
        "_resolve_mps_memory_tier",
        lambda **_kwargs: "low",
    )

    policy = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="stable_whisper",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert policy.device_spec == "mps"
    assert policy.device_type == "mps"
    assert policy.memory_tier == "low"
    assert policy.precision_candidates == ("float16", "float32")


def test_policy_keeps_fp16_then_fp32_on_high_memory_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-memory MPS keeps fit-first order; memory tier is informational."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda **_kwargs: _runtime(device_spec="mps", device_type="mps"),
    )
    monkeypatch.setattr(
        runtime_policy,
        "_resolve_mps_memory_tier",
        lambda **_kwargs: "high",
    )

    policy = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="stable_whisper",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert policy.device_spec == "mps"
    assert policy.device_type == "mps"
    assert policy.memory_tier == "high"
    assert policy.precision_candidates == ("float16", "float32")
    assert "mps_memory_tier_high_informational_only" in policy.reason


def test_policy_filters_bfloat16_for_stable_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable-whisper should drop bfloat16 from candidate list by capability."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda **_kwargs: _runtime(device_spec="cuda:0", device_type="cuda"),
    )

    policy = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="stable_whisper",
        requested_device="cuda:0",
        requested_dtype="bfloat16",
    )

    assert policy.device_spec == "cuda:0"
    assert policy.device_type == "cuda"
    assert policy.precision_candidates == ("float16", "float32")
    assert "precision_candidates_filtered_by_backend_capabilities" in policy.reason


def test_policy_falls_back_to_cpu_for_unsupported_backend_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends lacking MPS support should resolve to CPU without hard failure."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda **_kwargs: _runtime(device_spec="mps", device_type="mps"),
    )
    monkeypatch.setattr(
        runtime_policy,
        "_resolve_mps_memory_tier",
        lambda **_kwargs: "low",
    )

    policy = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="faster_whisper",
        requested_device="auto",
        requested_dtype="auto",
    )

    assert policy.device_spec == "cpu"
    assert policy.device_type == "cpu"
    assert policy.memory_tier == "not_applicable"
    assert policy.precision_candidates == ("float32",)
    assert "backend_faster_whisper_does_not_support_device_mps" in policy.reason


def test_policy_passes_configured_mps_threshold_to_memory_tier_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy should route configured threshold through memory-tier resolution."""
    monkeypatch.setattr(
        runtime_policy,
        "maybe_resolve_torch_runtime",
        lambda **_kwargs: _runtime(device_spec="mps", device_type="mps"),
    )
    captured: dict[str, float] = {}

    def _fake_memory_tier(*, low_memory_threshold_gb: float) -> str:
        captured["threshold"] = low_memory_threshold_gb
        return "high"

    monkeypatch.setattr(runtime_policy, "_resolve_mps_memory_tier", _fake_memory_tier)

    _ = runtime_policy.resolve_transcription_runtime_policy(
        backend_id="stable_whisper",
        requested_device="mps",
        requested_dtype="auto",
        mps_low_memory_threshold_gb=28.5,
    )

    assert captured["threshold"] == pytest.approx(28.5)
