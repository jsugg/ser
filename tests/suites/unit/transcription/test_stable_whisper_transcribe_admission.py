"""Contract tests for stable-whisper pre-transcribe admission helpers."""

from __future__ import annotations

import logging
from typing import Literal

from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.backends.stable_whisper_transcribe_admission import (
    resolve_transcribe_runtime_device,
)
from ser.transcript.mps_admission import MpsAdmissionDecision


def _runtime_request(*, device_type: str = "mps") -> BackendRuntimeRequest:
    return BackendRuntimeRequest(
        model_name="openai/whisper-large-v3",
        use_demucs=False,
        use_vad=True,
        device_spec=device_type,
        device_type=device_type,
        precision_candidates=("float16", "float32"),
        memory_tier="normal",
    )


def _decision(
    *,
    allow_mps: bool,
    reason_code: str,
    confidence: Literal["high", "medium", "low"] = "high",
) -> MpsAdmissionDecision:
    return MpsAdmissionDecision(
        allow_mps=allow_mps,
        reason_code=reason_code,
        required_bytes=1024,
        available_bytes=512,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence=confidence,
    )


def test_resolve_transcribe_runtime_device_enforces_cpu_fallback() -> None:
    """Admission denial should switch runtime device to CPU when enforceable."""
    model = object()
    captured: dict[str, object] = {}

    resolved_device = resolve_transcribe_runtime_device(
        model=model,
        runtime_request=_runtime_request(),
        settings=object(),
        runtime_device_type="mps",
        resolve_mps_admission_decision=lambda **kwargs: _decision(
            allow_mps=False,
            reason_code="mps_headroom_below_required_budget",
        ),
        should_enforce_transcribe_admission=lambda decision: not decision.allow_mps,
        log_mps_admission_control_fallback=lambda **kwargs: captured.update(
            {"fallback_logged": kwargs}
        ),
        move_model_to_cpu_runtime=lambda _model: True,
        set_mps_compatibility_enabled=lambda _model, *, enabled: captured.update(
            {"compat_enabled": enabled}
        ),
        set_runtime_device=lambda _model, *, device_type: captured.update(
            {"device_type": device_type}
        ),
        logger=logging.getLogger("tests.stable_whisper.transcribe_admission"),
    )

    assert resolved_device == "cpu"
    assert captured["compat_enabled"] is False
    assert captured["device_type"] == "cpu"
    fallback = captured["fallback_logged"]
    assert isinstance(fallback, dict)
    assert fallback["phase"] == "transcribe"


def test_resolve_transcribe_runtime_device_keeps_mps_when_not_enforced() -> None:
    """Low-confidence admission denial should keep one MPS attempt."""
    resolved_device = resolve_transcribe_runtime_device(
        model=object(),
        runtime_request=_runtime_request(),
        settings=object(),
        runtime_device_type="mps",
        resolve_mps_admission_decision=lambda **kwargs: _decision(
            allow_mps=False,
            reason_code="mps_headroom_estimate_below_required_budget",
            confidence="low",
        ),
        should_enforce_transcribe_admission=lambda _decision: False,
        log_mps_admission_control_fallback=lambda **kwargs: None,
        move_model_to_cpu_runtime=lambda _model: True,
        set_mps_compatibility_enabled=lambda _model, *, enabled: None,
        set_runtime_device=lambda _model, *, device_type: None,
        logger=logging.getLogger("tests.stable_whisper.transcribe_admission"),
    )

    assert resolved_device == "mps"


def test_resolve_transcribe_runtime_device_bypasses_non_mps_runtime() -> None:
    """Non-MPS runtime should bypass admission logic and keep current device."""
    called = {"resolver": False}

    resolved_device = resolve_transcribe_runtime_device(
        model=object(),
        runtime_request=_runtime_request(device_type="cpu"),
        settings=object(),
        runtime_device_type="cpu",
        resolve_mps_admission_decision=lambda **kwargs: (
            called.update({"resolver": True}),
            None,
        )[1],
        should_enforce_transcribe_admission=lambda _decision: True,
        log_mps_admission_control_fallback=lambda **kwargs: None,
        move_model_to_cpu_runtime=lambda _model: True,
        set_mps_compatibility_enabled=lambda _model, *, enabled: None,
        set_runtime_device=lambda _model, *, device_type: None,
        logger=logging.getLogger("tests.stable_whisper.transcribe_admission"),
    )

    assert resolved_device == "cpu"
    assert called["resolver"] is False
