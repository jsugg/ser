"""Unit tests for stable-whisper admission/compatibility helper delegation."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import ser.transcript.backends.stable_whisper_admission_runtime as admission_runtime
from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.mps_admission import MpsAdmissionDecision


def test_is_retryable_precision_failure_detects_sparsemps_notimplemented() -> None:
    """SparseMPS NotImplementedError should be treated as retryable."""
    err = NotImplementedError(
        "Could not run aten::foo because backend mps sparsemps path is unavailable."
    )
    assert admission_runtime.is_retryable_precision_failure(err) is True


def test_mps_compatibility_fallback_reason_uses_compatibility_reason() -> None:
    """Known compatibility activation errors should map to compatibility fallback."""
    err = RuntimeError("Loaded stable-whisper model does not expose a callable to().")
    assert (
        admission_runtime.mps_compatibility_fallback_reason(err)
        == "compatibility_activation_unavailable"
    )


def test_resolve_stable_whisper_mps_admission_decision_delegates_inputs(
    monkeypatch: Any,
) -> None:
    """Admission helper should delegate all arguments to shared resolver."""
    runtime_request = BackendRuntimeRequest(
        model_name="large-v3",
        use_demucs=False,
        use_vad=False,
        device_spec="mps",
        device_type="mps",
    )
    settings = cast(Any, SimpleNamespace(transcription=SimpleNamespace()))
    logger = logging.getLogger("tests.stable_whisper_admission_runtime")
    heuristic_resolver = lambda **_: MpsAdmissionDecision(  # noqa: E731
        allow_mps=True,
        reason_code="mps_headroom_sufficient",
        required_bytes=1,
        available_bytes=2,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )
    sentinel = MpsAdmissionDecision(
        allow_mps=False,
        reason_code="mps_headroom_below_required_budget",
        required_bytes=2,
        available_bytes=1,
        required_metric="headroom_budget",
        available_metric="headroom",
        confidence="high",
    )
    captured: dict[str, object] = {}

    def _fake_resolve(
        *,
        settings: object,
        runtime_request: BackendRuntimeRequest,
        phase: str,
        logger: logging.Logger,
        heuristic_resolver: object | None = None,
    ) -> MpsAdmissionDecision:
        captured["settings"] = settings
        captured["runtime_request"] = runtime_request
        captured["phase"] = phase
        captured["logger"] = logger
        captured["heuristic_resolver"] = heuristic_resolver
        return sentinel

    monkeypatch.setattr(
        admission_runtime,
        "resolve_mps_admission_decision",
        _fake_resolve,
    )

    resolved = admission_runtime.resolve_stable_whisper_mps_admission_decision(
        settings=settings,
        runtime_request=runtime_request,
        phase="transcribe",
        logger=logger,
        heuristic_resolver=heuristic_resolver,
    )

    assert resolved is sentinel
    assert captured["settings"] is settings
    assert captured["runtime_request"] == runtime_request
    assert captured["phase"] == "transcribe"
    assert captured["logger"] is logger
    assert captured["heuristic_resolver"] is heuristic_resolver


def test_should_enforce_stable_whisper_transcribe_admission_passthrough() -> None:
    """Helper wrapper should preserve transcribe admission enforcement rules."""
    enforced = MpsAdmissionDecision(
        allow_mps=False,
        reason_code="mps_headroom_unknown_large_model",
        required_bytes=1,
        available_bytes=0,
        required_metric="model_footprint_estimate",
        available_metric="driver_allocated",
        confidence="low",
    )
    advisory = MpsAdmissionDecision(
        allow_mps=False,
        reason_code="mps_headroom_estimate_below_required_budget",
        required_bytes=1,
        available_bytes=0,
        required_metric="headroom_budget",
        available_metric="headroom_estimate",
        confidence="low",
    )

    assert (
        admission_runtime.should_enforce_stable_whisper_transcribe_admission(enforced)
        is True
    )
    assert (
        admission_runtime.should_enforce_stable_whisper_transcribe_admission(advisory)
        is False
    )
