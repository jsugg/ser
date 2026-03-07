"""Pre-transcribe runtime-device admission helpers for stable-whisper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from ser.transcript.mps_admission import MpsAdmissionDecision

if TYPE_CHECKING:
    from ser.transcript.backends.base import BackendRuntimeRequest


def resolve_transcribe_runtime_device(
    *,
    model: object,
    runtime_request: BackendRuntimeRequest,
    settings: object,
    runtime_device_type: str,
    resolve_mps_admission_decision: Callable[..., MpsAdmissionDecision | None],
    should_enforce_transcribe_admission: Callable[[MpsAdmissionDecision], bool],
    log_mps_admission_control_fallback: Callable[..., None],
    move_model_to_cpu_runtime: Callable[[object], bool],
    set_mps_compatibility_enabled: Callable[..., None],
    set_runtime_device: Callable[..., None],
    logger: logging.Logger,
) -> str:
    """Resolves runtime device before transcription according to MPS admission policy."""
    if runtime_device_type != "mps":
        return runtime_device_type
    transcribe_admission = resolve_mps_admission_decision(
        settings=settings,
        runtime_request=runtime_request,
        phase="transcribe",
    )
    if transcribe_admission is None or transcribe_admission.allow_mps:
        return runtime_device_type
    if should_enforce_transcribe_admission(transcribe_admission):
        log_mps_admission_control_fallback(
            phase="transcribe",
            decision=transcribe_admission,
        )
        if move_model_to_cpu_runtime(model):
            set_mps_compatibility_enabled(model, enabled=False)
            set_runtime_device(model, device_type="cpu")
            return "cpu"
        return runtime_device_type
    logger.info(
        "MPS admission estimate below budget but confidence=%s; "
        "allowing one MPS attempt and relying on runtime fallback "
        "(reason=%s).",
        transcribe_admission.confidence,
        transcribe_admission.reason_code,
    )
    return runtime_device_type


__all__ = ["resolve_transcribe_runtime_device"]
