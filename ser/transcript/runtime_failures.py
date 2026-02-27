"""Failure classification utilities for transcription runtime retries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureDisposition(StrEnum):
    """Action to take after one runtime transcription failure."""

    RETRY_NEXT_PRECISION = "retry_next_precision"
    FAILOVER_CPU_NOW = "failover_cpu_now"
    FAIL_FAST = "fail_fast"


@dataclass(frozen=True, slots=True)
class TranscriptionFailureClassification:
    """Outcome of classifying one transcription failure."""

    disposition: FailureDisposition
    reason_code: str
    is_retryable: bool


def classify_stable_whisper_transcription_failure(
    *,
    err: Exception,
    runtime_device_type: str,
    precision: str,
) -> TranscriptionFailureClassification:
    """Classifies stable-whisper transcription failures into retry/failover actions."""
    message = " ".join(str(err).split()).lower()
    is_retryable = _is_retryable_runtime_failure(message=message, err=err)
    if not is_retryable:
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.FAIL_FAST,
            reason_code="non_retryable_runtime_error",
            is_retryable=False,
        )
    if (
        runtime_device_type == "mps"
        and precision == "float16"
        and _is_pure_mps_oom(message)
    ):
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.FAILOVER_CPU_NOW,
            reason_code="mps_oom_hard_float16",
            is_retryable=True,
        )
    return TranscriptionFailureClassification(
        disposition=FailureDisposition.RETRY_NEXT_PRECISION,
        reason_code="retryable_runtime_failure",
        is_retryable=True,
    )


def _is_retryable_runtime_failure(*, message: str, err: Exception) -> bool:
    """Returns whether one runtime failure is likely to succeed with fallback."""
    if isinstance(err, NotImplementedError):
        not_implemented_markers = (
            "sparsemps",
            "could not run",
            "aten::",
            "backend",
            "mps",
        )
        if all(marker in message for marker in not_implemented_markers):
            return True
        if "std_mean" in message and "mps" in message:
            return True
    retryable_markers = (
        "out of memory",
        "mps backend out of memory",
        "fp16 is not supported on cpu",
        "fp16 is not supported",
        "half precision is not supported",
        "bfloat16 is not supported",
        "unsupported dtype",
        "sparsemps",
        "aten::empty.memory_format",
        "cannot convert a mps tensor to float64 dtype",
        "std_mean.correction",
    )
    return any(marker in message for marker in retryable_markers)


def _is_pure_mps_oom(message: str) -> bool:
    """Returns whether one failure is a hard MPS OOM, not an operator/type gap."""
    if "out of memory" not in message or "mps" not in message:
        return False
    compatibility_gap_markers = (
        "sparsemps",
        "aten::empty.memory_format",
        "std_mean",
        "unsupported dtype",
        "cannot convert a mps tensor to float64 dtype",
        "not currently implemented",
        "could not run",
    )
    return not any(marker in message for marker in compatibility_gap_markers)
