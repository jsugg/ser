"""Tests for runtime failure classification in transcription adapters."""

from __future__ import annotations

from ser.transcript.runtime_failures import (
    FailureDisposition,
    classify_stable_whisper_transcription_failure,
)


def test_classifier_marks_pure_mps_oom_on_fp16_for_direct_cpu_failover() -> None:
    """Pure MPS OOM on fp16 should shortcut to CPU failover."""
    classification = classify_stable_whisper_transcription_failure(
        err=RuntimeError("MPS backend out of memory"),
        runtime_device_type="mps",
        precision="float16",
    )

    assert classification.disposition == FailureDisposition.FAILOVER_CPU_NOW
    assert classification.reason_code == "mps_oom_hard_float16"
    assert classification.is_retryable is True


def test_classifier_preserves_retry_path_for_operator_gaps() -> None:
    """SparseMPS/aten operator gaps should keep precision fallback behavior."""
    classification = classify_stable_whisper_transcription_failure(
        err=NotImplementedError(
            "Could not run 'aten::empty.memory_format' with arguments "
            "from the 'SparseMPS' backend."
        ),
        runtime_device_type="mps",
        precision="float16",
    )

    assert classification.disposition == FailureDisposition.RETRY_NEXT_PRECISION
    assert classification.is_retryable is True


def test_classifier_fails_fast_for_non_retryable_errors() -> None:
    """Non-retryable runtime failures should terminate without fallback retries."""
    classification = classify_stable_whisper_transcription_failure(
        err=RuntimeError("unexpected serialization corruption"),
        runtime_device_type="cpu",
        precision="float32",
    )

    assert classification.disposition == FailureDisposition.FAIL_FAST
    assert classification.reason_code == "non_retryable_runtime_error"
    assert classification.is_retryable is False
