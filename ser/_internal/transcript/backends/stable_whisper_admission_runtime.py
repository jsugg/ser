"""Stable-whisper admission and compatibility helper functions."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from ser.transcript.mps_admission import (
    MpsAdmissionDecision,
    resolve_mps_admission_decision,
    should_enforce_transcribe_admission,
)

if TYPE_CHECKING:
    from ser.config import AppConfig
    from ser.transcript.backends.base import BackendRuntimeRequest

type StableWhisperTranscriptionPhase = Literal["model_load", "transcribe"]
type ExceptionPredicate = Callable[[Exception], bool]
type MpsAdmissionHeuristicResolver = Callable[..., MpsAdmissionDecision]


def is_retryable_precision_failure(err: Exception) -> bool:
    """Returns whether one transcription failure may succeed on fallback precision."""
    message = str(err).strip().lower()
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


def is_compatibility_activation_error(err: Exception) -> bool:
    """Returns whether one MPS compatibility activation failure is fail-safe."""
    if not isinstance(err, RuntimeError):
        return False
    message = str(err).strip().lower()
    compatibility_markers = (
        "does not expose a callable to()",
        "does not expose register_buffer() after mps move",
        "loaded stable-whisper model does not expose register_buffer",
        "loaded stable-whisper model does not expose a callable to()",
    )
    return any(marker in message for marker in compatibility_markers)


def mps_compatibility_fallback_reason(
    err: Exception,
    *,
    retryable_precision_checker: ExceptionPredicate = is_retryable_precision_failure,
    compatibility_activation_checker: ExceptionPredicate = (is_compatibility_activation_error),
) -> str | None:
    """Returns one CPU-fallback reason for MPS compatibility activation failures."""
    if retryable_precision_checker(err):
        return "retryable_runtime_error"
    if compatibility_activation_checker(err):
        return "compatibility_activation_unavailable"
    return None


def resolve_stable_whisper_mps_admission_decision(
    *,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest,
    phase: StableWhisperTranscriptionPhase,
    logger: logging.Logger,
    heuristic_resolver: MpsAdmissionHeuristicResolver | None = None,
) -> MpsAdmissionDecision | None:
    """Returns one dynamic MPS admission decision when control is enabled."""
    return resolve_mps_admission_decision(
        settings=settings,
        runtime_request=runtime_request,
        phase=phase,
        logger=logger,
        heuristic_resolver=heuristic_resolver,
    )


def should_enforce_stable_whisper_transcribe_admission(
    decision: MpsAdmissionDecision,
) -> bool:
    """Returns whether one pre-transcribe admission decision should be enforced."""
    return should_enforce_transcribe_admission(decision)
