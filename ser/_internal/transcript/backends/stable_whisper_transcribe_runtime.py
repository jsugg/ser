"""Runtime retry/fallback helpers for stable-whisper transcription execution."""

from __future__ import annotations

import gc
import logging
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from ser.transcript.backends.base import BackendRuntimeRequest
from ser.transcript.runtime_failures import (
    FailureDisposition,
    TranscriptionFailureClassification,
    classify_stable_whisper_transcription_failure,
)

if TYPE_CHECKING:
    from ser.config import AppConfig


def classify_transcription_failure_for_runtime(
    *,
    err: Exception,
    runtime_device_type: str,
    precision: str,
    settings: AppConfig,
    hard_oom_shortcut_enabled: bool,
) -> TranscriptionFailureClassification:
    """Classifies one stable-whisper failure with hard-OOM policy adjustment."""
    classification = classify_stable_whisper_transcription_failure(
        err=err,
        runtime_device_type=runtime_device_type,
        precision=precision,
    )
    if (
        classification.disposition == FailureDisposition.FAILOVER_CPU_NOW
        and not hard_oom_shortcut_enabled
    ):
        return TranscriptionFailureClassification(
            disposition=FailureDisposition.RETRY_NEXT_PRECISION,
            reason_code=f"{classification.reason_code}_disabled",
            is_retryable=True,
        )
    return classification


def effective_precision_candidates(
    *,
    runtime_request: BackendRuntimeRequest,
    runtime_device_type: str,
) -> tuple[str, ...]:
    """Resolves one runtime precision order using actual loaded model device."""
    if runtime_device_type == "cpu":
        return ("float32",)
    return runtime_request.precision_candidates or ("float32",)


def release_torch_runtime_memory_for_retry(
    *,
    logger: logging.Logger,
) -> None:
    """Releases best-effort torch runtime caches before retry attempts."""
    gc.collect()
    torch_module = sys.modules.get("torch")
    if not isinstance(torch_module, ModuleType):
        return

    mps_module = getattr(torch_module, "mps", None)
    if isinstance(mps_module, ModuleType):
        is_available = getattr(mps_module, "is_available", None)
        empty_cache = getattr(mps_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch MPS cache before retry.",
                exc_info=True,
            )

    cuda_module = getattr(torch_module, "cuda", None)
    if isinstance(cuda_module, ModuleType):
        is_available = getattr(cuda_module, "is_available", None)
        empty_cache = getattr(cuda_module, "empty_cache", None)
        try:
            if callable(is_available) and is_available() and callable(empty_cache):
                empty_cache()
        except Exception:
            logger.debug(
                "Ignored failure while emptying torch CUDA cache before retry.",
                exc_info=True,
            )


def summarize_runtime_error(err: Exception, max_chars: int = 180) -> str:
    """Returns one single-line runtime error summary for retry logs."""
    normalized = " ".join(str(err).split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


__all__ = [
    "classify_transcription_failure_for_runtime",
    "effective_precision_candidates",
    "release_torch_runtime_memory_for_retry",
    "summarize_runtime_error",
]
