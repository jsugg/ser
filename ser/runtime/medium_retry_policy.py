"""Retry/fallback classification helpers for medium runtime inference."""

from __future__ import annotations

from ser.runtime.retry_primitives import (
    is_optional_dependency_runtime_error as _is_optional_dependency_runtime_error_impl,
)
from ser.runtime.retry_primitives import (
    jittered_retry_delay_seconds as _jittered_retry_delay_seconds_impl,
)

_MPS_OOM_SIGNATURE = "mps backend out of memory"
_NON_FINITE_SIGNATURE = "non-finite embeddings"
_HALF_FLOAT_MISMATCH_SIGNATURE = "input type (c10::half) and bias type (float)"


def retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    return _jittered_retry_delay_seconds_impl(
        base_delay=base_delay,
        attempt=attempt,
    )


def should_retry_on_cpu_after_transient_failure(err: Exception) -> bool:
    """Returns whether one transient failure should trigger CPU fallback retry."""
    message = str(err).strip().lower()
    if _MPS_OOM_SIGNATURE in message:
        return True
    if _HALF_FLOAT_MISMATCH_SIGNATURE in message:
        return True
    if _NON_FINITE_SIGNATURE in message:
        return True
    return False


def summarize_transient_failure(err: Exception) -> str:
    """Builds one compact summary line for medium transient fallback logs."""
    message = str(err).strip()
    if not message:
        return "transient backend failure"
    collapsed = " ".join(message.split())
    return collapsed if len(collapsed) <= 180 else f"{collapsed[:177]}..."


def is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    return _is_optional_dependency_runtime_error_impl(err)


__all__ = [
    "is_dependency_error",
    "retry_delay_seconds",
    "should_retry_on_cpu_after_transient_failure",
    "summarize_transient_failure",
]
