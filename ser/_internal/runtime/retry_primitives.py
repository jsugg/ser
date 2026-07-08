"""Shared retry/error classification primitives for runtime execution paths."""

from __future__ import annotations

import random


def jittered_retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Return bounded retry delay using linear backoff with small jitter."""
    if base_delay <= 0.0:
        return 0.0
    jitter = random.uniform(0.0, base_delay * 0.1)
    return (base_delay * float(attempt)) + jitter


def is_optional_dependency_runtime_error(err: RuntimeError) -> bool:
    """Return whether one runtime error indicates missing optional dependencies."""
    message = str(err).lower()
    return "requires optional dependencies" in message or "not installed" in message


__all__ = [
    "is_optional_dependency_runtime_error",
    "jittered_retry_delay_seconds",
]
