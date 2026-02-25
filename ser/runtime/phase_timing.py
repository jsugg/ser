"""Phase-oriented timing and logging helpers."""

from __future__ import annotations

import logging
from time import perf_counter

from ser.runtime.phase_contract import phase_label


def format_duration(duration_seconds: float) -> str:
    """Formats duration in a human-readable style for CLI logs."""
    total_milliseconds = max(0, int(round(duration_seconds * 1000.0)))
    if total_milliseconds <= 0:
        return "<1ms"

    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    parts: list[str] = []
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or minutes > 0:
        parts.append(f"{seconds}s")
    if milliseconds > 0:
        parts.append(f"{milliseconds}ms")
    return ", ".join(parts)


def log_phase_started(
    logger: logging.Logger,
    *,
    phase_name: str,
    profile: str | None = None,
) -> float:
    """Logs phase start and returns monotonic start timestamp."""
    del profile
    logger.info("%s started.", phase_label(phase_name))
    return perf_counter()


def log_phase_completed(
    logger: logging.Logger,
    *,
    phase_name: str,
    started_at: float,
    profile: str | None = None,
    level: int = logging.INFO,
) -> float:
    """Logs phase completion and returns elapsed duration in seconds."""
    elapsed_seconds = perf_counter() - started_at
    del profile
    logger.log(
        level,
        "%s completed in %s.",
        phase_label(phase_name),
        format_duration(elapsed_seconds),
    )
    return elapsed_seconds


def log_phase_failed(
    logger: logging.Logger,
    *,
    phase_name: str,
    started_at: float,
    profile: str | None = None,
    level: int = logging.WARNING,
) -> float:
    """Logs phase failure duration and returns elapsed duration in seconds."""
    elapsed_seconds = perf_counter() - started_at
    del profile
    logger.log(
        level,
        "%s failed after %s.",
        phase_label(phase_name),
        format_duration(elapsed_seconds),
    )
    return elapsed_seconds
