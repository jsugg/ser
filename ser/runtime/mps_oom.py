"""Helpers for parsing and summarizing MPS out-of-memory runtime messages."""

from __future__ import annotations

import re

_MPS_OOM_SIGNATURE = "mps backend out of memory"
_MPS_ALLOCATED_PATTERN = re.compile(
    r"mps allocated:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_OTHER_ALLOC_PATTERN = re.compile(
    r"other allocations:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_MAX_ALLOWED_PATTERN = re.compile(
    r"max allowed:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_REQUIRED_PATTERN = re.compile(
    r"tried to allocate\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)


def is_mps_out_of_memory_error(message: str) -> bool:
    """Returns whether one error message indicates MPS out-of-memory pressure."""
    return _MPS_OOM_SIGNATURE in message.strip().lower()


def summarize_mps_oom_memory(message: str) -> str:
    """Builds a compact required/available memory summary for MPS OOM logs."""
    required_mib = extract_memory_mib(message, _MPS_REQUIRED_PATTERN)
    allocated_mib = extract_memory_mib(message, _MPS_ALLOCATED_PATTERN)
    other_allocated_mib = extract_memory_mib(message, _MPS_OTHER_ALLOC_PATTERN) or 0.0
    max_allowed_mib = extract_memory_mib(message, _MPS_MAX_ALLOWED_PATTERN)
    if (
        required_mib is not None
        and allocated_mib is not None
        and max_allowed_mib is not None
    ):
        available_mib = max(max_allowed_mib - allocated_mib - other_allocated_mib, 0.0)
        return (
            f"required={format_memory_mib(required_mib)} "
            f"available={format_memory_mib(available_mib)}"
        )
    if required_mib is not None:
        return f"required={format_memory_mib(required_mib)}"
    return "memory headroom exhausted"


def extract_memory_mib(message: str, pattern: re.Pattern[str]) -> float | None:
    """Extracts one memory quantity from text and converts it to MiB."""
    match = pattern.search(message)
    if match is None:
        return None
    amount_text, unit_text = match.groups()
    try:
        amount = float(amount_text)
    except ValueError:
        return None
    unit = unit_text.lower()
    if unit == "kb":
        return amount / 1024.0
    if unit == "mb":
        return amount
    if unit == "gb":
        return amount * 1024.0
    if unit == "tb":
        return amount * 1024.0 * 1024.0
    return None


def format_memory_mib(memory_mib: float) -> str:
    """Formats one MiB value into a compact MB/GB string for one-line logs."""
    if memory_mib >= 1024.0:
        return f"{memory_mib / 1024.0:.2f}GB"
    return f"{memory_mib:.1f}MB"


__all__ = [
    "extract_memory_mib",
    "format_memory_mib",
    "is_mps_out_of_memory_error",
    "summarize_mps_oom_memory",
]
