"""Contract tests for MPS out-of-memory parsing helpers."""

from __future__ import annotations

import re

import pytest

from ser.runtime import mps_oom


def test_is_mps_out_of_memory_error_detects_signature() -> None:
    """Detector should match canonical MPS out-of-memory signature."""
    assert mps_oom.is_mps_out_of_memory_error("MPS backend out of memory")
    assert not mps_oom.is_mps_out_of_memory_error("cuda out of memory")


def test_summarize_mps_oom_memory_formats_required_and_available() -> None:
    """Summary should include required and available memory when all fields exist."""
    message = (
        "MPS backend out of memory (MPS allocated: 3.13 GB, other allocations: "
        "220.17 MB, max allowed: 3.40 GB). Tried to allocate 85.83 MB on private pool."
    )
    summary = mps_oom.summarize_mps_oom_memory(message)

    assert summary.startswith("required=85.8MB available=")
    assert summary.endswith("MB")


def test_extract_memory_mib_supports_known_units() -> None:
    """Extractor should convert KB/MB/GB values to MiB."""
    pattern = re.compile(r"value:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)", re.I)

    assert mps_oom.extract_memory_mib("value: 1024 KB", pattern) == pytest.approx(1.0)
    assert mps_oom.extract_memory_mib("value: 12 MB", pattern) == pytest.approx(12.0)
    assert mps_oom.extract_memory_mib("value: 2 GB", pattern) == pytest.approx(2048.0)
    assert mps_oom.extract_memory_mib("value: 1 TB", pattern) == pytest.approx(
        1024.0 * 1024.0
    )
