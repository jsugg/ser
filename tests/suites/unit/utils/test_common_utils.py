"""Unit coverage for shared elapsed-time formatting helpers."""

from __future__ import annotations

import pytest

from ser.utils.common_utils import display_elapsed_time

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("elapsed_time", "output_format", "expected"),
    [
        (12.3456, "long", "12.35 seconds"),
        (65.0, "long", "1 min 5 seconds"),
        (12.3456, "short", "12.35s"),
        (65.0, "short", "1m5s"),
    ],
)
def test_display_elapsed_time_formats_short_and_long_variants(
    elapsed_time: float,
    output_format: str,
    expected: str,
) -> None:
    """Elapsed time formatting should stay stable across supported styles."""
    assert display_elapsed_time(elapsed_time, _format=output_format) == expected
