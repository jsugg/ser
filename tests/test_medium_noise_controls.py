"""Contract tests for medium noise-control helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from ser.models.medium_noise_controls import (
    MediumNoiseControlStats,
    apply_medium_noise_controls,
    merge_medium_noise_stats,
)


def _pooled_features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 0.01, 0.01],
            [2.0, 2.0, 0.50, 0.50],
            [3.0, 3.0, 0.20, 0.20],
        ],
        dtype=np.float64,
    )


def test_apply_medium_noise_controls_filters_and_caps() -> None:
    """Helper should apply std-threshold then deterministic max-window cap."""
    filtered, stats = apply_medium_noise_controls(
        _pooled_features(),
        min_window_std=0.15,
        max_windows_per_clip=1,
    )

    np.testing.assert_allclose(
        filtered,
        np.asarray([[2.0, 2.0, 0.50, 0.50]], dtype=np.float64),
    )
    assert stats.total_windows == 3
    assert stats.kept_windows == 1
    assert stats.dropped_low_std_windows == 1
    assert stats.dropped_cap_windows == 1
    assert stats.forced_keep_windows == 0


def test_apply_medium_noise_controls_forced_keep() -> None:
    """Helper should force-keep one window when threshold would drop all rows."""
    filtered, stats = apply_medium_noise_controls(
        _pooled_features(),
        min_window_std=10.0,
        max_windows_per_clip=0,
    )

    np.testing.assert_allclose(
        filtered,
        np.asarray([[2.0, 2.0, 0.50, 0.50]], dtype=np.float64),
    )
    assert stats.forced_keep_windows == 1
    assert stats.kept_windows == 1


def test_merge_medium_noise_stats_sums_fields() -> None:
    """Merge helper should add all per-clip counters."""
    merged = merge_medium_noise_stats(
        base=MediumNoiseControlStats(
            total_windows=3,
            kept_windows=2,
            dropped_low_std_windows=1,
            dropped_cap_windows=0,
            forced_keep_windows=0,
        ),
        incoming=MediumNoiseControlStats(
            total_windows=4,
            kept_windows=3,
            dropped_low_std_windows=1,
            dropped_cap_windows=1,
            forced_keep_windows=1,
        ),
    )

    assert merged == MediumNoiseControlStats(
        total_windows=7,
        kept_windows=5,
        dropped_low_std_windows=2,
        dropped_cap_windows=1,
        forced_keep_windows=1,
    )


def test_apply_medium_noise_controls_rejects_invalid_shape() -> None:
    """Helper should reject invalid pooled-feature rank/shape."""
    with pytest.raises(RuntimeError, match="non-empty 2D matrix"):
        _ = apply_medium_noise_controls(
            np.asarray([1.0, 2.0], dtype=np.float64),
            min_window_std=0.0,
            max_windows_per_clip=0,
        )
