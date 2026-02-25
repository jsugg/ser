"""Tests for medium training label-noise controls."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest

import ser.config as config
from ser.models import emotion_model as em


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _pooled_features() -> np.ndarray:
    """Builds deterministic mean+std pooled feature rows."""
    return np.asarray(
        [
            [1.0, 1.0, 0.01, 0.01],
            [2.0, 2.0, 0.50, 0.50],
            [3.0, 3.0, 0.20, 0.20],
        ],
        dtype=np.float64,
    )


def test_medium_noise_controls_default_keeps_all_windows() -> None:
    """Default settings should keep all windows without drops."""
    filtered, stats = em._apply_medium_noise_controls(_pooled_features())

    np.testing.assert_allclose(filtered, _pooled_features())
    assert stats.total_windows == 3
    assert stats.kept_windows == 3
    assert stats.dropped_low_std_windows == 0
    assert stats.dropped_cap_windows == 0
    assert stats.forced_keep_windows == 0


def test_medium_noise_controls_filters_low_std_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Min-std threshold should drop low-variance windows deterministically."""
    monkeypatch.setenv("SER_MEDIUM_MIN_WINDOW_STD", "0.25")
    config.reload_settings()

    filtered, stats = em._apply_medium_noise_controls(_pooled_features())

    np.testing.assert_allclose(
        filtered,
        np.asarray([[2.0, 2.0, 0.50, 0.50]], dtype=np.float64),
    )
    assert stats.total_windows == 3
    assert stats.kept_windows == 1
    assert stats.dropped_low_std_windows == 2
    assert stats.dropped_cap_windows == 0
    assert stats.forced_keep_windows == 0


def test_medium_noise_controls_forced_keep_when_all_rows_are_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When threshold drops all rows, one max-std row should be forced to remain."""
    monkeypatch.setenv("SER_MEDIUM_MIN_WINDOW_STD", "10.0")
    config.reload_settings()

    filtered, stats = em._apply_medium_noise_controls(_pooled_features())

    np.testing.assert_allclose(
        filtered,
        np.asarray([[2.0, 2.0, 0.50, 0.50]], dtype=np.float64),
    )
    assert stats.total_windows == 3
    assert stats.kept_windows == 1
    assert stats.dropped_low_std_windows == 2
    assert stats.forced_keep_windows == 1


def test_medium_noise_controls_caps_windows_per_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Max-window cap should downsample rows deterministically across the clip."""
    monkeypatch.setenv("SER_MEDIUM_MAX_WINDOWS_PER_CLIP", "2")
    config.reload_settings()

    filtered, stats = em._apply_medium_noise_controls(_pooled_features())

    np.testing.assert_allclose(
        filtered,
        np.asarray(
            [
                [1.0, 1.0, 0.01, 0.01],
                [3.0, 3.0, 0.20, 0.20],
            ],
            dtype=np.float64,
        ),
    )
    assert stats.total_windows == 3
    assert stats.kept_windows == 2
    assert stats.dropped_cap_windows == 1
