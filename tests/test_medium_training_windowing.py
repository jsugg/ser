"""Tests for shared medium temporal-window policy across train and inference."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest

import ser.config as config
from ser.models import emotion_model
from ser.repr import EncodedSequence
from ser.runtime import medium_inference


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _encoded_sequence() -> EncodedSequence:
    """Builds deterministic encoded timeline for temporal-window tests."""
    return EncodedSequence(
        embeddings=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            dtype=np.float32,
        ),
        frame_start_seconds=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )


def test_medium_train_and_infer_share_configured_temporal_window_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Train and infer helper paths should emit identical configured windows."""
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_SIZE_SECONDS", "2.0")
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_STRIDE_SECONDS", "1.0")
    config.reload_settings()

    encoded = _encoded_sequence()
    train_windows = emotion_model._pooling_windows_from_encoded_frames(encoded)
    infer_windows = medium_inference._pooling_windows_from_encoded_frames(encoded)

    expected = [(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]
    assert [(w.start_seconds, w.end_seconds) for w in train_windows] == expected
    assert [(w.start_seconds, w.end_seconds) for w in infer_windows] == expected


def test_medium_temporal_window_policy_uses_runtime_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime env overrides should change generated medium pooling windows."""
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_SIZE_SECONDS", "1.5")
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_STRIDE_SECONDS", "0.5")
    config.reload_settings()

    encoded = _encoded_sequence()
    windows = medium_inference._pooling_windows_from_encoded_frames(encoded)

    assert [(w.start_seconds, w.end_seconds) for w in windows] == [
        (0.0, 1.5),
        (0.5, 2.0),
        (1.0, 2.5),
        (1.5, 3.0),
        (2.0, 3.5),
        (2.5, 4.0),
    ]
