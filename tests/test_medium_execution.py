"""Tests for medium execution owner helpers."""

from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pytest

import ser.config as config
from ser.repr import EncodedSequence
from ser.runtime import medium_execution
from ser.runtime.medium_prediction import (
    confidence_and_probabilities as _confidence_and_probabilities_impl,
)
from ser.runtime.medium_prediction import predict_labels as _predict_labels_impl


class _PredictModel:
    """Deterministic classifier stub for medium execution tests."""

    def __init__(self) -> None:
        self.classes_ = np.asarray(["happy", "sad"], dtype=object)

    def predict(self, _features: np.ndarray) -> np.ndarray:
        return np.asarray(["happy", "happy", "sad"], dtype=object)

    def predict_proba(self, _features: np.ndarray) -> np.ndarray:
        return np.asarray(
            [[0.9, 0.1], [0.75, 0.25], [0.2, 0.8]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class _LoadedModelStub:
    model: object
    expected_feature_size: int | None


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None]:
    """Keep runtime settings stable across tests."""

    config.reload_settings()
    yield
    config.reload_settings()


def _encoded_sequence() -> EncodedSequence:
    """Build a deterministic encoded timeline."""

    return EncodedSequence(
        embeddings=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        frame_start_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )


def test_pooling_windows_from_encoded_frames_uses_runtime_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Window helper should honor configured runtime overrides."""

    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_SIZE_SECONDS", "2.0")
    monkeypatch.setenv("SER_MEDIUM_POOL_WINDOW_STRIDE_SECONDS", "1.0")
    runtime_config = config.reload_settings().medium_runtime

    windows = medium_execution.pooling_windows_from_encoded_frames(
        _encoded_sequence(),
        runtime_config=runtime_config,
    )

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 2.0),
        (1.0, 3.0),
    ]


def test_run_medium_inference_once_builds_frames_and_segments() -> None:
    """Execution helper should build frame and merged segment predictions."""

    runtime_config = config.reload_settings().medium_runtime
    result = medium_execution.run_medium_inference_once(
        loaded_model=_LoadedModelStub(model=_PredictModel(), expected_feature_size=4),
        encoded=_encoded_sequence(),
        runtime_config=runtime_config,
        predict_labels=lambda model, features: _predict_labels_impl(
            model=model,
            features=features,
        ),
        confidence_and_probabilities=lambda model, features, expected_rows: (
            _confidence_and_probabilities_impl(
                model=model,
                features=features,
                expected_rows=expected_rows,
                logger=logging.getLogger("ser.tests.medium_execution"),
            )
        ),
    )

    assert len(result.frames) == 3
    assert [frame.emotion for frame in result.frames] == ["happy", "happy", "sad"]
    assert [
        (segment.emotion, segment.start_seconds, segment.end_seconds) for segment in result.segments
    ] == [("happy", 0.0, 2.0), ("sad", 2.0, 3.0)]


def test_run_medium_inference_once_rejects_feature_size_mismatch() -> None:
    """Execution helper should fail fast on incompatible pooled feature width."""

    runtime_config = config.reload_settings().medium_runtime
    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        _ = medium_execution.run_medium_inference_once(
            loaded_model=_LoadedModelStub(model=_PredictModel(), expected_feature_size=3),
            encoded=_encoded_sequence(),
            runtime_config=runtime_config,
            predict_labels=lambda model, features: _predict_labels_impl(
                model=model,
                features=features,
            ),
            confidence_and_probabilities=lambda model, features, expected_rows: (
                _confidence_and_probabilities_impl(
                    model=model,
                    features=features,
                    expected_rows=expected_rows,
                    logger=logging.getLogger("ser.tests.medium_execution"),
                )
            ),
        )
