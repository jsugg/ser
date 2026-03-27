"""Contract tests for medium prediction/probability helper functions."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from ser.runtime.medium_prediction import (
    confidence_and_probabilities,
    predict_labels,
)


def _features() -> np.ndarray[Any, np.dtype[np.float64]]:
    return np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)


class _PredictOnlyModel:
    def __init__(self, *, labels: list[str]) -> None:
        self._labels = np.asarray(labels, dtype=object)

    def predict(
        self,
        features: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.object_]]:
        del features
        return self._labels


class _PredictProbaModel(_PredictOnlyModel):
    def __init__(
        self,
        *,
        labels: list[str],
        probabilities: list[list[float]],
        classes: list[str],
    ) -> None:
        super().__init__(labels=labels)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=object)

    def predict_proba(
        self,
        features: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        del features
        return self._probabilities


def test_predict_labels_returns_row_aligned_strings() -> None:
    """Predict helper should coerce medium model labels to strings."""
    model = _PredictOnlyModel(labels=["happy", "sad"])
    assert predict_labels(model=model, features=_features()) == ["happy", "sad"]


def test_predict_labels_rejects_row_mismatch() -> None:
    """Predict helper should fail fast when prediction rows mismatch features."""
    model = _PredictOnlyModel(labels=["happy"])
    with pytest.raises(RuntimeError, match="prediction row count mismatch"):
        _ = predict_labels(model=model, features=_features())


def test_confidence_and_probabilities_falls_back_without_predict_proba() -> None:
    """Confidence helper should return deterministic fallback for predict-only models."""
    confidences, probabilities = confidence_and_probabilities(
        model=_PredictOnlyModel(labels=["happy", "sad"]),
        features=_features(),
        expected_rows=2,
        logger=logging.getLogger("tests.medium_prediction"),
    )

    assert confidences == [1.0, 1.0]
    assert probabilities == [None, None]


def test_confidence_and_probabilities_returns_probability_maps() -> None:
    """Probability helper should return per-row confidence and class maps."""
    confidences, probabilities = confidence_and_probabilities(
        model=_PredictProbaModel(
            labels=["happy", "sad"],
            probabilities=[[0.8, 0.2], [0.3, 0.7]],
            classes=["happy", "sad"],
        ),
        features=_features(),
        expected_rows=2,
        logger=logging.getLogger("tests.medium_prediction"),
    )

    assert confidences == pytest.approx([0.8, 0.7])
    assert probabilities == [
        pytest.approx({"happy": 0.8, "sad": 0.2}),
        pytest.approx({"happy": 0.3, "sad": 0.7}),
    ]
