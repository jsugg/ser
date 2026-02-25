"""Timestamp boundary semantics for emotion segmentation."""

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from ser.domain import EmotionSegment
from ser.features import FeatureFrame
from ser.models import emotion_model as em


class StaticModel(MLPClassifier):
    """Simple deterministic predictor used for timestamp semantics tests."""

    def __init__(self, predictions: list[str]) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self._predictions = predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.asarray(self._predictions, dtype=object)


def _frame(start_seconds: float, end_seconds: float, marker: float) -> FeatureFrame:
    return FeatureFrame(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        features=np.asarray([marker, marker], dtype=np.float64),
    )


def test_predict_emotions_uses_explicit_frame_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segment transitions should follow frame boundaries, not averaged duration."""
    frames = [
        _frame(0.0, 2.0, 1.0),
        _frame(1.0, 3.0, 2.0),
        _frame(2.0, 4.4, 3.0),
    ]

    monkeypatch.setattr(em, "extract_feature_frames", lambda _file: frames)
    monkeypatch.setattr(
        em,
        "load_model",
        lambda: em.LoadedModel(
            model=StaticModel(["angry", "happy", "happy"]),
            expected_feature_size=2,
        ),
    )

    segments = em.predict_emotions("sample.wav")

    assert segments == [
        EmotionSegment("angry", 0.0, 2.0),
        EmotionSegment("happy", 1.0, 4.4),
    ]


def test_predict_emotions_single_frame_bounds_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-frame inference should preserve that frame's start and end."""
    frame = _frame(0.25, 0.75, 1.0)

    monkeypatch.setattr(em, "extract_feature_frames", lambda _file: [frame])
    monkeypatch.setattr(
        em,
        "load_model",
        lambda: em.LoadedModel(
            model=StaticModel(["sad"]),
            expected_feature_size=2,
        ),
    )

    segments = em.predict_emotions("sample.wav")

    assert segments == [EmotionSegment("sad", 0.25, 0.75)]


def test_predict_emotions_rejects_frame_prediction_length_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatch between frame count and prediction count should fail clearly."""
    frames = [_frame(0.0, 1.0, 1.0), _frame(1.0, 2.0, 2.0)]

    monkeypatch.setattr(em, "extract_feature_frames", lambda _file: frames)
    monkeypatch.setattr(
        em,
        "load_model",
        lambda: em.LoadedModel(
            model=StaticModel(["neutral"]),
            expected_feature_size=2,
        ),
    )

    with pytest.raises(RuntimeError, match="Frame/prediction length mismatch"):
        em.predict_emotions("sample.wav")
