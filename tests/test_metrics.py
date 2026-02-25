"""Unit tests for SER evaluation metrics."""

import pytest

from ser.train.metrics import compute_ser_metrics


def test_compute_ser_metrics_returns_expected_values() -> None:
    """Metrics should match deterministic fixture values."""
    y_true = ["happy", "happy", "sad", "sad", "angry", "angry"]
    y_pred = ["happy", "sad", "sad", "sad", "angry", "happy"]

    metrics = compute_ser_metrics(y_true=y_true, y_pred=y_pred)

    assert metrics["labels"] == ["angry", "happy", "sad"]
    assert metrics["uar"] == pytest.approx(2.0 / 3.0)
    assert metrics["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.5 + 0.8) / 3.0)
    assert metrics["per_class_recall"] == {
        "angry": pytest.approx(0.5),
        "happy": pytest.approx(0.5),
        "sad": pytest.approx(1.0),
    }
    assert metrics["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 2],
    ]


def test_compute_ser_metrics_uses_explicit_label_order() -> None:
    """Explicit label order should drive confusion matrix axis ordering."""
    labels = ["sad", "angry", "happy"]
    metrics = compute_ser_metrics(
        y_true=["happy", "sad", "angry"],
        y_pred=["sad", "sad", "angry"],
        labels=labels,
    )

    assert metrics["labels"] == labels
    per_class_recall = metrics["per_class_recall"]
    assert isinstance(per_class_recall, dict)
    assert list(per_class_recall.keys()) == labels
    assert metrics["confusion_matrix"] == [
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_compute_ser_metrics_rejects_mismatched_lengths() -> None:
    """Mismatched y_true/y_pred lengths should fail early."""
    with pytest.raises(ValueError, match="same length"):
        compute_ser_metrics(y_true=["happy"], y_pred=[])
