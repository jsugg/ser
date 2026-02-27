"""Unit tests for SER evaluation metrics."""

import pytest

from ser.train.metrics import (
    compute_grouped_ser_metrics_by_sample,
    compute_ser_metrics,
)


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


def test_grouped_ser_metrics_by_sample_aggregates_majority_vote() -> None:
    """Window-level predictions should collapse to per-sample grouped metrics."""
    grouped = compute_grouped_ser_metrics_by_sample(
        y_true=["happy", "happy", "sad", "sad"],
        y_pred=["happy", "sad", "sad", "sad"],
        sample_ids=["s1", "s1", "s2", "s2"],
        group_ids=["ravdess", "ravdess", "crema-d", "crema-d"],
        min_support=1,
    )

    assert grouped["unit"] == "samples"
    assert grouped["min_support"] == 1
    included = grouped["included"]
    assert isinstance(included, dict)
    ravdess = included["ravdess"]
    crema = included["crema-d"]
    assert ravdess["support"] == 1
    assert crema["support"] == 1
    assert ravdess["metrics"]["labels"] == ["happy"]
    assert crema["metrics"]["labels"] == ["sad"]


def test_grouped_ser_metrics_by_sample_applies_min_support() -> None:
    """Groups below threshold should be listed under excluded."""
    grouped = compute_grouped_ser_metrics_by_sample(
        y_true=["happy", "happy", "sad", "sad"],
        y_pred=["happy", "happy", "sad", "sad"],
        sample_ids=["s1", "s1", "s2", "s2"],
        group_ids=["a", "a", "b", "b"],
        min_support=2,
    )

    included = grouped["included"]
    excluded = grouped["excluded"]
    assert isinstance(included, dict)
    assert isinstance(excluded, dict)
    assert included == {}
    assert excluded["a"]["support"] == 1
    assert excluded["b"]["support"] == 1
