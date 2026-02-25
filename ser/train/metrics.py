"""SER-focused evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import confusion_matrix, f1_score, recall_score


def compute_ser_metrics(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute stable SER metrics for model evaluation and reporting.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Optional explicit class order.

    Returns:
        A dictionary containing UAR, macro-F1, per-class recall,
        confusion matrix, and class ordering.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            "Expected y_true and y_pred to have the same length; "
            f"got {len(y_true)} and {len(y_pred)}."
        )
    if not y_true:
        raise ValueError("Expected non-empty label sequences for metric computation.")

    label_order = (
        [str(label) for label in labels]
        if labels is not None
        else sorted({*map(str, y_true), *map(str, y_pred)})
    )

    uar = float(
        recall_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            labels=label_order,
        )
    )
    macro_f1 = float(
        f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            labels=label_order,
        )
    )
    confusion = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=label_order)
    per_class_recall: dict[str, float] = {}
    for idx, label in enumerate(label_order):
        row_total = int(confusion[idx].sum())
        true_positive = int(confusion[idx, idx])
        per_class_recall[label] = (
            float(true_positive) / float(row_total) if row_total > 0 else 0.0
        )
    return {
        "labels": label_order,
        "uar": uar,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
    }
