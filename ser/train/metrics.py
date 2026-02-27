"""SER-focused evaluation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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


def compute_grouped_ser_metrics_by_sample(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    sample_ids: Sequence[str],
    group_ids: Sequence[str],
    min_support: int,
) -> dict[str, Any]:
    """Compute group metrics with minimum support, aggregated per sample.

    The inputs are window-level predictions/labels. The function aggregates
    per-sample via majority vote on ``sample_ids``, then computes SER metrics
    per group id (e.g., corpus or language).
    """

    if not (len(y_true) == len(y_pred) == len(sample_ids) == len(group_ids)):
        raise ValueError("y_true/y_pred/sample_ids/group_ids must have equal length")
    if min_support < 1:
        raise ValueError("min_support must be >= 1")
    if not y_true:
        return {
            "unit": "samples",
            "min_support": min_support,
            "included": {},
            "excluded": {},
        }

    per_sample_true: dict[str, list[str]] = {}
    per_sample_pred: dict[str, list[str]] = {}
    per_sample_group: dict[str, list[str]] = {}
    for true_label, pred_label, sample_id, group_id in zip(
        y_true,
        y_pred,
        sample_ids,
        group_ids,
        strict=True,
    ):
        per_sample_true.setdefault(str(sample_id), []).append(str(true_label))
        per_sample_pred.setdefault(str(sample_id), []).append(str(pred_label))
        per_sample_group.setdefault(str(sample_id), []).append(str(group_id))

    def _mode(values: list[str]) -> str:
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        # Deterministic tie-break.
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    sample_true: list[str] = []
    sample_pred: list[str] = []
    sample_group: list[str] = []
    for sample_id in sorted(per_sample_true):
        sample_true.append(_mode(per_sample_true[sample_id]))
        sample_pred.append(_mode(per_sample_pred[sample_id]))
        sample_group.append(_mode(per_sample_group[sample_id]))

    grouped_true: dict[str, list[str]] = {}
    grouped_pred: dict[str, list[str]] = {}
    for true_label, pred_label, group_id in zip(
        sample_true,
        sample_pred,
        sample_group,
        strict=True,
    ):
        grouped_true.setdefault(group_id, []).append(true_label)
        grouped_pred.setdefault(group_id, []).append(pred_label)

    included: dict[str, Any] = {}
    excluded: dict[str, Any] = {}
    for group_id in sorted(grouped_true):
        support = len(grouped_true[group_id])
        if support < min_support:
            excluded[group_id] = {"support": support}
            continue
        included[group_id] = {
            "support": support,
            "metrics": compute_ser_metrics(
                y_true=grouped_true[group_id],
                y_pred=grouped_pred[group_id],
            ),
        }
    return {
        "unit": "samples",
        "min_support": min_support,
        "included": included,
        "excluded": excluded,
    }
