"""Medium-profile prediction helpers for label/probability decoding."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

type FeatureMatrix = NDArray[np.float64]


def predict_labels(model: object, features: FeatureMatrix) -> list[str]:
    """Runs model prediction and validates row-aligned label output."""
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise RuntimeError("Loaded medium model does not expose a callable predict().")

    labels = np.asarray(predict(features), dtype=object)
    if labels.ndim != 1:
        raise RuntimeError(
            "Medium model predict() returned invalid rank; expected 1D labels."
        )
    if int(labels.shape[0]) != int(features.shape[0]):
        raise RuntimeError(
            "Medium model prediction row count mismatch. "
            f"Expected {int(features.shape[0])}, got {int(labels.shape[0])}."
        )
    return [str(item) for item in labels.tolist()]


def confidence_and_probabilities(
    model: object,
    features: FeatureMatrix,
    *,
    expected_rows: int,
    logger: logging.Logger,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability mappings."""
    fallback_confidence = [1.0] * expected_rows
    fallback_probabilities: list[dict[str, float] | None] = [None] * expected_rows

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        logger.warning(
            "Medium model does not expose predict_proba; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    classes_attr = getattr(model, "classes_", None)
    class_values: list[object] | None = None
    if isinstance(classes_attr, np.ndarray):
        class_values = list(classes_attr.tolist())
    elif isinstance(classes_attr, list | tuple):
        class_values = list(classes_attr)
    if class_values is None:
        logger.warning(
            "Medium model classes_ metadata is unavailable; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    try:
        raw_probabilities = np.asarray(predict_proba(features), dtype=np.float64)
    except Exception as err:
        logger.warning(
            "Medium model predict_proba failed; using fallback. Error: %s", err
        )
        return fallback_confidence, fallback_probabilities

    if raw_probabilities.ndim != 2:
        logger.warning(
            "Medium model predict_proba returned invalid rank %s; using fallback.",
            raw_probabilities.shape,
        )
        return fallback_confidence, fallback_probabilities
    if int(raw_probabilities.shape[0]) != expected_rows:
        logger.warning(
            "Medium model predict_proba row mismatch (expected=%s, got=%s); using fallback.",
            expected_rows,
            int(raw_probabilities.shape[0]),
        )
        return fallback_confidence, fallback_probabilities

    class_labels = [str(item) for item in class_values]
    if int(raw_probabilities.shape[1]) != len(class_labels):
        logger.warning(
            "Medium model predict_proba class mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            int(raw_probabilities.shape[1]),
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {class_labels[idx]: float(row[idx]) for idx in range(len(class_labels))}
        for row in raw_probabilities
    ]
    return confidences, probabilities


__all__ = ["confidence_and_probabilities", "predict_labels"]
