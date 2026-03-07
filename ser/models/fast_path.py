"""Fast-profile inference helpers extracted from the legacy emotion model module."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from statistics import fmean

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ser.features import FeatureFrame
from ser.runtime.schema import FramePrediction, InferenceResult, SegmentPrediction

type EmotionClassifier = MLPClassifier | Pipeline


def frame_confidence_and_probabilities(
    model: EmotionClassifier,
    feature_matrix: np.ndarray,
    frame_count: int,
    *,
    logger: logging.Logger,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability maps."""
    fallback_confidence = [1.0] * frame_count
    fallback_probabilities: list[dict[str, float] | None] = [None] * frame_count

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        logger.warning(
            "Loaded model does not expose predict_proba; using confidence=1.0 fallback."
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
            "Loaded model predict_proba path missing classes_; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    class_labels = [str(item) for item in class_values]
    raw_probabilities = np.asarray(predict_proba(feature_matrix), dtype=np.float64)
    if raw_probabilities.ndim != 2:
        logger.warning(
            "Unexpected predict_proba output shape %s; using confidence fallback.",
            raw_probabilities.shape,
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[0] != frame_count:
        logger.warning(
            "predict_proba frame count mismatch (expected=%s, got=%s); using fallback.",
            frame_count,
            raw_probabilities.shape[0],
        )
        return fallback_confidence, fallback_probabilities
    if raw_probabilities.shape[1] != len(class_labels):
        logger.warning(
            "predict_proba class count mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            raw_probabilities.shape[1],
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {class_labels[idx]: float(row[idx]) for idx in range(len(class_labels))}
        for row in raw_probabilities
    ]
    return confidences, probabilities


def aggregate_probabilities(
    probabilities: list[dict[str, float] | None],
) -> dict[str, float] | None:
    """Averages per-frame probabilities when all frames provide full maps."""
    if not probabilities or any(item is None for item in probabilities):
        return None

    first = probabilities[0]
    if first is None:
        return None
    labels = list(first.keys())
    if any(
        item is None or set(item.keys()) != set(labels) for item in probabilities[1:]
    ):
        return None

    aggregates: dict[str, float] = {}
    for label in labels:
        values = [item[label] for item in probabilities if item is not None]
        aggregates[label] = float(fmean(values))
    return aggregates


def segment_predictions(
    frame_predictions: list[FramePrediction],
) -> list[SegmentPrediction]:
    """Merges adjacent equal frame labels into segment-level predictions."""
    if not frame_predictions:
        return []

    segments: list[SegmentPrediction] = []
    active_emotion = frame_predictions[0].emotion
    active_start = frame_predictions[0].start_seconds
    active_end = frame_predictions[0].end_seconds
    active_confidences = [frame_predictions[0].confidence]
    active_probabilities = [frame_predictions[0].probabilities]

    for frame in frame_predictions[1:]:
        if frame.emotion == active_emotion:
            active_end = frame.end_seconds
            active_confidences.append(frame.confidence)
            active_probabilities.append(frame.probabilities)
            continue

        segments.append(
            SegmentPrediction(
                emotion=active_emotion,
                start_seconds=active_start,
                end_seconds=active_end,
                confidence=float(fmean(active_confidences)),
                probabilities=aggregate_probabilities(active_probabilities),
            )
        )
        active_emotion = frame.emotion
        active_start = frame.start_seconds
        active_end = frame.end_seconds
        active_confidences = [frame.confidence]
        active_probabilities = [frame.probabilities]

    segments.append(
        SegmentPrediction(
            emotion=active_emotion,
            start_seconds=active_start,
            end_seconds=active_end,
            confidence=float(fmean(active_confidences)),
            probabilities=aggregate_probabilities(active_probabilities),
        )
    )
    return segments


def predict_emotions_detailed_with_model(
    file: str,
    *,
    model: EmotionClassifier,
    expected_feature_size: int | None,
    output_schema_version: str,
    extract_feature_frames_fn: Callable[[str], Sequence[FeatureFrame]],
    logger: logging.Logger,
) -> InferenceResult:
    """Runs fast-path inference with preloaded model and returns detailed predictions."""
    feature_frames = list(extract_feature_frames_fn(file))
    if not feature_frames:
        logger.warning("No features extracted for file %s.", file)
        return InferenceResult(
            schema_version=output_schema_version,
            segments=[],
            frames=[],
        )

    feature_vectors: list[np.ndarray] = [frame.features for frame in feature_frames]
    if expected_feature_size is not None:
        invalid_sizes = {
            vector.shape[0]
            for vector in feature_vectors
            if vector.shape[0] != expected_feature_size
        }
        if invalid_sizes:
            raise ValueError(
                "Feature vector size mismatch for loaded model. "
                f"Expected {expected_feature_size}, "
                f"got {sorted(invalid_sizes)}."
            )

    feature_matrix = np.asarray(feature_vectors, dtype=np.float64)
    predicted_emotions: list[str] = [
        str(item) for item in model.predict(feature_matrix)
    ]
    if len(predicted_emotions) != len(feature_frames):
        raise RuntimeError(
            "Frame/prediction length mismatch. "
            f"Got {len(feature_frames)} frames and {len(predicted_emotions)} predictions."
        )
    confidences, probabilities = frame_confidence_and_probabilities(
        model=model,
        feature_matrix=feature_matrix,
        frame_count=len(feature_frames),
        logger=logger,
    )

    logger.debug(
        "Emotion model prediction completed for %d frames.",
        len(predicted_emotions),
    )
    if not predicted_emotions:
        logger.warning("No emotions predicted for file %s.", file)
        return InferenceResult(
            schema_version=output_schema_version,
            segments=[],
            frames=[],
        )

    frame_predictions = [
        FramePrediction(
            start_seconds=feature_frames[idx].start_seconds,
            end_seconds=feature_frames[idx].end_seconds,
            emotion=predicted_emotions[idx],
            confidence=confidences[idx],
            probabilities=probabilities[idx],
        )
        for idx in range(len(feature_frames))
    ]
    logger.debug("Timestamp extraction started.")
    resolved_segments = segment_predictions(frame_predictions)
    logger.debug(
        "Timestamp extraction completed for %d segments.",
        len(resolved_segments),
    )
    return InferenceResult(
        schema_version=output_schema_version,
        segments=resolved_segments,
        frames=frame_predictions,
    )
