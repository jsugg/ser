"""Medium-profile inference runner with encode-once/pool-many semantics."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel, load_model
from ser.pool import mean_std_pool
from ser.repr import EncodedSequence, PoolingWindow, XLSRBackend
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
)
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type FeatureMatrix = NDArray[np.float64]


class MediumModelUnavailableError(FileNotFoundError):
    """Raised when a compatible medium-profile model artifact is unavailable."""


def run_medium_inference(request: InferenceRequest, settings: AppConfig) -> InferenceResult:
    """Runs medium-profile inference via XLS-R encode-once/pool-many execution.

    Args:
        request: Runtime inference request payload.
        settings: Active application settings.

    Returns:
        Detailed inference result with frame and segment predictions.

    Raises:
        MediumModelUnavailableError: If no medium-compatible model artifact exists.
        RuntimeError: If classifier/model contracts are invalid.
        ValueError: If feature dimensions are incompatible with the loaded artifact.
    """
    del settings
    loaded_model = load_model()
    _ensure_medium_compatible_model(loaded_model)

    audio, sample_rate = read_audio_file(request.file_path)
    backend = XLSRBackend()
    encoded = backend.encode_sequence(audio, sample_rate)
    windows = _pooling_windows_from_encoded_frames(encoded)
    feature_matrix = mean_std_pool(encoded, windows)

    expected_size = loaded_model.expected_feature_size
    if expected_size is not None and int(feature_matrix.shape[1]) != expected_size:
        raise ValueError(
            "Feature vector size mismatch for medium-profile model. "
            f"Expected {expected_size}, got {int(feature_matrix.shape[1])}."
        )

    predictions = _predict_labels(loaded_model.model, feature_matrix)
    confidences, probabilities = _confidence_and_probabilities(
        loaded_model.model,
        feature_matrix,
        expected_rows=len(windows),
    )

    frame_predictions = [
        FramePrediction(
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
            emotion=predictions[index],
            confidence=confidences[index],
            probabilities=probabilities[index],
        )
        for index, window in enumerate(windows)
    ]
    return InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=_segment_predictions(frame_predictions),
        frames=frame_predictions,
    )


def _ensure_medium_compatible_model(loaded_model: LoadedModel) -> None:
    """Validates that loaded artifact metadata is compatible with medium runtime."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise MediumModelUnavailableError(
            "Medium profile requires a v2 model artifact metadata envelope. "
            "Train a medium-profile model before inference."
        )

    backend_id = metadata.get("backend_id")
    if backend_id != "hf_xlsr":
        raise MediumModelUnavailableError(
            "No medium-profile model artifact is available. "
            f"Found backend_id={backend_id!r}; expected 'hf_xlsr'."
        )

    profile = metadata.get("profile")
    if profile != "medium":
        raise MediumModelUnavailableError(
            "No medium-profile model artifact is available. "
            f"Found profile={profile!r}; expected 'medium'."
        )


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
) -> list[PoolingWindow]:
    """Creates one pooling window per encoded frame boundary."""
    return [
        PoolingWindow(
            start_seconds=float(start),
            end_seconds=float(end),
        )
        for start, end in zip(
            encoded.frame_start_seconds,
            encoded.frame_end_seconds,
            strict=True,
        )
    ]


def _predict_labels(model: object, features: FeatureMatrix) -> list[str]:
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


def _confidence_and_probabilities(
    model: object,
    features: FeatureMatrix,
    *,
    expected_rows: int,
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
    if not isinstance(classes_attr, (list, tuple, np.ndarray)):
        logger.warning(
            "Medium model classes_ metadata is unavailable; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    try:
        raw_probabilities = np.asarray(predict_proba(features), dtype=np.float64)
    except Exception as err:
        logger.warning("Medium model predict_proba failed; using fallback. Error: %s", err)
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

    class_labels = [str(item) for item in classes_attr]
    if int(raw_probabilities.shape[1]) != len(class_labels):
        logger.warning(
            "Medium model predict_proba class mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            int(raw_probabilities.shape[1]),
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {
            class_labels[idx]: float(row[idx])
            for idx in range(len(class_labels))
        }
        for row in raw_probabilities
    ]
    return confidences, probabilities


def _aggregate_probabilities(
    probabilities: Sequence[dict[str, float] | None],
) -> dict[str, float] | None:
    """Aggregates per-frame probability dictionaries into a segment-level mapping."""
    valid = [item for item in probabilities if item is not None]
    if not valid:
        return None

    labels = sorted({label for item in valid for label in item.keys()})
    aggregated = {
        label: float(
            fmean(
                float(item.get(label, 0.0))
                for item in valid
            )
        )
        for label in labels
    }
    return aggregated


def _segment_predictions(
    frame_predictions: list[FramePrediction],
) -> list[SegmentPrediction]:
    """Merges adjacent equal-label frame predictions into segments."""
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
                probabilities=_aggregate_probabilities(active_probabilities),
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
            probabilities=_aggregate_probabilities(active_probabilities),
        )
    )
    return segments
