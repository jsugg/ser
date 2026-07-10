"""Accurate-profile execution helpers for the encode/pool/predict pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ser.config import ProfileRuntimeConfig
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import EncodedSequence, PoolingWindow
from ser.runtime.postprocessing import (
    build_segment_postprocessing_config,
    postprocess_frame_predictions,
)
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
)

type FeatureMatrix = NDArray[np.float64]
type ConfidenceResult = tuple[list[float], list[dict[str, float] | None]]


class LoadedModelLike(Protocol):
    """Loaded-model contract required for accurate execution."""

    @property
    def model(self) -> object: ...

    @property
    def expected_feature_size(self) -> int | None: ...


def pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    runtime_config: ProfileRuntimeConfig,
) -> list[PoolingWindow]:
    """Create temporal pooling windows from the runtime policy."""

    return temporal_pooling_windows(
        encoded,
        window_size_seconds=runtime_config.pool_window_size_seconds,
        window_stride_seconds=runtime_config.pool_window_stride_seconds,
    )


def run_accurate_inference_once(
    *,
    loaded_model: LoadedModelLike,
    encoded: EncodedSequence,
    runtime_config: ProfileRuntimeConfig,
    predict_labels: Callable[[object, FeatureMatrix], list[str]],
    confidence_and_probabilities: Callable[[object, FeatureMatrix, int], ConfidenceResult],
) -> InferenceResult:
    """Run the pure accurate execution pipeline once for an encoded sequence."""

    windows = pooling_windows_from_encoded_frames(
        encoded,
        runtime_config=runtime_config,
    )
    feature_matrix = mean_std_pool(encoded, windows)

    expected_size = loaded_model.expected_feature_size
    if expected_size is not None and int(feature_matrix.shape[1]) != expected_size:
        raise ValueError(
            "Feature vector size mismatch for accurate-profile model. "
            f"Expected {expected_size}, got {int(feature_matrix.shape[1])}."
        )

    predictions = predict_labels(loaded_model.model, feature_matrix)
    confidences, probabilities = confidence_and_probabilities(
        loaded_model.model,
        feature_matrix,
        len(windows),
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
        segments=postprocess_frame_predictions(
            frame_predictions,
            config=build_segment_postprocessing_config(runtime_config),
        ),
        frames=frame_predictions,
    )


__all__ = [
    "ConfidenceResult",
    "FeatureMatrix",
    "LoadedModelLike",
    "pooling_windows_from_encoded_frames",
    "run_accurate_inference_once",
]
