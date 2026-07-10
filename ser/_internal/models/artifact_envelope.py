"""Versioned artifact-envelope helpers for SER model persistence."""

from __future__ import annotations

from typing import NamedTuple, cast

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ser.models.artifact_loading import (
    build_model_artifact_envelope,
    deserialize_model_artifact_envelope,
)
from ser.models.artifact_metadata import (
    build_v2_artifact_metadata,
    normalize_v2_artifact_metadata,
    read_positive_int,
)
from ser.runtime.schema import ARTIFACT_SCHEMA_VERSION

MODEL_ARTIFACT_VERSION = 2
DEFAULT_BACKEND_ID = "handcrafted"
DEFAULT_PROFILE_ID = "fast"
DEFAULT_FRAME_SIZE_SECONDS = 3.0
DEFAULT_FRAME_STRIDE_SECONDS = 1.0
DEFAULT_POOLING_STRATEGY = "mean"

type EmotionClassifier = MLPClassifier | Pipeline


class LoadedModel(NamedTuple):
    """Loaded model object and optional expected feature-vector length."""

    model: EmotionClassifier
    expected_feature_size: int | None
    artifact_metadata: dict[str, object] | None = None


def normalize_model_artifact_metadata(
    metadata: dict[str, object],
) -> dict[str, object]:
    """Validates and normalizes artifact metadata to the current envelope shape."""
    return normalize_v2_artifact_metadata(
        metadata,
        artifact_version=MODEL_ARTIFACT_VERSION,
    )


def build_model_artifact_metadata(
    *,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
    backend_id: str = DEFAULT_BACKEND_ID,
    profile: str = DEFAULT_PROFILE_ID,
    feature_dim: int | None = None,
    frame_size_seconds: float = DEFAULT_FRAME_SIZE_SECONDS,
    frame_stride_seconds: float = DEFAULT_FRAME_STRIDE_SECONDS,
    pooling_strategy: str = DEFAULT_POOLING_STRATEGY,
    backend_model_id: str | None = None,
    torch_device: str | None = None,
    torch_dtype: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Builds normalized artifact metadata for persisted model envelopes."""
    return build_v2_artifact_metadata(
        artifact_version=MODEL_ARTIFACT_VERSION,
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        feature_vector_size=feature_vector_size,
        training_samples=training_samples,
        labels=labels,
        backend_id=backend_id,
        profile=profile,
        feature_dim=feature_dim,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        pooling_strategy=pooling_strategy,
        backend_model_id=backend_model_id,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        provenance=provenance,
    )


def build_model_artifact(
    model: EmotionClassifier,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
    backend_id: str = DEFAULT_BACKEND_ID,
    profile: str = DEFAULT_PROFILE_ID,
    feature_dim: int | None = None,
    frame_size_seconds: float = DEFAULT_FRAME_SIZE_SECONDS,
    frame_stride_seconds: float = DEFAULT_FRAME_STRIDE_SECONDS,
    pooling_strategy: str = DEFAULT_POOLING_STRATEGY,
    backend_model_id: str | None = None,
    torch_device: str | None = None,
    torch_dtype: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Constructs one versioned model artifact envelope for safe persistence."""
    metadata = build_model_artifact_metadata(
        feature_vector_size=feature_vector_size,
        training_samples=training_samples,
        labels=labels,
        backend_id=backend_id,
        profile=profile,
        feature_dim=feature_dim,
        frame_size_seconds=frame_size_seconds,
        frame_stride_seconds=frame_stride_seconds,
        pooling_strategy=pooling_strategy,
        backend_model_id=backend_model_id,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        provenance=provenance,
    )
    return build_model_artifact_envelope(
        artifact_version=MODEL_ARTIFACT_VERSION,
        model=model,
        metadata=metadata,
    )


def deserialize_model_artifact(payload: object) -> LoadedModel:
    """Validates and unwraps one persisted model artifact payload."""
    return deserialize_model_artifact_envelope(
        payload,
        artifact_version=MODEL_ARTIFACT_VERSION,
        model_instance_check=lambda model: isinstance(model, MLPClassifier | Pipeline),
        normalize_metadata=normalize_model_artifact_metadata,
        read_positive_int=read_positive_int,
        loaded_model_factory=lambda model, expected_feature_size, artifact_metadata: (
            LoadedModel(
                model=cast(EmotionClassifier, model),
                expected_feature_size=expected_feature_size,
                artifact_metadata=artifact_metadata,
            )
        ),
    )
