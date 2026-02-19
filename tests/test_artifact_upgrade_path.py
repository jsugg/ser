"""Compatibility tests for artifact metadata v2 upgrade behavior."""

from __future__ import annotations

import pytest
from sklearn.neural_network import MLPClassifier

from ser.models import emotion_model as em


def _build_classifier() -> MLPClassifier:
    """Creates a deterministic classifier instance for artifact tests."""
    return MLPClassifier(
        hidden_layer_sizes=(2,),
        max_iter=20,
        solver="lbfgs",
        random_state=0,
    )


def test_deserialize_upgrades_v1_artifact_metadata_to_v2() -> None:
    """v1 model envelopes should normalize to the v2 metadata shape."""
    payload = {
        "artifact_version": 1,
        "model": _build_classifier(),
        "metadata": {
            "artifact_version": 1,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 193,
            "training_samples": 24,
            "labels": ["happy", "sad"],
        },
    }

    loaded = em._deserialize_model_artifact(payload)

    assert loaded.expected_feature_size == 193
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["artifact_version"] == em.MODEL_ARTIFACT_VERSION
    assert loaded.artifact_metadata["artifact_schema_version"] == "v2"
    assert loaded.artifact_metadata["backend_id"] == "handcrafted"
    assert loaded.artifact_metadata["profile"] == "fast"
    assert loaded.artifact_metadata["feature_dim"] == 193
    assert loaded.artifact_metadata["pooling_strategy"] == "mean"


def test_deserialize_round_trips_v2_artifact_metadata() -> None:
    """v2 model envelopes should deserialize without lossy metadata changes."""
    artifact = em._build_model_artifact(
        model=_build_classifier(),
        feature_vector_size=193,
        training_samples=32,
        labels=["sad", "happy", "happy"],
    )

    loaded = em._deserialize_model_artifact(artifact)

    assert loaded.expected_feature_size == 193
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["artifact_version"] == em.MODEL_ARTIFACT_VERSION
    assert loaded.artifact_metadata["artifact_schema_version"] == "v2"
    assert loaded.artifact_metadata["labels"] == ["happy", "sad"]


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_match"),
    [
        ("backend_id", "", "backend_id"),
        ("feature_dim", "invalid", "feature_dim"),
    ],
)
def test_deserialize_rejects_invalid_v2_metadata(
    field_name: str,
    field_value: object,
    error_match: str,
) -> None:
    """Invalid v2 metadata should fail with actionable validation errors."""
    artifact = em._build_model_artifact(
        model=_build_classifier(),
        feature_vector_size=193,
        training_samples=32,
        labels=["happy", "sad"],
    )
    metadata = artifact.get("metadata")
    assert isinstance(metadata, dict)
    metadata[field_name] = field_value

    with pytest.raises(ValueError, match=error_match):
        em._deserialize_model_artifact(artifact)


def test_deserialize_accepts_legacy_raw_model_object() -> None:
    """Raw legacy pickles should remain loadable during migration."""
    loaded = em._deserialize_model_artifact(_build_classifier())
    assert loaded.expected_feature_size is None
    assert loaded.artifact_metadata is None
