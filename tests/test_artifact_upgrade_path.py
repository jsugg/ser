"""Compatibility tests for strict artifact metadata deserialization behavior."""

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


def test_deserialize_rejects_v1_artifact_metadata() -> None:
    """Legacy v1 model envelopes should fail until regenerated."""
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

    with pytest.raises(ValueError, match="Unsupported model artifact version"):
        em._deserialize_model_artifact(payload)


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


def test_deserialize_round_trips_medium_v2_artifact_metadata() -> None:
    """Medium artifacts should persist hf_xlsr/profile metadata in v2 envelope."""
    artifact = em._build_model_artifact(
        model=_build_classifier(),
        feature_vector_size=2048,
        training_samples=12,
        labels=["sad", "happy", "sad"],
        backend_id="hf_xlsr",
        profile="medium",
        feature_dim=2048,
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        pooling_strategy="mean_std",
        backend_model_id="facebook/wav2vec2-xls-r-300m",
    )

    loaded = em._deserialize_model_artifact(artifact)

    assert loaded.expected_feature_size == 2048
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["backend_id"] == "hf_xlsr"
    assert loaded.artifact_metadata["profile"] == "medium"
    assert (
        loaded.artifact_metadata["backend_model_id"] == "facebook/wav2vec2-xls-r-300m"
    )
    assert loaded.artifact_metadata["pooling_strategy"] == "mean_std"
    assert loaded.artifact_metadata["feature_dim"] == 2048


def test_deserialize_round_trips_accurate_v2_artifact_metadata() -> None:
    """Accurate artifacts should persist hf_whisper/profile metadata in v2 envelope."""
    artifact = em._build_model_artifact(
        model=_build_classifier(),
        feature_vector_size=2560,
        training_samples=10,
        labels=["sad", "happy", "angry"],
        backend_id="hf_whisper",
        profile="accurate",
        feature_dim=2560,
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        pooling_strategy="mean_std",
        backend_model_id="openai/whisper-large-v3",
    )

    loaded = em._deserialize_model_artifact(artifact)

    assert loaded.expected_feature_size == 2560
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["backend_id"] == "hf_whisper"
    assert loaded.artifact_metadata["profile"] == "accurate"
    assert loaded.artifact_metadata["backend_model_id"] == "openai/whisper-large-v3"
    assert loaded.artifact_metadata["pooling_strategy"] == "mean_std"
    assert loaded.artifact_metadata["feature_dim"] == 2560


def test_deserialize_round_trips_optional_torch_runtime_metadata() -> None:
    """Artifacts should preserve optional torch runtime selector metadata."""
    artifact = em._build_model_artifact(
        model=_build_classifier(),
        feature_vector_size=2560,
        training_samples=10,
        labels=["sad", "happy", "angry"],
        backend_id="hf_whisper",
        profile="accurate",
        feature_dim=2560,
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        pooling_strategy="mean_std",
        backend_model_id="openai/whisper-large-v3",
        torch_device="cuda:0",
        torch_dtype="float16",
    )

    loaded = em._deserialize_model_artifact(artifact)

    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["torch_device"] == "cuda:0"
    assert loaded.artifact_metadata["torch_dtype"] == "float16"


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_match"),
    [
        ("backend_id", "", "backend_id"),
        ("backend_model_id", "", "backend_model_id"),
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


def test_deserialize_rejects_legacy_raw_model_object() -> None:
    """Raw legacy pickles should fail closed without metadata envelope."""
    with pytest.raises(
        ValueError,
        match="versioned dictionary envelope",
    ):
        em._deserialize_model_artifact(_build_classifier())
