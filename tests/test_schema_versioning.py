"""Tests for runtime schema versioning contracts."""

from ser.runtime.schema import (
    ARTIFACT_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
)


def test_schema_constants_default_to_v1() -> None:
    """Schema seed constants should remain stable until explicit migration."""
    assert OUTPUT_SCHEMA_VERSION == "v1"
    assert ARTIFACT_SCHEMA_VERSION == "v1"


def test_inference_result_contract_shape() -> None:
    """Inference result should expose versioned segments and frames."""
    frame = FramePrediction(
        start_seconds=0.0,
        end_seconds=1.0,
        emotion="happy",
        confidence=0.8,
        probabilities={"happy": 0.8, "sad": 0.2},
    )
    segment = SegmentPrediction(
        emotion="happy",
        start_seconds=0.0,
        end_seconds=1.0,
        confidence=0.8,
        probabilities={"happy": 0.8, "sad": 0.2},
    )
    result = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[segment],
        frames=[frame],
    )

    assert result.schema_version == "v1"
    assert len(result.frames) == 1
    assert len(result.segments) == 1
