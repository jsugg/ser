"""Tests for medium-profile encode-once inference execution."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier

import ser.config as config
from ser.models import emotion_model
from ser.repr import EncodedSequence
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_inference import (
    MediumModelUnavailableError,
    run_medium_inference,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


class _PredictModel(MLPClassifier):
    """Deterministic classifier stub for medium inference contract tests."""

    def __init__(
        self,
        *,
        predictions: list[str],
        probabilities: list[list[float]],
        classes: list[str],
    ) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self._predictions = np.asarray(predictions, dtype=object)
        self._probabilities = np.asarray(probabilities, dtype=np.float64)
        self.classes_ = np.asarray(classes, dtype=object)
        self.last_features: NDArray[np.float64] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.last_features = np.asarray(X, dtype=np.float64)
        return self._predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.last_features = np.asarray(X, dtype=np.float64)
        return self._probabilities


class _FakeBackend:
    """Deterministic backend stub that tracks encode invocation count."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_sequence(
        self,
        _audio: NDArray[np.float32],
        _sample_rate: int,
    ) -> EncodedSequence:
        self.encode_calls += 1
        return EncodedSequence(
            embeddings=np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ],
                dtype=np.float32,
            ),
            frame_start_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            frame_end_seconds=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            backend_id="hf_xlsr",
        )


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _medium_metadata(
    feature_vector_size: int = 4,
    *,
    backend_model_id: str | None = emotion_model.MEDIUM_MODEL_ID,
) -> dict[str, object]:
    """Builds minimal medium-profile artifact metadata for loader tests."""
    metadata: dict[str, object] = {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-19T00:00:00+00:00",
        "feature_vector_size": feature_vector_size,
        "training_samples": 8,
        "labels": ["happy", "sad"],
        "backend_id": "hf_xlsr",
        "profile": "medium",
        "feature_dim": feature_vector_size,
        "frame_size_seconds": 1.0,
        "frame_stride_seconds": 1.0,
        "pooling_strategy": "mean_std",
    }
    if backend_model_id is not None:
        metadata["backend_model_id"] = backend_model_id
    return metadata


def test_run_medium_inference_uses_encode_once_and_returns_schema_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium inference should encode once and return deterministic segments."""
    backend = _FakeBackend()
    model = _PredictModel(
        predictions=["happy", "happy", "sad"],
        probabilities=[[0.9, 0.1], [0.75, 0.25], [0.2, 0.8]],
        classes=["happy", "sad"],
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend", lambda **_kwargs: backend
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=model,
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(),
        ),
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        config.reload_settings(),
    )

    assert backend.encode_calls == 1
    assert result.schema_version == OUTPUT_SCHEMA_VERSION
    assert len(result.frames) == 3
    assert result.frames[0].emotion == "happy"
    assert result.frames[2].emotion == "sad"
    assert [
        (segment.emotion, segment.start_seconds, segment.end_seconds)
        for segment in result.segments
    ] == [("happy", 0.0, 2.0), ("sad", 2.0, 3.0)]
    assert [segment.confidence for segment in result.segments] == pytest.approx(
        [0.825, 0.8]
    )
    assert result.segments[0].probabilities == pytest.approx(
        {"happy": 0.825, "sad": 0.175}
    )
    assert result.segments[1].probabilities == pytest.approx({"happy": 0.2, "sad": 0.8})
    assert model.last_features is not None
    np.testing.assert_allclose(
        model.last_features,
        np.asarray(
            [
                [1.0, 2.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 0.0],
                [5.0, 6.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_run_medium_inference_fails_fast_for_non_medium_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-medium artifacts should be rejected before expensive encoding work."""
    backend = _FakeBackend()
    model = _PredictModel(
        predictions=["happy"],
        probabilities=[[1.0, 0.0]],
        classes=["happy", "sad"],
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend", lambda **_kwargs: backend
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=model,
            expected_feature_size=4,
            artifact_metadata={
                **_medium_metadata(),
                "backend_id": "handcrafted",
                "profile": "fast",
            },
        ),
    )

    with pytest.raises(MediumModelUnavailableError, match="No medium-profile model"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav",
                language="en",
                save_transcript=False,
            ),
            config.reload_settings(),
        )
    assert backend.encode_calls == 0


def test_run_medium_inference_rejects_feature_size_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature width mismatches should fail with actionable error details."""
    backend = _FakeBackend()
    model = _PredictModel(
        predictions=["happy", "sad", "sad"],
        probabilities=[[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]],
        classes=["happy", "sad"],
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.ones(8, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend", lambda **_kwargs: backend
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=model,
            expected_feature_size=8,
            artifact_metadata=_medium_metadata(feature_vector_size=8),
        ),
    )

    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav",
                language="en",
                save_transcript=False,
            ),
            config.reload_settings(),
        )
    assert backend.encode_calls == 1


def test_run_medium_inference_uses_configured_medium_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium runtime should initialize XLSR backend with configured model id."""
    monkeypatch.setenv("SER_MEDIUM_MODEL_ID", "unit-test/xlsr")
    settings = config.reload_settings()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(
                predictions=["happy"],
                probabilities=[[1.0, 0.0]],
                classes=["happy", "sad"],
            ),
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(backend_model_id="unit-test/xlsr"),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.ones(8, dtype=np.float32), 4),
    )

    class _BackendStub:
        def __init__(
            self,
            *,
            model_id: str,
            cache_dir: Path,
            device: str = "auto",
            dtype: str = "auto",
        ) -> None:
            captured["model_id"] = model_id
            captured["cache_dir"] = cache_dir
            captured["device"] = device
            captured["dtype"] = dtype

    monkeypatch.setattr("ser.runtime.medium_inference.XLSRBackend", _BackendStub)
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once",
        lambda **_kwargs: InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        ),
    )

    run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert captured["model_id"] == "unit-test/xlsr"
    assert captured["cache_dir"] == settings.models.huggingface_cache_root
    assert captured["device"] == settings.torch_runtime.device
    assert captured["dtype"] == settings.torch_runtime.dtype


def test_run_medium_inference_requires_backend_model_id_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium runtime should reject artifacts missing backend_model_id metadata."""
    monkeypatch.setenv("SER_MEDIUM_MODEL_ID", "unit-test/xlsr")
    settings = config.reload_settings()
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(
                predictions=["happy"],
                probabilities=[[1.0, 0.0]],
                classes=["happy", "sad"],
            ),
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(
                backend_model_id=None,
            ),
        ),
    )

    with pytest.raises(MediumModelUnavailableError, match="backend_model_id"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )


def test_run_medium_inference_warns_on_torch_runtime_metadata_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime should warn when artifact and runtime torch selectors differ."""
    monkeypatch.setenv("SER_TORCH_DEVICE", "cpu")
    monkeypatch.setenv("SER_TORCH_DTYPE", "float32")
    settings = config.reload_settings()
    metadata = _medium_metadata(backend_model_id=settings.models.medium_model_id)
    metadata["torch_device"] = "cuda:0"
    metadata["torch_dtype"] = "float16"
    warnings: list[str] = []
    monkeypatch.setattr(
        "ser.runtime.medium_inference.logger.warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(
                predictions=["happy", "happy", "happy"],
                probabilities=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                classes=["happy", "sad"],
            ),
            expected_feature_size=4,
            artifact_metadata=metadata,
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.ones(8, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend",
        lambda **_kwargs: _FakeBackend(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )

    run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert warnings
    assert "torch runtime selectors differ" in warnings[0]


def test_medium_single_flight_serializes_same_profile_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent medium calls should execute one-at-a-time for one profile/model tuple."""
    settings = config.reload_settings()
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(
                predictions=["happy"],
                probabilities=[[1.0, 0.0]],
                classes=["happy", "sad"],
            ),
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(
                backend_model_id=settings.models.medium_model_id
            ),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.ones(8, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )

    counters = {"active": 0, "max_active": 0}
    counter_lock = threading.Lock()

    def fake_attempt(**_kwargs: object) -> InferenceResult:
        with counter_lock:
            counters["active"] += 1
            counters["max_active"] = max(counters["max_active"], counters["active"])
        time.sleep(0.05)
        with counter_lock:
            counters["active"] -= 1
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
        )

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once",
        fake_attempt,
    )

    errors: list[Exception] = []
    request = InferenceRequest(
        file_path="sample.wav", language="en", save_transcript=False
    )

    def invoke() -> None:
        try:
            run_medium_inference(request, settings)
        except (
            Exception
        ) as err:  # pragma: no cover - defensive capture for assertion clarity
            errors.append(err)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert counters["max_active"] == 1
