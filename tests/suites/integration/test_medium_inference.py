"""Public medium inference behavior tests."""

from __future__ import annotations

import pickle
import threading
import time

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier

import ser.config as config
from ser._internal.runtime import medium_public_boundary as medium_boundary
from ser.models import emotion_model
from ser.repr import EncodedSequence
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_inference import (
    MediumInferenceExecutionError,
    MediumInferenceTimeoutError,
    MediumModelUnavailableError,
    MediumRuntimeDependencyError,
    MediumTransientBackendError,
    run_medium_inference,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reset_ambient_settings")]


class _PredictModel(MLPClassifier):
    """Deterministic classifier stub for medium inference contract tests."""

    def __init__(self) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self.classes_ = np.asarray(["happy", "sad"], dtype=object)
        self.last_features: NDArray[np.float64] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        self.last_features = np.asarray(X, dtype=np.float64)
        return np.asarray(["happy"] * int(X.shape[0]), dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        self.last_features = np.asarray(X, dtype=np.float64)
        return np.asarray([[0.9, 0.1]] * int(X.shape[0]), dtype=np.float64)


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


def _medium_metadata(
    feature_vector_size: int = 4,
    *,
    backend_model_id: str | None = emotion_model.MEDIUM_MODEL_ID,
    backend_id: str = "hf_xlsr",
    profile: str = "medium",
) -> dict[str, object]:
    """Builds minimal medium-profile artifact metadata for loader tests."""
    metadata: dict[str, object] = {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-19T00:00:00+00:00",
        "feature_vector_size": feature_vector_size,
        "training_samples": 8,
        "labels": ["happy", "sad"],
        "backend_id": backend_id,
        "profile": profile,
        "feature_dim": feature_vector_size,
        "frame_size_seconds": 1.0,
        "frame_stride_seconds": 1.0,
        "pooling_strategy": "mean_std",
    }
    if backend_model_id is not None:
        metadata["backend_model_id"] = backend_model_id
    return metadata


def _patch_runtime_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend_model_id: str,
    backend: _FakeBackend | None = None,
    metadata: dict[str, object] | None = None,
) -> _FakeBackend:
    """Patches model/audio/backend prerequisites for medium runtime tests."""
    resolved_backend = backend or _FakeBackend()
    active_metadata = metadata or _medium_metadata(backend_model_id=backend_model_id)
    monkeypatch.setattr(
        medium_boundary,
        "load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=active_metadata,
        ),
    )
    monkeypatch.setattr(
        medium_boundary,
        "read_audio_file",
        lambda _file_path, *, audio_read_config=None: (
            np.linspace(0.0, 1.0, 16, dtype=np.float32),
            4,
        ),
    )
    monkeypatch.setattr(medium_boundary, "XLSRBackend", lambda **_kwargs: resolved_backend)
    return resolved_backend


def test_run_medium_inference_uses_encode_once_and_returns_schema_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium inference should encode once and return deterministic segments."""
    settings = config.reload_settings()
    backend = _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert backend.encode_calls == 1
    assert result.schema_version == OUTPUT_SCHEMA_VERSION
    assert len(result.frames) == 3
    assert [frame.emotion for frame in result.frames] == ["happy", "happy", "happy"]


def test_run_medium_inference_fails_fast_for_non_medium_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-medium artifacts should be rejected before expensive encoding work."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
        metadata=_medium_metadata(
            backend_model_id=settings.models.medium_model_id,
            backend_id="hf_whisper",
        ),
    )

    with pytest.raises(MediumModelUnavailableError, match="backend_id"):
        run_medium_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )


def test_medium_timeout_retries_up_to_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should retry up to `max_retries + 1` attempts and then fail."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0.5")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
    )

    calls = {"attempts": 0, "sleeps": 0}

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> object:
        calls["attempts"] += 1
        raise MediumInferenceTimeoutError("timeout")

    monkeypatch.setattr(medium_boundary, "_run_with_timeout_impl", fake_timeout_runner)
    monkeypatch.setattr(medium_boundary, "retry_delay_seconds", lambda **_kwargs: 0.1)
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(MediumInferenceTimeoutError, match="timeout"):
        run_medium_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 3
    assert calls["sleeps"] == 2


def test_medium_process_isolation_uses_spawn_safe_worker_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-isolated medium calls should pass a top-level picklable worker target."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_MEDIUM_PROCESS_ISOLATION", "true")
    settings = config.reload_settings()
    captured: dict[str, object] = {}
    expected = medium_boundary.InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("In-process load_model path should not run in process mode.")

    def fake_process_runner(*_args: object, **kwargs: object) -> medium_boundary.InferenceResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(medium_boundary, "load_model", fail_if_called)
    monkeypatch.setattr(medium_boundary, "_run_with_process_timeout_impl", fake_process_runner)

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    worker_target = captured["worker_target"]
    assert result == expected
    assert callable(worker_target)
    qualname = getattr(worker_target, "__qualname__", "")
    assert isinstance(qualname, str)
    assert "<locals>" not in qualname
    assert pickle.dumps(worker_target)


def test_medium_dependency_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency failures should bypass retry policy."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "4")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
    )

    calls = {"attempts": 0}

    def fake_attempt(**_kwargs: object) -> object:
        calls["attempts"] += 1
        raise MediumRuntimeDependencyError("missing runtime dependency")

    monkeypatch.setattr(
        medium_boundary,
        "_run_with_timeout_impl",
        lambda **kwargs: kwargs["operation"](),
    )
    monkeypatch.setattr(medium_boundary, "run_medium_inference_once", fake_attempt)

    with pytest.raises(MediumRuntimeDependencyError, match="missing runtime dependency"):
        run_medium_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 1


def test_medium_transient_failure_respects_retry_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient backend failures should stop after bounded retry attempts."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
    )

    calls = {"attempts": 0}

    def fake_attempt(**_kwargs: object) -> object:
        calls["attempts"] += 1
        raise MediumTransientBackendError("transient backend failure")

    monkeypatch.setattr(
        medium_boundary,
        "_run_with_timeout_impl",
        lambda **kwargs: kwargs["operation"](),
    )
    monkeypatch.setattr(medium_boundary, "run_medium_inference_once", fake_attempt)

    with pytest.raises(MediumInferenceExecutionError, match="retry budget"):
        run_medium_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 3


def test_medium_single_flight_serializes_same_profile_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-flight registry should serialize same-profile medium calls."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.medium_model_id,
    )
    monkeypatch.setattr(
        medium_boundary,
        "_run_with_timeout_impl",
        lambda **kwargs: kwargs["operation"](),
    )

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def fake_attempt(**_kwargs: object) -> object:
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            return medium_boundary.InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(medium_boundary, "run_medium_inference_once", fake_attempt)

    def invoke() -> None:
        result = run_medium_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )
        assert result.schema_version == OUTPUT_SCHEMA_VERSION

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["max_active"] == 1
