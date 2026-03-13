"""Public accurate inference behavior tests."""

from __future__ import annotations

import pickle
import threading
import time

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

import ser.config as config
from ser._internal.runtime import accurate_public_boundary as accurate_boundary
from ser.models import emotion_model
from ser.repr import EncodedSequence
from ser.runtime.accurate_inference import (
    AccurateInferenceExecutionError,
    AccurateInferenceTimeoutError,
    AccurateModelUnavailableError,
    AccurateRuntimeDependencyError,
    AccurateTransientBackendError,
    run_accurate_inference,
)
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reset_ambient_settings")]


class _PredictModel(MLPClassifier):
    """Deterministic model stub for accurate runtime tests."""

    def __init__(self) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self.classes_ = np.asarray(["happy", "sad"], dtype=object)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(["happy"] * int(X.shape[0]), dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray([[0.9, 0.1]] * int(X.shape[0]), dtype=np.float64)


class _FakeBackend:
    """Deterministic accurate backend stub."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def encode_sequence(self, _audio: np.ndarray, _sample_rate: int) -> EncodedSequence:
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
            backend_id="hf_whisper",
        )


def _accurate_metadata(
    feature_vector_size: int = 4,
    *,
    backend_model_id: str | None = emotion_model.ACCURATE_MODEL_ID,
    backend_id: str = "hf_whisper",
    profile: str = "accurate",
) -> dict[str, object]:
    """Builds minimal accurate-profile artifact metadata for runtime tests."""
    metadata: dict[str, object] = {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-21T00:00:00+00:00",
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
    """Patches model/audio/backend prerequisites for accurate runtime tests."""
    resolved_backend = backend or _FakeBackend()
    active_metadata = metadata or _accurate_metadata(backend_model_id=backend_model_id)
    monkeypatch.setattr(
        accurate_boundary,
        "load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=active_metadata,
        ),
    )
    monkeypatch.setattr(
        accurate_boundary,
        "read_audio_file",
        lambda _file_path, *, audio_read_config=None: (
            np.linspace(0.0, 1.0, 16, dtype=np.float32),
            4,
        ),
    )
    monkeypatch.setattr(
        accurate_boundary,
        "WhisperBackend",
        lambda **_kwargs: resolved_backend,
    )
    return resolved_backend


def test_accurate_timeout_retries_up_to_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should retry up to `max_retries + 1` attempts and then fail."""
    monkeypatch.setenv("SER_ACCURATE_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_ACCURATE_RETRY_BACKOFF_SECONDS", "0.5")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    calls = {"attempts": 0, "sleeps": 0}

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> object:
        calls["attempts"] += 1
        raise AccurateInferenceTimeoutError("timeout")

    monkeypatch.setattr(accurate_boundary, "_run_with_timeout_impl", fake_timeout_runner)
    monkeypatch.setattr(
        accurate_boundary,
        "retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(AccurateInferenceTimeoutError, match="timeout"):
        run_accurate_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 3
    assert calls["sleeps"] == 2


def test_accurate_process_isolation_uses_spawn_safe_worker_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-isolated accurate calls should pass a top-level picklable worker target."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ACCURATE_PROCESS_ISOLATION", "true")
    settings = config.reload_settings()
    captured: dict[str, object] = {}
    expected = accurate_boundary.InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("In-process load_model path should not run in process mode.")

    def fake_process_runner(*_args: object, **kwargs: object) -> accurate_boundary.InferenceResult:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(accurate_boundary, "load_model", fail_if_called)
    monkeypatch.setattr(accurate_boundary, "_run_with_process_timeout_impl", fake_process_runner)

    result = run_accurate_inference(
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


def test_accurate_transient_backend_failure_respects_retry_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient backend failures should stop after bounded retry attempts."""
    monkeypatch.setenv("SER_ACCURATE_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setenv("SER_ACCURATE_RETRY_BACKOFF_SECONDS", "0")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    calls = {"attempts": 0}

    def fake_attempt(**_kwargs: object) -> object:
        calls["attempts"] += 1
        raise AccurateTransientBackendError("transient backend failure")

    monkeypatch.setattr(
        accurate_boundary,
        "_run_with_timeout_impl",
        lambda **kwargs: kwargs["operation"](),
    )
    monkeypatch.setattr(accurate_boundary, "run_accurate_inference_once", fake_attempt)

    with pytest.raises(AccurateInferenceExecutionError, match="retry budget"):
        run_accurate_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 3


def test_accurate_dependency_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency failures should bypass retry policy."""
    monkeypatch.setenv("SER_ACCURATE_MAX_TRANSIENT_RETRIES", "4")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    calls = {"attempts": 0}

    def fake_attempt(**_kwargs: object) -> object:
        calls["attempts"] += 1
        raise AccurateRuntimeDependencyError("missing runtime dependency")

    monkeypatch.setattr(
        accurate_boundary,
        "_run_with_timeout_impl",
        lambda **kwargs: kwargs["operation"](),
    )
    monkeypatch.setattr(accurate_boundary, "run_accurate_inference_once", fake_attempt)

    with pytest.raises(AccurateRuntimeDependencyError, match="missing runtime dependency"):
        run_accurate_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 1


def test_accurate_inference_returns_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate runtime should return deterministic inference schema payload."""
    settings = config.reload_settings()
    backend = _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    result = run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert backend.encode_calls == 1
    assert result.schema_version == OUTPUT_SCHEMA_VERSION
    assert len(result.frames) == 3
    assert [frame.emotion for frame in result.frames] == ["happy", "happy", "happy"]


def test_accurate_inference_rejects_non_accurate_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched artifact metadata should be rejected before inference."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
        metadata=_accurate_metadata(
            backend_model_id=settings.models.accurate_model_id,
            backend_id="hf_xlsr",
        ),
    )

    with pytest.raises(AccurateModelUnavailableError, match="backend_id"):
        run_accurate_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )


def test_accurate_single_flight_serializes_same_profile_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-flight registry should serialize same-profile accurate calls."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )
    monkeypatch.setattr(
        accurate_boundary,
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
            return accurate_boundary.InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(accurate_boundary, "run_accurate_inference_once", fake_attempt)

    def invoke() -> None:
        result = run_accurate_inference(
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
