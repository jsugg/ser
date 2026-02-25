"""Tests for accurate runtime timeout/retry and fallback behavior."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

import ser.config as config
import ser.runtime.accurate_inference as accurate_inference
from ser.models import emotion_model
from ser.runtime.accurate_inference import (
    AccurateInferenceExecutionError,
    AccurateInferenceTimeoutError,
    AccurateModelUnavailableError,
    AccurateRuntimeDependencyError,
    AccurateTransientBackendError,
    run_accurate_inference,
)
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


class _PredictModel(MLPClassifier):
    """Deterministic model stub for accurate runtime tests."""

    def __init__(self) -> None:
        super().__init__(hidden_layer_sizes=(1,), max_iter=1, random_state=0)
        self.classes_ = np.asarray(["happy", "sad"], dtype=object)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray(["happy"] * int(X.shape[0]), dtype=object)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.asarray([[0.9, 0.1]] * int(X.shape[0]), dtype=np.float64)


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _accurate_metadata(
    feature_vector_size: int = 4,
    *,
    backend_model_id: str | None = emotion_model.ACCURATE_MODEL_ID,
) -> dict[str, object]:
    """Builds minimal accurate-profile artifact metadata for runtime tests."""
    metadata: dict[str, object] = {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-21T00:00:00+00:00",
        "feature_vector_size": feature_vector_size,
        "training_samples": 8,
        "labels": ["happy", "sad"],
        "backend_id": "hf_whisper",
        "profile": "accurate",
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
) -> None:
    """Patches model/audio/backend prerequisites for retry/timeout tests."""
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=_accurate_metadata(backend_model_id=backend_model_id),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.WhisperBackend",
        lambda **_kwargs: object(),
    )


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

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout", fake_timeout_runner
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(AccurateInferenceTimeoutError, match="timeout"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 3
    assert calls["sleeps"] == 2


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

    def fake_attempt(
        *,
        loaded_model: emotion_model.LoadedModel,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: object,
    ) -> object:
        del loaded_model, backend, audio, sample_rate, runtime_config
        calls["attempts"] += 1
        raise AccurateTransientBackendError("transient backend failure")

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_accurate_inference_once", fake_attempt
    )

    with pytest.raises(AccurateInferenceExecutionError, match="retry budget"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 3


def test_accurate_non_retryable_value_error_exits_without_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature-contract value errors should not trigger retries."""
    monkeypatch.setenv("SER_ACCURATE_MAX_TIMEOUT_RETRIES", "3")
    monkeypatch.setenv("SER_ACCURATE_MAX_TRANSIENT_RETRIES", "3")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    calls = {"attempts": 0, "sleeps": 0}

    def fake_attempt(
        *,
        loaded_model: emotion_model.LoadedModel,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: object,
    ) -> object:
        del loaded_model, backend, audio, sample_rate, runtime_config
        calls["attempts"] += 1
        raise ValueError("Feature vector size mismatch")

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_accurate_inference_once", fake_attempt
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 1
    assert calls["sleeps"] == 0


def test_accurate_dependency_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency failures should fail immediately without retry loop."""
    monkeypatch.setenv("SER_ACCURATE_MAX_TIMEOUT_RETRIES", "3")
    monkeypatch.setenv("SER_ACCURATE_MAX_TRANSIENT_RETRIES", "3")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )

    calls = {"attempts": 0}

    def fake_attempt(
        *,
        loaded_model: emotion_model.LoadedModel,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: object,
    ) -> object:
        del loaded_model, backend, audio, sample_rate, runtime_config
        calls["attempts"] += 1
        raise AccurateRuntimeDependencyError("transformers missing")

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_accurate_inference_once", fake_attempt
    )

    with pytest.raises(AccurateRuntimeDependencyError, match="transformers"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 1


def test_accurate_inference_rejects_non_accurate_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate runtime should reject artifacts with non-accurate metadata."""
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata={
                **_accurate_metadata(),
                "backend_id": "hf_xlsr",
                "profile": "medium",
            },
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )

    with pytest.raises(AccurateModelUnavailableError, match="hf_whisper"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            config.reload_settings(),
        )


def test_accurate_inference_returns_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate runtime should return deterministic inference schema payload."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_accurate_inference_once",
        lambda **_kwargs: expected,
    )

    result = run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected


def test_accurate_backend_setup_runs_before_timeout_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend setup should execute before timeout-wrapped compute operation."""
    settings = config.reload_settings()
    backend_model_id = settings.models.accurate_model_id
    setup_calls = {"count": 0}
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )

    class _BackendStub:
        def prepare_runtime(self) -> None:
            setup_calls["count"] += 1

    backend = _BackendStub()
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=_accurate_metadata(backend_model_id=backend_model_id),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._build_backend_for_profile",
        lambda **_kwargs: backend,
    )

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        assert setup_calls["count"] == 1
        return expected

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        fake_timeout_runner,
    )

    result = run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert setup_calls["count"] == 1


def test_accurate_inference_uses_configured_accurate_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend initialization should honor the configured accurate model id."""
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-tiny")
    settings = config.reload_settings()
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=_accurate_metadata(
                backend_model_id=settings.models.accurate_model_id
            ),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    captured: dict[str, object] = {}

    class _BackendStub:
        def __init__(self, *, model_id: str, cache_dir: Path) -> None:
            captured["model_id"] = model_id
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr("ser.runtime.accurate_inference.WhisperBackend", _BackendStub)
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )

    def _fake_run_once(**kwargs: object) -> InferenceResult:
        captured["backend"] = kwargs["backend"]
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_accurate_inference_once",
        _fake_run_once,
    )

    run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert captured["model_id"] == "unit-test/whisper-tiny"
    assert captured["cache_dir"] == settings.models.huggingface_cache_root
    assert isinstance(captured["backend"], _BackendStub)


def test_accurate_inference_rejects_mismatched_backend_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime should reject artifacts with backend_model_id mismatch."""
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "unit-test/whisper-large")
    settings = config.reload_settings()
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=768,
            artifact_metadata=_accurate_metadata(
                feature_vector_size=768,
                backend_model_id="unit-test/whisper-tiny",
            ),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    with pytest.raises(AccurateModelUnavailableError, match="backend_model_id"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )


def test_accurate_inference_requires_backend_model_id_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifacts missing backend_model_id should be rejected by strict checks."""
    monkeypatch.setenv("SER_ACCURATE_MODEL_ID", "openai/whisper-tiny")
    settings = config.reload_settings()
    monkeypatch.setattr(
        "ser.runtime.accurate_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=768,
            artifact_metadata=_accurate_metadata(
                feature_vector_size=768,
                backend_model_id=None,
            ),
        ),
    )
    with pytest.raises(AccurateModelUnavailableError, match="backend_model_id"):
        run_accurate_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )


def test_accurate_single_flight_serializes_same_profile_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent accurate calls should execute one-at-a-time for one profile/model tuple."""
    settings = config.reload_settings()
    _patch_runtime_prerequisites(
        monkeypatch,
        backend_model_id=settings.models.accurate_model_id,
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )

    counters = {"active": 0, "max_active": 0}
    counter_lock = threading.Lock()

    def fake_attempt(
        *,
        loaded_model: emotion_model.LoadedModel,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: object,
    ) -> InferenceResult:
        del loaded_model, backend, audio, sample_rate, runtime_config
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
        "ser.runtime.accurate_inference._run_accurate_inference_once",
        fake_attempt,
    )

    errors: list[Exception] = []
    request = InferenceRequest(
        file_path="sample.wav", language="en", save_transcript=False
    )

    def invoke() -> None:
        try:
            run_accurate_inference(request, settings)
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


def test_accurate_profile_pipeline_uses_process_timeout_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline accurate calls should route attempts through process timeout path."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ACCURATE_MAX_TIMEOUT_RETRIES", "1")
    settings = config.reload_settings()

    calls = {"process": 0, "sleep": 0}
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError(
            "In-process load_model path should not run in process mode."
        )

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        calls["process"] += 1
        if calls["process"] == 1:
            raise AccurateInferenceTimeoutError("timeout")
        return expected

    monkeypatch.setattr("ser.runtime.accurate_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_process_timeout",
        fake_process_runner,
    )
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleep", calls["sleep"] + 1),
    )

    result = run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 2
    assert calls["sleep"] == 1


def test_accurate_profile_pipeline_allows_timeout_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline accurate calls should pass zero timeout to disable budgets."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_ACCURATE_TIMEOUT_SECONDS", "0")
    settings = config.reload_settings()
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )
    calls = {"process": 0}

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError(
            "In-process load_model path should not run in process mode."
        )

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        timeout_seconds = _kwargs.get("timeout_seconds")
        assert timeout_seconds == 0.0
        calls["process"] += 1
        return expected

    monkeypatch.setattr("ser.runtime.accurate_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.accurate_inference._run_with_process_timeout",
        fake_process_runner,
    )

    result = run_accurate_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 1


def test_accurate_process_timeout_applies_after_setup_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process timeout budget should start only after setup phase is acknowledged."""
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )
    poll_calls: list[float | None] = []
    settings = config.reload_settings()

    class _ParentConnection:
        def __init__(self) -> None:
            self._messages: list[tuple[object, ...]] = [
                ("phase", "setup_complete"),
                ("ok", expected),
            ]

        def recv(self) -> tuple[object, ...]:
            if not self._messages:
                raise EOFError
            return self._messages.pop(0)

        def poll(self, timeout: float | None = None) -> bool:
            poll_calls.append(timeout)
            return True

        def close(self) -> None:
            return None

    class _ChildConnection:
        def close(self) -> None:
            return None

    class _Process:
        def __init__(self) -> None:
            self.exitcode = 0
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def kill(self) -> None:
            self._alive = False

    class _Context:
        def __init__(self) -> None:
            self._parent = _ParentConnection()
            self._child = _ChildConnection()
            self._process = _Process()

        def Pipe(
            self, duplex: bool = False
        ) -> tuple[_ParentConnection, _ChildConnection]:  # noqa: N802
            del duplex
            return self._parent, self._child

        def Process(self, **_kwargs: object) -> _Process:  # noqa: N802
            return self._process

    monkeypatch.setattr(
        "ser.runtime.accurate_inference.mp.get_context",
        lambda _name: _Context(),
    )
    payload = accurate_inference.AccurateProcessPayload(
        request=InferenceRequest(
            file_path="sample.wav", language="en", save_transcript=False
        ),
        settings=settings,
        expected_backend_id="hf_whisper",
        expected_profile="accurate",
        expected_backend_model_id=settings.models.accurate_model_id,
    )

    result = accurate_inference._run_with_process_timeout(payload, timeout_seconds=7.0)

    assert result == expected
    assert poll_calls == [7.0]
