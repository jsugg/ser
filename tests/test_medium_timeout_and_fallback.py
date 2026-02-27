"""Tests for medium runtime timeout/retry and fallback behavior."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

import ser.config as config
import ser.runtime.medium_inference as medium_inference
from ser.models import emotion_model
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_inference import (
    MediumInferenceExecutionError,
    MediumInferenceTimeoutError,
    MediumRuntimeDependencyError,
    MediumTransientBackendError,
    run_medium_inference,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


class _PredictModel(MLPClassifier):
    """Deterministic model stub for medium runtime tests."""

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


def _medium_metadata(feature_vector_size: int = 4) -> dict[str, object]:
    """Builds minimal medium-profile artifact metadata for runtime tests."""
    return {
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
        "backend_model_id": emotion_model.MEDIUM_MODEL_ID,
    }


def _patch_runtime_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches model/audio/backend prerequisites for retry/timeout tests."""
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend", lambda **_kwargs: object()
    )


def test_medium_timeout_retries_up_to_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should retry up to `max_retries + 1` attempts and then fail."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0.5")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(monkeypatch)

    calls = {"attempts": 0, "sleeps": 0}

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> object:
        calls["attempts"] += 1
        raise MediumInferenceTimeoutError("timeout")

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout", fake_timeout_runner
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(MediumInferenceTimeoutError, match="timeout"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 3
    assert calls["sleeps"] == 2


def test_medium_transient_backend_failure_respects_retry_upper_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient backend failures should stop after bounded retry attempts."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "2")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(monkeypatch)

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
        raise MediumTransientBackendError("transient backend failure")

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once", fake_attempt
    )

    with pytest.raises(MediumInferenceExecutionError, match="retry budget"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 3


def test_medium_non_retryable_value_error_exits_without_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature-contract value errors should not trigger retries."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "3")
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "3")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(monkeypatch)

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
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once", fake_attempt
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(ValueError, match="Feature vector size mismatch"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 1
    assert calls["sleeps"] == 0


def test_medium_dependency_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency failures should fail immediately without retry loop."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "3")
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "3")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(monkeypatch)

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
        raise MediumRuntimeDependencyError("transformers missing")

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once", fake_attempt
    )

    with pytest.raises(MediumRuntimeDependencyError, match="transformers"):
        run_medium_inference(
            InferenceRequest(
                file_path="sample.wav", language="en", save_transcript=False
            ),
            settings,
        )

    assert calls["attempts"] == 1


def test_medium_backend_setup_runs_before_timeout_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend setup should execute before timeout-wrapped compute operation."""
    settings = config.reload_settings()
    setup_calls = {"count": 0}
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )

    class _BackendStub:
        def prepare_runtime(self) -> None:
            setup_calls["count"] += 1

    backend = _BackendStub()
    monkeypatch.setattr(
        "ser.runtime.medium_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=_PredictModel(),
            expected_feature_size=4,
            artifact_metadata=_medium_metadata(),
        ),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.read_audio_file",
        lambda _file_path: (np.linspace(0.0, 1.0, 16, dtype=np.float32), 4),
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference.XLSRBackend",
        lambda **_kwargs: backend,
    )

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        assert setup_calls["count"] == 1
        return expected

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        fake_timeout_runner,
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert setup_calls["count"] == 1


def test_medium_profile_pipeline_uses_process_timeout_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline medium calls should route attempts through process timeout path."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_MEDIUM_MAX_TIMEOUT_RETRIES", "1")
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
            raise MediumInferenceTimeoutError("timeout")
        return expected

    monkeypatch.setattr("ser.runtime.medium_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_process_timeout",
        fake_process_runner,
    )
    monkeypatch.setattr(
        "ser.runtime.medium_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleep", calls["sleep"] + 1),
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 2
    assert calls["sleep"] == 1


def test_medium_profile_pipeline_allows_timeout_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline medium calls should pass zero timeout to disable budgets."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_MEDIUM_TIMEOUT_SECONDS", "0")
    settings = config.reload_settings()
    calls = {"process": 0}
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError(
            "In-process load_model path should not run in process mode."
        )

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        timeout_seconds = _kwargs.get("timeout_seconds")
        assert timeout_seconds == 0.0
        calls["process"] += 1
        return expected

    monkeypatch.setattr("ser.runtime.medium_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_process_timeout",
        fake_process_runner,
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 1


def test_medium_profile_pipeline_retries_on_cpu_after_mps_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-mode medium retries should demote to CPU after one MPS failure."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "1")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0")
    settings = config.reload_settings()

    devices: list[tuple[str, str]] = []
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )
    oom_message = (
        "MPS backend out of memory (MPS allocated: 3.13 GB, other allocations: "
        "220.17 MB, max allowed: 3.40 GB). Tried to allocate 85.83 MB on private pool."
    )

    monkeypatch.setattr(
        "ser.runtime.medium_inference._resolve_medium_feature_runtime_policy",
        lambda **_kwargs: FeatureRuntimePolicy(
            device="mps",
            dtype="float16",
            reason="test_policy",
        ),
    )

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError(
            "In-process load_model path should not run in process mode."
        )

    def fake_process_runner(
        payload: medium_inference.MediumProcessPayload,
        *,
        timeout_seconds: float,
    ) -> InferenceResult:
        del timeout_seconds
        devices.append(
            (
                payload.settings.torch_runtime.device,
                payload.settings.torch_runtime.dtype,
            )
        )
        if len(devices) == 1:
            raise MediumTransientBackendError(oom_message)
        return expected

    monkeypatch.setattr("ser.runtime.medium_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_process_timeout",
        fake_process_runner,
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert devices == [("mps", "float16"), ("cpu", "float32")]


def test_medium_in_process_rebuilds_backend_on_cpu_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process medium retries should rebuild backend on CPU after MPS failures."""
    monkeypatch.setenv("SER_MEDIUM_MAX_TRANSIENT_RETRIES", "1")
    monkeypatch.setenv("SER_MEDIUM_RETRY_BACKOFF_SECONDS", "0")
    settings = config.reload_settings()
    _patch_runtime_prerequisites(monkeypatch)

    backend_selectors: list[tuple[str, str]] = []
    calls = {"attempts": 0}
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[]
    )

    class _BackendStub:
        def __init__(self, *, device: str, dtype: str, **_kwargs: object) -> None:
            backend_selectors.append((device, dtype))

        def prepare_runtime(self) -> None:
            return None

    monkeypatch.setattr(
        "ser.runtime.medium_inference._resolve_medium_feature_runtime_policy",
        lambda **_kwargs: FeatureRuntimePolicy(
            device="mps",
            dtype="float16",
            reason="test_policy",
        ),
    )
    monkeypatch.setattr("ser.runtime.medium_inference.XLSRBackend", _BackendStub)
    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_with_timeout",
        lambda operation, timeout_seconds: operation(),
    )

    def fake_attempt(
        *,
        loaded_model: emotion_model.LoadedModel,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: object,
    ) -> InferenceResult:
        del loaded_model, backend, audio, sample_rate, runtime_config
        calls["attempts"] += 1
        if calls["attempts"] == 1:
            raise MediumTransientBackendError(
                "Input type (c10::Half) and bias type (float) should be the same"
            )
        return expected

    monkeypatch.setattr(
        "ser.runtime.medium_inference._run_medium_inference_once",
        fake_attempt,
    )

    result = run_medium_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["attempts"] == 2
    assert backend_selectors == [("mps", "float16"), ("cpu", "float32")]


def test_medium_process_timeout_applies_after_setup_phase(
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
        "ser.runtime.medium_inference.mp.get_context",
        lambda _name: _Context(),
    )
    payload = medium_inference.MediumProcessPayload(
        request=InferenceRequest(
            file_path="sample.wav", language="en", save_transcript=False
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )

    result = medium_inference._run_with_process_timeout(payload, timeout_seconds=11.0)

    assert result == expected
    assert poll_calls == [11.0]
