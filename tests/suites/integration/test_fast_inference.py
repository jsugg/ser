"""Public fast inference behavior tests."""

from __future__ import annotations

import pickle
import threading
import time
from typing import cast

import pytest

import ser.config as config
import ser.models.training_support as training_support
from ser._internal.runtime import fast_public_boundary as fast_boundary
from ser.models import emotion_model
from ser.runtime.contracts import InferenceRequest
from ser.runtime.fast_inference import (
    FastInferenceTimeoutError,
    FastModelUnavailableError,
    run_fast_inference,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reset_ambient_settings")]


def _fast_metadata(
    *,
    backend_id: str = "handcrafted",
    profile: str = "fast",
) -> dict[str, object]:
    """Builds minimal fast-profile artifact metadata for runtime tests."""
    return {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-24T00:00:00+00:00",
        "feature_vector_size": 193,
        "training_samples": 8,
        "labels": ["happy", "sad"],
        "backend_id": backend_id,
        "profile": profile,
        "feature_dim": 193,
        "frame_size_seconds": 3.0,
        "frame_stride_seconds": 1.0,
        "pooling_strategy": "mean",
    }


def _patch_fast_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    """Patches model/detail prerequisites for fast runtime tests."""
    monkeypatch.setattr(
        fast_boundary,
        "load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=cast(training_support.EmotionClassifier, object()),
            expected_feature_size=193,
            artifact_metadata=metadata or _fast_metadata(),
        ),
    )


def test_fast_inference_returns_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast runtime should return deterministic inference schema payload."""
    settings = config.reload_settings()
    _patch_fast_prerequisites(monkeypatch)
    expected = InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])
    monkeypatch.setattr(
        fast_boundary,
        "predict_emotions_detailed",
        lambda _file_path, loaded_model=None: expected,
    )

    result = run_fast_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected


def test_fast_timeout_retries_up_to_configured_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeouts should retry up to `max_timeout_retries + 1` attempts and then fail."""
    monkeypatch.setenv("SER_FAST_MAX_TIMEOUT_RETRIES", "2")
    monkeypatch.setenv("SER_FAST_RETRY_BACKOFF_SECONDS", "0.5")
    monkeypatch.setenv("SER_FAST_TIMEOUT_SECONDS", "1.0")
    settings = config.reload_settings()
    _patch_fast_prerequisites(monkeypatch)

    calls = {"attempts": 0, "sleeps": 0}

    def fake_timeout_runner(*_args: object, **_kwargs: object) -> object:
        calls["attempts"] += 1
        raise FastInferenceTimeoutError("timeout")

    monkeypatch.setattr(fast_boundary, "_run_with_timeout_impl", fake_timeout_runner)
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleeps", calls["sleeps"] + 1),
    )

    with pytest.raises(FastInferenceTimeoutError, match="timeout"):
        run_fast_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )

    assert calls["attempts"] == 3
    assert calls["sleeps"] == 2


def test_fast_profile_pipeline_uses_process_timeout_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline fast calls should route attempts through process timeout path."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_FAST_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("SER_FAST_MAX_TIMEOUT_RETRIES", "1")
    monkeypatch.setenv("SER_FAST_RETRY_BACKOFF_SECONDS", "0.1")
    settings = config.reload_settings()

    calls = {"process": 0, "sleep": 0}
    captured: dict[str, object] = {}
    expected = InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("In-process load_model path should not run in process mode.")

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        captured.update(_kwargs)
        calls["process"] += 1
        if calls["process"] == 1:
            raise FastInferenceTimeoutError("timeout")
        return expected

    monkeypatch.setattr(fast_boundary, "load_model", fail_if_called)
    monkeypatch.setattr(fast_boundary, "_run_with_process_timeout_impl", fake_process_runner)
    monkeypatch.setattr(
        "ser.runtime.policy.time.sleep",
        lambda _delay: calls.__setitem__("sleep", calls["sleep"] + 1),
    )

    result = run_fast_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 2
    assert calls["sleep"] == 1
    worker_target = captured["worker_target"]
    assert callable(worker_target)
    qualname = getattr(worker_target, "__qualname__", "")
    assert isinstance(qualname, str)
    assert "<locals>" not in qualname
    assert pickle.dumps(worker_target)


def test_fast_inference_rejects_mismatched_artifact_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mismatched fast artifact metadata should be rejected before inference."""
    settings = config.reload_settings()
    _patch_fast_prerequisites(
        monkeypatch,
        metadata=_fast_metadata(backend_id="hf_whisper"),
    )

    with pytest.raises(FastModelUnavailableError, match="backend_id"):
        run_fast_inference(
            InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
            settings,
        )


def test_fast_single_flight_serializes_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-flight registry should serialize same-profile fast calls."""
    settings = config.reload_settings()
    _patch_fast_prerequisites(monkeypatch)

    state = {"active": 0, "max_active": 0}
    state_lock = threading.Lock()

    def fake_predict(_file_path: str, *, loaded_model: object | None = None) -> InferenceResult:
        del loaded_model
        with state_lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        try:
            time.sleep(0.05)
            return InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(fast_boundary, "predict_emotions_detailed", fake_predict)

    def invoke() -> None:
        result = run_fast_inference(
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
