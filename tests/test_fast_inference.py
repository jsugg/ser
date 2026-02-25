"""Tests for fast runtime timeout/retry and policy-wrapper behavior."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from typing import cast

import pytest

import ser.config as config
import ser.runtime.fast_inference as fast_inference
from ser.models import emotion_model
from ser.runtime.contracts import InferenceRequest
from ser.runtime.fast_inference import (
    FastInferenceTimeoutError,
    run_fast_inference,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps global settings stable across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def _fast_metadata() -> dict[str, object]:
    """Builds minimal fast-profile artifact metadata for runtime tests."""
    return {
        "artifact_version": emotion_model.MODEL_ARTIFACT_VERSION,
        "artifact_schema_version": "v2",
        "created_at_utc": "2026-02-24T00:00:00+00:00",
        "feature_vector_size": 193,
        "training_samples": 8,
        "labels": ["happy", "sad"],
        "backend_id": "handcrafted",
        "profile": "fast",
        "feature_dim": 193,
        "frame_size_seconds": 3.0,
        "frame_stride_seconds": 1.0,
        "pooling_strategy": "mean",
    }


def _patch_fast_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patches model/detail prerequisites for fast runtime tests."""
    monkeypatch.setattr(
        "ser.runtime.fast_inference.load_model",
        lambda **_kwargs: emotion_model.LoadedModel(
            model=cast(emotion_model.EmotionClassifier, object()),
            expected_feature_size=193,
            artifact_metadata=_fast_metadata(),
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
        "ser.runtime.fast_inference.predict_emotions_detailed",
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

    monkeypatch.setattr(
        "ser.runtime.fast_inference._run_with_timeout",
        fake_timeout_runner,
    )
    monkeypatch.setattr(
        "ser.runtime.fast_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
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
    settings = config.reload_settings()

    calls = {"process": 0, "sleep": 0}
    expected = InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("In-process load_model path should not run in process mode.")

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        calls["process"] += 1
        if calls["process"] == 1:
            raise FastInferenceTimeoutError("timeout")
        return expected

    monkeypatch.setattr("ser.runtime.fast_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.fast_inference._run_with_process_timeout",
        fake_process_runner,
    )
    monkeypatch.setattr(
        "ser.runtime.fast_inference._retry_delay_seconds",
        lambda **_kwargs: 0.1,
    )
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


def test_fast_profile_pipeline_allows_timeout_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profile-pipeline fast calls should pass zero timeout to disable budgets."""
    monkeypatch.setenv("SER_ENABLE_PROFILE_PIPELINE", "true")
    monkeypatch.setenv("SER_FAST_PROCESS_ISOLATION", "true")
    monkeypatch.setenv("SER_FAST_TIMEOUT_SECONDS", "0")
    settings = config.reload_settings()
    expected = InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])
    calls = {"process": 0}

    def fail_if_called(**_kwargs: object) -> object:
        raise AssertionError("In-process load_model path should not run in process mode.")

    def fake_process_runner(*_args: object, **_kwargs: object) -> InferenceResult:
        timeout_seconds = _kwargs.get("timeout_seconds")
        assert timeout_seconds == 0.0
        calls["process"] += 1
        return expected

    monkeypatch.setattr("ser.runtime.fast_inference.load_model", fail_if_called)
    monkeypatch.setattr(
        "ser.runtime.fast_inference._run_with_process_timeout",
        fake_process_runner,
    )

    result = run_fast_inference(
        InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings,
    )

    assert result == expected
    assert calls["process"] == 1


def test_fast_process_timeout_applies_after_setup_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process timeout budget should start only after setup phase is acknowledged."""
    expected = InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])
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

        def Pipe(self, duplex: bool = False) -> tuple[_ParentConnection, _ChildConnection]:  # noqa: N802
            del duplex
            return self._parent, self._child

        def Process(self, **_kwargs: object) -> _Process:  # noqa: N802
            return self._process

    monkeypatch.setattr(
        "ser.runtime.fast_inference.mp.get_context",
        lambda _name: _Context(),
    )
    payload = fast_inference.FastProcessPayload(
        request=InferenceRequest(file_path="sample.wav", language="en", save_transcript=False),
        settings=settings,
    )

    result = fast_inference._run_with_process_timeout(payload, timeout_seconds=5.0)

    assert result == expected
    assert poll_calls == [5.0]


def test_fast_single_flight_serializes_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent fast calls should execute one-at-a-time for one profile tuple."""
    settings = config.reload_settings()
    _patch_fast_prerequisites(monkeypatch)
    monkeypatch.setattr(
        "ser.runtime.fast_inference._run_with_timeout",
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
        return InferenceResult(schema_version=OUTPUT_SCHEMA_VERSION, segments=[], frames=[])

    monkeypatch.setattr(
        "ser.runtime.fast_inference._run_fast_inference_once",
        fake_attempt,
    )

    errors: list[Exception] = []
    request = InferenceRequest(file_path="sample.wav", language="en", save_transcript=False)

    def invoke() -> None:
        try:
            run_fast_inference(request, settings)
        except Exception as err:  # pragma: no cover - defensive capture for assertion clarity
            errors.append(err)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert counters["max_active"] == 1
