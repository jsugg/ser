"""Delegation contracts for shared runtime worker lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing.connection import Connection
from types import SimpleNamespace
from typing import cast

import pytest

import ser.runtime.accurate_inference as accurate_inference
import ser.runtime.fast_inference as fast_inference
import ser.runtime.medium_inference as medium_inference
from ser import config
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


def test_fast_recv_worker_message_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fast wrapper should delegate with stable label/error wiring."""
    captured: dict[str, object] = {}
    expected = ("phase", "setup_complete")

    def _fake_impl(**kwargs: object) -> tuple[str, str]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(fast_inference, "_recv_worker_message_impl", _fake_impl)
    connection = cast(Connection, SimpleNamespace())

    resolved = fast_inference._recv_worker_message(connection, stage="setup")

    assert resolved == expected
    assert captured["connection"] is connection
    assert captured["stage"] == "setup"
    assert captured["worker_label"] == "Fast inference"
    assert captured["error_factory"] is fast_inference.FastInferenceExecutionError


def test_medium_recv_worker_message_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium wrapper should delegate with stable label/error wiring."""
    captured: dict[str, object] = {}
    expected = ("phase", "setup_complete")

    def _fake_impl(**kwargs: object) -> tuple[str, str]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(medium_inference, "_recv_worker_message_impl", _fake_impl)
    connection = cast(Connection, SimpleNamespace())

    resolved = medium_inference._recv_worker_message(connection, stage="setup")

    assert resolved == expected
    assert captured["connection"] is connection
    assert captured["stage"] == "setup"
    assert captured["worker_label"] == "Medium inference"
    assert captured["error_factory"] is medium_inference.MediumInferenceExecutionError


def test_accurate_recv_worker_message_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate wrapper should delegate with stable label/error wiring."""
    captured: dict[str, object] = {}
    expected = ("phase", "setup_complete")

    def _fake_impl(**kwargs: object) -> tuple[str, str]:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(accurate_inference, "_recv_worker_message_impl", _fake_impl)
    connection = cast(Connection, SimpleNamespace())

    resolved = accurate_inference._recv_worker_message(connection, stage="setup")

    assert resolved == expected
    assert captured["connection"] is connection
    assert captured["stage"] == "setup"
    assert captured["worker_label"] == "Accurate inference"
    assert captured["error_factory"] is accurate_inference.AccurateInferenceExecutionError


def test_medium_raise_worker_error_delegates_to_internal_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium worker-error wrapper should delegate with stable mapping wiring."""
    captured: dict[str, object] = {}

    def _fake_impl(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(medium_inference, "_raise_worker_error_impl", _fake_impl)
    medium_inference._raise_worker_error("ValueError", "bad payload")

    assert captured["error_type"] == "ValueError"
    assert captured["message"] == "bad payload"
    assert captured["worker_label"] == "Medium inference"
    assert captured["unknown_error_factory"] is medium_inference.MediumInferenceExecutionError
    known_error_factories = cast(dict[str, object], captured["known_error_factories"])
    assert "MediumTransientBackendError" in known_error_factories
    assert "MediumInferenceTimeoutError" in known_error_factories


def test_medium_process_timeout_delegates_setup_compute_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium process-timeout wrapper should delegate handshake orchestration."""
    captured: dict[str, object] = {}
    sentinel_context = cast(object, SimpleNamespace())
    settings = config.reload_settings()
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )

    def _fake_impl(**kwargs: object) -> tuple[str, InferenceResult]:
        captured.update(kwargs)
        on_setup_complete = cast(Callable[[], None], kwargs["on_setup_complete"])
        on_setup_complete()
        return ("ok", expected)

    monkeypatch.setattr(
        medium_inference,
        "_run_process_setup_compute_handshake_impl",
        _fake_impl,
    )
    monkeypatch.setattr(
        medium_inference.mp,
        "get_context",
        lambda _name: sentinel_context,
    )
    payload = medium_inference.MediumProcessPayload(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )

    resolved = medium_inference._run_with_process_timeout(payload, timeout_seconds=7.0)

    assert resolved == expected
    assert captured["context"] is sentinel_context
    assert captured["worker_label"] == "Medium inference"
    assert captured["timeout_seconds"] == 7.0
    assert captured["timeout_error_factory"] is medium_inference.MediumInferenceTimeoutError
    assert captured["execution_error_factory"] is medium_inference.MediumInferenceExecutionError
    assert captured["worker_target"] is medium_inference._worker_entry


def test_accurate_process_timeout_delegates_setup_compute_handshake(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate process-timeout wrapper should delegate handshake orchestration."""
    captured: dict[str, object] = {}
    sentinel_context = cast(object, SimpleNamespace())
    settings = config.reload_settings()
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )

    def _fake_impl(**kwargs: object) -> tuple[str, InferenceResult]:
        captured.update(kwargs)
        on_setup_complete = cast(Callable[[], None], kwargs["on_setup_complete"])
        on_setup_complete()
        return ("ok", expected)

    monkeypatch.setattr(
        accurate_inference,
        "_run_process_setup_compute_handshake_impl",
        _fake_impl,
    )
    monkeypatch.setattr(
        accurate_inference.mp,
        "get_context",
        lambda _name: sentinel_context,
    )
    payload = accurate_inference.AccurateProcessPayload(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_id="hf_whisper",
        expected_profile="accurate",
        expected_backend_model_id=settings.models.accurate_model_id,
    )

    resolved = accurate_inference._run_with_process_timeout(payload, timeout_seconds=7.0)

    assert resolved == expected
    assert captured["context"] is sentinel_context
    assert captured["worker_label"] == "Accurate inference"
    assert captured["timeout_seconds"] == 7.0
    assert captured["timeout_error_factory"] is accurate_inference.AccurateInferenceTimeoutError
    assert captured["execution_error_factory"] is accurate_inference.AccurateInferenceExecutionError
    assert captured["worker_target"] is accurate_inference._worker_entry
