"""Tests for accurate execution-flow orchestration helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

import ser.config as config
from ser.repr import EncodedSequence, FeatureBackend, PoolingWindow
from ser.runtime.accurate_execution_flow import (
    execute_accurate_inference_with_retry,
    run_accurate_retryable_operation,
)
from ser.runtime.accurate_worker_operation import (
    AccurateRetryOperationState,
    PreparedAccurateOperation,
)
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult

pytestmark = pytest.mark.unit


class _BackendStub(FeatureBackend):
    """Minimal feature-backend stub for orchestration tests."""

    @property
    def backend_id(self) -> str:
        return "hf_whisper"

    @property
    def feature_dim(self) -> int:
        return 4

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        del audio, sample_rate
        return EncodedSequence(
            embeddings=np.ones((1, 4), dtype=np.float32),
            frame_start_seconds=np.asarray([0.0], dtype=np.float64),
            frame_end_seconds=np.asarray([1.0], dtype=np.float64),
            backend_id=self.backend_id,
        )

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> NDArray[np.float64]:
        del encoded, windows
        return np.ones((1, 4), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ModelStub:
    expected_feature_size: int | None = 4


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    settings: config.AppConfig


def test_run_accurate_retryable_operation_forwards_retry_state() -> None:
    """Retryable accurate helper should forward the current retry state."""
    backend = _BackendStub()
    settings = config.reload_settings()
    prepared = PreparedAccurateOperation(
        loaded_model=_ModelStub(),
        backend=backend,
        audio=np.ones(4, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=settings.accurate_runtime,
    )
    retry_state = AccurateRetryOperationState[_PayloadStub, _BackendStub](
        process_payload=None,
        active_backend=backend,
    )
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )
    captured: dict[str, object] = {}

    def _run_inference_operation(**kwargs: object) -> InferenceResult:
        captured.update(kwargs)
        return expected

    result = run_accurate_retryable_operation(
        enforce_timeout=True,
        use_process_isolation=False,
        retry_state=retry_state,
        prepared_operation=prepared,
        timeout_seconds=11.0,
        expected_profile="accurate",
        logger=logging.getLogger("ser.tests.accurate_execution_flow"),
        inference_phase_name="emotion_inference",
        log_phase_started=lambda *args, **kwargs: None,
        log_phase_completed=lambda *args, **kwargs: None,
        log_phase_failed=lambda *args, **kwargs: None,
        run_with_process_timeout=lambda *args, **kwargs: expected,
        run_accurate_inference_once=lambda **kwargs: expected,
        run_with_timeout=lambda **kwargs: expected,
        run_inference_operation=_run_inference_operation,
        timeout_error_factory=RuntimeError,
        runtime_error_factory=RuntimeError,
    )

    assert result == expected
    assert captured["enforce_timeout"] is True
    assert captured["use_process_isolation"] is False
    assert captured["process_payload"] is None
    assert captured["prepared_operation"] is prepared
    assert captured["active_backend"] is backend
    assert captured["timeout_seconds"] == 11.0
    assert captured["expected_profile"] == "accurate"
    assert captured["inference_phase_name"] == "emotion_inference"


def test_execute_accurate_inference_with_retry_wires_retry_and_setup() -> None:
    """Execution-flow helper should finalize setup and wire retry policy inputs."""
    settings = config.reload_settings()
    backend = _BackendStub()
    retry_state = AccurateRetryOperationState[_PayloadStub, _BackendStub](
        process_payload=None,
        active_backend=backend,
    )
    expected = InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=[],
        frames=[],
    )
    captured: dict[str, object] = {}

    def _run_retry_policy(
        *,
        operation: Callable[[], InferenceResult],
        **kwargs: object,
    ) -> InferenceResult:
        captured["retry_policy"] = kwargs
        captured["operation_result"] = operation()
        return expected

    def _build_transient_failure_handler(
        **kwargs: object,
    ) -> Callable[..., None]:
        captured["transient"] = kwargs
        return lambda *_args: None

    result = execute_accurate_inference_with_retry(
        use_process_isolation=False,
        retry_state=retry_state,
        prepared_operation=None,
        setup_started_at=1.0,
        settings=settings,
        timeout_seconds=settings.accurate_runtime.timeout_seconds,
        backend=None,
        expected_backend_id="hf_whisper",
        expected_profile="accurate",
        allow_retries=True,
        enforce_timeout=True,
        cpu_backend_builder=lambda: backend,
        logger=logging.getLogger("ser.tests.accurate_execution_flow"),
        setup_phase_name="emotion_setup",
        finalize_in_process_setup=lambda **kwargs: captured.__setitem__("finalize", kwargs),
        prepare_accurate_backend_runtime=lambda _backend: None,
        log_phase_started=lambda *args, **kwargs: None,
        log_phase_completed=lambda *args, **kwargs: None,
        log_phase_failed=lambda *args, **kwargs: None,
        build_transient_failure_handler=_build_transient_failure_handler,
        should_retry_on_cpu_after_transient_failure=lambda _err: True,
        summarize_transient_failure=lambda err: err,
        process_payload_cpu_fallback=lambda payload: payload,
        run_retry_policy=_run_retry_policy,
        retry_delay_seconds=lambda **kwargs: 0.0,
        run_with_retry_policy=lambda **kwargs: expected,
        passthrough_error_types=(RuntimeError, ValueError),
        run_accurate_retryable_operation=lambda **kwargs: expected,
        timeout_error_type=RuntimeError,
        transient_error_type=ValueError,
        transient_exhausted_error=lambda err: RuntimeError(str(err)),
        runtime_error_factory=lambda err: RuntimeError(str(err)),
    )

    assert result == expected
    assert captured["operation_result"] == expected
    finalize = captured["finalize"]
    assert isinstance(finalize, dict)
    assert finalize["use_process_isolation"] is False
    assert finalize["state"] is retry_state
    transient = captured["transient"]
    assert isinstance(transient, dict)
    assert transient["state"] is retry_state
    assert transient["expected_backend_id"] == "hf_whisper"
    retry_policy = captured["retry_policy"]
    assert isinstance(retry_policy, dict)
    assert retry_policy["runtime_config"] == settings.accurate_runtime
    assert retry_policy["allow_retries"] is True
    assert retry_policy["profile_label"] == "Accurate"
