"""Tests for medium lock-body execution orchestration helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from ser import config
from ser.config import AppConfig
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_execution_context import MediumExecutionContext
from ser.runtime.medium_execution_flow import execute_medium_inference_with_retry
from ser.runtime.medium_worker_operation import MediumRetryOperationState


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    request: InferenceRequest
    settings: AppConfig
    expected_backend_model_id: str


def test_execute_medium_inference_with_retry_wires_helper_chain() -> None:
    """Helper should compose finalize, attempt execution, transient hook, and retry policy."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id="unit-test/xlsr-v1",
    )
    retry_state = MediumRetryOperationState[_PayloadStub, object, object](
        process_payload=payload,
        prepared_operation=None,
    )
    context = MediumExecutionContext[_PayloadStub, object, object](
        runtime_config=settings.medium_runtime,
        expected_backend_model_id="unit-test/xlsr-v1",
        runtime_policy=FeatureRuntimePolicy(
            device="mps",
            dtype="float16",
            reason="policy_override",
        ),
        use_process_isolation=True,
        retry_state=retry_state,
        setup_started_at=1.5,
    )
    captured: dict[str, object] = {}

    def _transient_handler(_err: Exception, _attempt: int, _failures: int) -> None:
        return None

    result = execute_medium_inference_with_retry(
        execution_context=context,
        injected_backend=None,
        enforce_timeout=True,
        allow_retries=True,
        logger=logging.getLogger(__name__),
        profile="medium",
        setup_phase_name="setup",
        inference_phase_name="inference",
        finalize_in_process_setup=lambda **kwargs: captured.update({"finalize": kwargs}),
        prepare_medium_backend_runtime=lambda _backend: None,
        log_phase_started=lambda *_args, **_kwargs: None,
        log_phase_completed=lambda *_args, **_kwargs: None,
        log_phase_failed=lambda *_args, **_kwargs: None,
        run_inference_operation=lambda **kwargs: captured.update({"run_inference": kwargs}) or "ok",
        run_with_process_timeout=lambda _payload, _timeout_seconds: "never",
        run_process_operation=lambda _prepared: "never",
        run_with_timeout=lambda _operation, _timeout_seconds: "never",
        build_transient_failure_handler=lambda **kwargs: captured.update({"transient_args": kwargs})
        or _transient_handler,
        should_retry_on_cpu_after_transient_failure=lambda _err: False,
        summarize_transient_failure=lambda err: str(err),
        process_payload_cpu_fallback=lambda current_payload: current_payload,
        in_process_cpu_backend_builder=lambda: object(),
        replace_prepared_backend=lambda prepared, _active_backend: prepared,
        run_retry_policy=lambda **kwargs: captured.update({"retry_policy": kwargs})
        or kwargs["operation"](),
        retry_delay_seconds=lambda **_kwargs: 0.0,
        timeout_error_type=TimeoutError,
        transient_error_type=RuntimeError,
        transient_exhausted_error=lambda err: RuntimeError(str(err)),
        run_with_retry_policy=lambda **_kwargs: "unused",
        passthrough_error_types=(ValueError,),
        runtime_error_factory=lambda err: RuntimeError(str(err)),
    )

    assert result == "ok"
    finalize_kwargs = cast(dict[str, object], captured["finalize"])
    assert finalize_kwargs["use_process_isolation"] is True
    assert finalize_kwargs["state"] is retry_state
    assert finalize_kwargs["setup_started_at"] == 1.5
    assert finalize_kwargs["profile"] == "medium"

    retry_kwargs = cast(dict[str, object], captured["retry_policy"])
    assert retry_kwargs["runtime_config"] is context.runtime_config
    assert retry_kwargs["allow_retries"] is True
    assert retry_kwargs["profile_label"] == "Medium"
    assert retry_kwargs["on_transient_failure"] is _transient_handler
    assert callable(retry_kwargs["run_with_retry_policy"])

    run_inference_kwargs = cast(dict[str, object], captured["run_inference"])
    assert run_inference_kwargs["use_process_isolation"] is True
    assert run_inference_kwargs["process_payload"] is payload
    assert run_inference_kwargs["prepared_operation"] is None
    assert run_inference_kwargs["timeout_seconds"] == settings.medium_runtime.timeout_seconds

    transient_kwargs = cast(dict[str, object], captured["transient_args"])
    assert transient_kwargs["state"] is retry_state
    assert transient_kwargs["policy_device"] == "mps"
    assert transient_kwargs["injected_backend"] is None
