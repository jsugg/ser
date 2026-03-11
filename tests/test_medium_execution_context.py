"""Tests for medium execution-context preparation helper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from ser.config import AppConfig
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_execution_context import prepare_execution_context
from ser.runtime.medium_worker_operation import MediumRetryOperationState


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    request: InferenceRequest
    settings: AppConfig
    expected_backend_model_id: str


def test_prepare_execution_context_resolves_policy_and_builds_isolated_retry_state() -> None:
    """Helper should resolve runtime policy and prepare isolated retry state."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            medium_runtime=SimpleNamespace(process_isolation=True),
            runtime_flags=SimpleNamespace(profile_pipeline=True),
            torch_runtime=SimpleNamespace(device="cpu", dtype="float32"),
        ),
    )
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    captured: dict[str, object] = {}
    expected_state = MediumRetryOperationState[_PayloadStub, object, object]()

    context = prepare_execution_context(
        request=request,
        settings=settings,
        loaded_model=None,
        backend=None,
        enforce_timeout=True,
        resolve_medium_model_id=lambda _settings: "unit-test/xlsr-v1",
        resolve_runtime_policy=lambda _settings: FeatureRuntimePolicy(
            device="mps",
            dtype="float16",
            reason="policy_override",
        ),
        log_selector_adjustment=lambda device, dtype, reason: captured.update(
            {"selector_log": (device, dtype, reason)}
        ),
        prepare_retry_state=lambda **kwargs: (
            captured.update(
                {
                    "prepare_inputs": {
                        **kwargs,
                        "process_payload": kwargs["build_process_payload"](),
                    }
                }
            )
            or (expected_state, 1.25)
        ),
        build_process_payload=lambda expected_backend_model_id, device, dtype: (
            _PayloadStub(
                request=request,
                settings=cast(
                    AppConfig,
                    SimpleNamespace(torch_runtime=SimpleNamespace(device=device, dtype=dtype)),
                ),
                expected_backend_model_id=expected_backend_model_id,
            )
        ),
        prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-process setup should not run in this test")
        ),
        logger=logging.getLogger(__name__),
        profile="medium",
        setup_phase_name="setup",
        log_phase_started=lambda *_args, **_kwargs: 0.0,
        log_phase_failed=lambda *_args, **_kwargs: 0.0,
    )

    prepare_inputs = cast(dict[str, object], captured["prepare_inputs"])
    assert context.runtime_config is settings.medium_runtime
    assert context.expected_backend_model_id == "unit-test/xlsr-v1"
    assert context.runtime_policy == FeatureRuntimePolicy(
        device="mps",
        dtype="float16",
        reason="policy_override",
    )
    assert context.use_process_isolation is True
    assert context.retry_state is expected_state
    assert context.setup_started_at == 1.25
    assert captured["selector_log"] == ("mps", "float16", "policy_override")
    assert prepare_inputs["use_process_isolation"] is True
    assert prepare_inputs["expected_backend_model_id"] == "unit-test/xlsr-v1"
    assert prepare_inputs["policy_device"] == "mps"
    assert prepare_inputs["policy_dtype"] == "float16"
    process_payload = cast(_PayloadStub, prepare_inputs["process_payload"])
    assert process_payload.request == request
    assert process_payload.expected_backend_model_id == "unit-test/xlsr-v1"
    assert process_payload.settings.torch_runtime.device == "mps"
    assert process_payload.settings.torch_runtime.dtype == "float16"


def test_prepare_execution_context_skips_selector_log_for_matching_runtime() -> None:
    """Helper should keep in-process execution when runtime prerequisites are injected."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            medium_runtime=SimpleNamespace(process_isolation=True),
            runtime_flags=SimpleNamespace(profile_pipeline=True),
            torch_runtime=SimpleNamespace(device="cpu", dtype="float32"),
        ),
    )
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    captured: dict[str, object] = {}
    expected_state = MediumRetryOperationState[_PayloadStub, object, object]()

    context = prepare_execution_context(
        request=request,
        settings=settings,
        loaded_model=None,
        backend=object(),
        enforce_timeout=True,
        resolve_medium_model_id=lambda _settings: "unit-test/xlsr-v1",
        resolve_runtime_policy=lambda _settings: FeatureRuntimePolicy(
            device="cpu",
            dtype="float32",
            reason="unchanged",
        ),
        log_selector_adjustment=lambda device, dtype, reason: captured.update(
            {"selector_log": (device, dtype, reason)}
        ),
        prepare_retry_state=lambda **kwargs: (
            captured.update({"use_process_isolation": kwargs["use_process_isolation"]})
            or (expected_state, None)
        ),
        build_process_payload=lambda expected_backend_model_id, device, dtype: (
            _PayloadStub(
                request=request,
                settings=cast(
                    AppConfig,
                    SimpleNamespace(torch_runtime=SimpleNamespace(device=device, dtype=dtype)),
                ),
                expected_backend_model_id=expected_backend_model_id,
            )
        ),
        prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-process setup should not run in this test")
        ),
        logger=logging.getLogger(__name__),
        profile="medium",
        setup_phase_name="setup",
        log_phase_started=lambda *_args, **_kwargs: 0.0,
        log_phase_failed=lambda *_args, **_kwargs: 0.0,
    )

    assert context.use_process_isolation is False
    assert context.retry_state is expected_state
    assert context.setup_started_at is None
    assert "selector_log" not in captured
    assert captured["use_process_isolation"] is False
