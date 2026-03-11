"""Tests for medium worker-operation setup helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np
import pytest

import ser.config as config
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_worker_operation import (
    MediumRetryOperationState,
    PreparedMediumOperation,
    build_transient_failure_handler,
    ensure_medium_compatible_model,
    finalize_in_process_setup,
    prepare_in_process_operation,
    prepare_process_operation,
    prepare_retry_state,
    run_inference_operation,
    run_process_operation,
)


@dataclass(frozen=True, slots=True)
class _LoadedModelStub:
    artifact_metadata: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class _RuntimePolicyStub:
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class _PayloadStub:
    request: InferenceRequest
    settings: config.AppConfig
    expected_backend_model_id: str


def test_ensure_medium_compatible_model_requires_backend_model_id() -> None:
    """Compatibility guard should require exact medium backend model id metadata."""
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "backend_model_id": "unit-test/xlsr-v1",
        }
    )

    with pytest.raises(RuntimeError, match="expected 'unit-test/xlsr-v2'"):
        ensure_medium_compatible_model(
            loaded_model,
            expected_backend_model_id="unit-test/xlsr-v2",
            unavailable_error_factory=RuntimeError,
        )


def test_prepare_process_operation_maps_model_load_failures() -> None:
    """Worker setup should map missing-model failures to domain-specific errors."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )

    with pytest.raises(FileNotFoundError, match="missing medium model"):
        _ = prepare_process_operation(
            payload,
            load_medium_model=lambda _settings, _model_id: (_ for _ in ()).throw(
                FileNotFoundError("missing medium model")
            ),
            ensure_medium_compatible_model=lambda _loaded_model, _model_id: None,
            resolve_runtime_policy=lambda _settings: _RuntimePolicyStub(
                device="cpu",
                dtype="float32",
            ),
            warn_on_runtime_selector_mismatch=lambda _loaded_model, _device, _dtype: (None),
            read_audio_file=lambda _file_path: (np.ones(8, dtype=np.float32), 16_000),
            build_medium_backend=lambda _settings, _model_id, _device, _dtype: object(),
            prepare_medium_backend_runtime=lambda _backend: None,
            model_unavailable_error_factory=FileNotFoundError,
            model_load_error_factory=RuntimeError,
        )


def test_prepare_in_process_operation_reuses_injected_model_and_backend() -> None:
    """In-process setup helper should preserve injected runtime prerequisites."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "backend_model_id": settings.models.medium_model_id,
        }
    )
    backend = object()
    captured: dict[str, object] = {}

    prepared = prepare_in_process_operation(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        expected_backend_model_id=settings.models.medium_model_id,
        runtime_device="mps",
        runtime_dtype="float16",
        load_medium_model=lambda _settings, _model_id: (_ for _ in ()).throw(
            AssertionError("load_medium_model should not be called")
        ),
        ensure_medium_compatible_model=lambda _loaded_model, model_id: captured.update(
            {"model_id": model_id}
        ),
        warn_on_runtime_selector_mismatch=lambda _loaded_model, device, dtype: (
            captured.update({"device": device, "dtype": dtype})
        ),
        read_audio_file=lambda _file_path: (np.ones(8, dtype=np.float32), 16_000),
        build_medium_backend=lambda _settings, _model_id, _device, _dtype: (_ for _ in ()).throw(
            AssertionError("build_medium_backend should not be called")
        ),
        model_unavailable_error_factory=FileNotFoundError,
        model_load_error_factory=RuntimeError,
    )

    assert isinstance(prepared, PreparedMediumOperation)
    assert prepared.loaded_model is loaded_model
    assert prepared.backend is backend
    assert prepared.sample_rate == 16_000
    assert prepared.audio.dtype == np.float32
    assert captured == {
        "model_id": settings.models.medium_model_id,
        "device": "mps",
        "dtype": "float16",
    }


def test_prepare_retry_state_builds_process_payload_when_isolated() -> None:
    """Retry-state helper should build process payload in isolation mode only."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    payload = _PayloadStub(
        request=request,
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )

    state, setup_started_at = prepare_retry_state(
        use_process_isolation=True,
        request=request,
        settings=settings,
        loaded_model=None,
        backend=None,
        expected_backend_model_id=settings.models.medium_model_id,
        policy_device="mps",
        policy_dtype="float16",
        logger=logging.getLogger("ser.tests.medium_worker_operation"),
        profile="medium",
        setup_phase_name="setup",
        log_phase_started=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup phase should not start in isolation path")
        ),
        log_phase_failed=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup phase failure should not log in isolation path")
        ),
        build_process_payload=lambda: payload,
        prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-process setup should not run in isolation path")
        ),
    )

    assert setup_started_at is None
    assert state.process_payload is payload
    assert state.prepared_operation is None


def test_prepare_retry_state_logs_failure_for_inprocess_setup_errors() -> None:
    """Retry-state helper should log setup failure when in-process setup raises."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    phase_events: list[tuple[str, object]] = []
    logger = logging.getLogger("ser.tests.medium_worker_operation")

    def _log_phase_started(*_args: object, **_kwargs: object) -> float:
        return 1.5

    def _log_phase_failed(*_args: object, **kwargs: object) -> float:
        phase_events.append(("failed", kwargs.get("started_at")))
        return 0.0

    with pytest.raises(RuntimeError, match="setup failed"):
        _ = prepare_retry_state(
            use_process_isolation=False,
            request=request,
            settings=settings,
            loaded_model=None,
            backend=None,
            expected_backend_model_id=settings.models.medium_model_id,
            policy_device="cpu",
            policy_dtype="float32",
            logger=logger,
            profile="medium",
            setup_phase_name="setup",
            log_phase_started=_log_phase_started,
            log_phase_failed=_log_phase_failed,
            build_process_payload=lambda: (_ for _ in ()).throw(
                AssertionError("process payload should not be built in in-process path")
            ),
            prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("setup failed")
            ),
        )

    assert phase_events == [("failed", 1.5)]


def test_prepare_and_run_process_operation_delegates_compute() -> None:
    """Worker setup helper should build prepared payload used by compute helper."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "backend_model_id": settings.models.medium_model_id,
        }
    )
    backend = object()
    captured: dict[str, object] = {}

    prepared = prepare_process_operation(
        payload,
        load_medium_model=lambda _settings, _model_id: loaded_model,
        ensure_medium_compatible_model=lambda _loaded_model, _model_id: None,
        resolve_runtime_policy=lambda _settings: _RuntimePolicyStub(
            device="mps",
            dtype="float16",
        ),
        warn_on_runtime_selector_mismatch=lambda _loaded_model, device, dtype: (
            captured.update({"device": device, "dtype": dtype})
        ),
        read_audio_file=lambda _file_path: (np.ones(8, dtype=np.float32), 16_000),
        build_medium_backend=lambda _settings, _model_id, _device, _dtype: backend,
        prepare_medium_backend_runtime=lambda _backend: captured.update({"runtime_prepared": True}),
        model_unavailable_error_factory=FileNotFoundError,
        model_load_error_factory=RuntimeError,
    )

    assert isinstance(prepared, PreparedMediumOperation)
    assert prepared.loaded_model is loaded_model
    assert prepared.backend is backend
    assert prepared.sample_rate == 16_000
    assert prepared.audio.dtype == np.float32
    assert captured["device"] == "mps"
    assert captured["dtype"] == "float16"
    assert captured["runtime_prepared"] is True

    expected_result = object()

    result = run_process_operation(
        prepared,
        run_medium_inference_once=lambda _loaded_model, _backend, _audio, _sample_rate, _runtime_config: (
            expected_result
        ),
    )

    assert result is expected_result


def test_finalize_in_process_setup_prepares_backend_and_logs_completion() -> None:
    """Finalize helper should warm backend runtime and emit setup completion event."""
    settings = config.reload_settings()
    prepared = PreparedMediumOperation(
        loaded_model=object(),
        backend=object(),
        audio=np.ones(8, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=settings.medium_runtime,
    )
    state = MediumRetryOperationState[_PayloadStub, object, object](prepared_operation=prepared)
    phase_events: list[str] = []
    logger = logging.getLogger("ser.tests.medium_worker_operation")

    def _prepare_medium_backend_runtime(_backend: object) -> None:
        phase_events.append("prepared")

    def _log_phase_completed(*_args: object, **_kwargs: object) -> float:
        phase_events.append("completed")
        return 0.0

    def _log_phase_failed(*_args: object, **_kwargs: object) -> float:
        phase_events.append("failed")
        return 0.0

    finalize_in_process_setup(
        use_process_isolation=False,
        state=state,
        setup_started_at=2.0,
        logger=logger,
        profile="medium",
        setup_phase_name="setup",
        prepare_medium_backend_runtime=_prepare_medium_backend_runtime,
        log_phase_completed=_log_phase_completed,
        log_phase_failed=_log_phase_failed,
        runtime_error_factory=RuntimeError,
    )

    assert phase_events == ["prepared", "completed"]


def test_build_transient_failure_handler_demotes_process_payload_to_cpu() -> None:
    """Transient fallback helper should demote process payload runtime selectors to CPU."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=replace(
            settings,
            torch_runtime=config.TorchRuntimeConfig(
                device="mps",
                dtype="float16",
                enable_mps_fallback=settings.torch_runtime.enable_mps_fallback,
            ),
        ),
        expected_backend_model_id=settings.models.medium_model_id,
    )
    state = MediumRetryOperationState[_PayloadStub, object, object](process_payload=payload)
    on_transient_failure = build_transient_failure_handler(
        state=state,
        use_process_isolation=True,
        injected_backend=None,
        policy_device="mps",
        logger=logging.getLogger("ser.tests.medium_worker_operation"),
        should_retry_on_cpu_after_transient_failure=lambda err: (
            "out of memory" in str(err).lower()
        ),
        summarize_transient_failure=lambda err: str(err),
        process_payload_cpu_fallback=lambda payload: replace(
            payload,
            settings=replace(
                payload.settings,
                torch_runtime=config.TorchRuntimeConfig(
                    device="cpu",
                    dtype="float32",
                    enable_mps_fallback=(payload.settings.torch_runtime.enable_mps_fallback),
                ),
            ),
        ),
        in_process_cpu_backend_builder=lambda: (_ for _ in ()).throw(
            AssertionError("in-process backend builder should not run in process mode")
        ),
        prepare_medium_backend_runtime=lambda _backend: (_ for _ in ()).throw(
            AssertionError("prepare runtime should not run in process mode")
        ),
        replace_prepared_backend=lambda prepared, _backend: prepared,
        runtime_error_factory=RuntimeError,
    )

    on_transient_failure(RuntimeError("MPS backend out of memory"), 1, 1)

    assert state.cpu_fallback_applied is True
    assert state.process_payload is not None
    assert state.process_payload.settings.torch_runtime.device == "cpu"
    assert state.process_payload.settings.torch_runtime.dtype == "float32"


def test_run_inference_operation_requires_process_payload_when_isolated() -> None:
    """Isolated timeout path should fail closed when payload is missing."""
    settings = config.reload_settings()
    with pytest.raises(RuntimeError, match="process payload is missing"):
        _ = run_inference_operation(
            enforce_timeout=True,
            use_process_isolation=True,
            process_payload=None,
            prepared_operation=None,
            timeout_seconds=settings.medium_runtime.timeout_seconds,
            logger=logging.getLogger("ser.tests.medium_worker_operation"),
            profile="medium",
            inference_phase_name="inference",
            log_phase_started=lambda *_args, **_kwargs: 0.0,
            log_phase_completed=lambda *_args, **_kwargs: 0.0,
            log_phase_failed=lambda *_args, **_kwargs: 0.0,
            run_with_process_timeout=lambda _payload, _timeout_seconds: object(),
            run_process_operation=lambda _prepared: object(),
            run_with_timeout=lambda _operation, _timeout_seconds: object(),
            runtime_error_factory=RuntimeError,
        )


def test_run_inference_operation_delegates_process_timeout_runner() -> None:
    """Isolated timeout path should delegate directly to process timeout runner."""
    settings = config.reload_settings()
    payload = _PayloadStub(
        request=InferenceRequest(
            file_path="sample.wav",
            language="en",
            save_transcript=False,
        ),
        settings=settings,
        expected_backend_model_id=settings.models.medium_model_id,
    )
    expected = object()
    captured: dict[str, object] = {}

    result = run_inference_operation(
        enforce_timeout=True,
        use_process_isolation=True,
        process_payload=payload,
        prepared_operation=None,
        timeout_seconds=settings.medium_runtime.timeout_seconds,
        logger=logging.getLogger("ser.tests.medium_worker_operation"),
        profile="medium",
        inference_phase_name="inference",
        log_phase_started=lambda *_args, **_kwargs: 0.0,
        log_phase_completed=lambda *_args, **_kwargs: 0.0,
        log_phase_failed=lambda *_args, **_kwargs: 0.0,
        run_with_process_timeout=lambda operation_payload, timeout_seconds: (
            captured.update(
                {
                    "payload": operation_payload,
                    "timeout_seconds": timeout_seconds,
                }
            )
            or expected
        ),
        run_process_operation=lambda _prepared: object(),
        run_with_timeout=lambda _operation, _timeout_seconds: object(),
        runtime_error_factory=RuntimeError,
    )

    assert result is expected
    assert captured["payload"] is payload
    assert captured["timeout_seconds"] == settings.medium_runtime.timeout_seconds


def test_run_inference_operation_in_process_timeout_logs_and_delegates() -> None:
    """In-process timeout path should wrap compute call and emit phase timing."""
    settings = config.reload_settings()
    prepared = PreparedMediumOperation(
        loaded_model=object(),
        backend=object(),
        audio=np.ones(8, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=settings.medium_runtime,
    )
    expected = object()
    phase_events: list[tuple[object, ...]] = []
    captured: dict[str, object] = {}

    def _log_phase_started(
        _logger: logging.Logger,
        *,
        phase_name: str,
        profile: str,
    ) -> float:
        phase_events.append(("started", phase_name, profile))
        return 42.0

    def _log_phase_completed(
        _logger: logging.Logger,
        *,
        phase_name: str,
        started_at: float | None,
        profile: str,
    ) -> float:
        phase_events.append(("completed", phase_name, started_at, profile))
        return 0.0

    def _log_phase_failed(
        _logger: logging.Logger,
        *,
        phase_name: str,
        started_at: float | None,
        profile: str,
    ) -> float:
        phase_events.append(("failed", phase_name, started_at, profile))
        return 0.0

    result = run_inference_operation(
        enforce_timeout=True,
        use_process_isolation=False,
        process_payload=None,
        prepared_operation=prepared,
        timeout_seconds=settings.medium_runtime.timeout_seconds,
        logger=logging.getLogger("ser.tests.medium_worker_operation"),
        profile="medium",
        inference_phase_name="inference",
        log_phase_started=_log_phase_started,
        log_phase_completed=_log_phase_completed,
        log_phase_failed=_log_phase_failed,
        run_with_process_timeout=lambda _payload, _timeout_seconds: object(),
        run_process_operation=lambda operation: (
            captured.update({"prepared_operation": operation}) or expected
        ),
        run_with_timeout=lambda operation, timeout_seconds: (
            captured.update({"timeout_seconds": timeout_seconds}) or operation()
        ),
        runtime_error_factory=RuntimeError,
    )

    assert result is expected
    assert captured["prepared_operation"] is prepared
    assert captured["timeout_seconds"] == settings.medium_runtime.timeout_seconds
    assert phase_events == [
        ("started", "inference", "medium"),
        ("completed", "inference", 42.0, "medium"),
    ]


def test_run_inference_operation_in_process_timeout_logs_failure() -> None:
    """In-process timeout path should emit failed phase event when compute raises."""
    settings = config.reload_settings()
    prepared = PreparedMediumOperation(
        loaded_model=object(),
        backend=object(),
        audio=np.ones(8, dtype=np.float32),
        sample_rate=16_000,
        runtime_config=settings.medium_runtime,
    )
    phase_events: list[tuple[object, ...]] = []

    def _log_phase_failed(
        _logger: logging.Logger,
        *,
        phase_name: str,
        started_at: float | None,
        profile: str,
    ) -> float:
        phase_events.append((phase_name, started_at, profile))
        return 0.0

    with pytest.raises(RuntimeError, match="boom"):
        _ = run_inference_operation(
            enforce_timeout=True,
            use_process_isolation=False,
            process_payload=None,
            prepared_operation=prepared,
            timeout_seconds=settings.medium_runtime.timeout_seconds,
            logger=logging.getLogger("ser.tests.medium_worker_operation"),
            profile="medium",
            inference_phase_name="inference",
            log_phase_started=lambda *_args, **_kwargs: 11.0,
            log_phase_completed=lambda *_args, **_kwargs: 0.0,
            log_phase_failed=_log_phase_failed,
            run_with_process_timeout=lambda _payload, _timeout_seconds: object(),
            run_process_operation=lambda _prepared: (_ for _ in ()).throw(RuntimeError("boom")),
            run_with_timeout=lambda operation, _timeout_seconds: operation(),
            runtime_error_factory=RuntimeError,
        )

    assert phase_events == [("inference", 11.0, "medium")]
