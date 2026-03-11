"""Contract tests for accurate worker-operation setup helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ser.config as config
from ser.config import AppConfig, ProfileRuntimeConfig
from ser.runtime.accurate_worker_operation import (
    AccurateRetryOperationState,
    PreparedAccurateOperation,
    finalize_in_process_setup,
    prepare_in_process_operation,
    prepare_process_operation,
    prepare_retry_state,
    run_inference_operation,
    run_process_operation,
)
from ser.runtime.contracts import InferenceRequest


@dataclass(frozen=True)
class _Request:
    file_path: str


@dataclass(frozen=True)
class _Payload:
    request: InferenceRequest
    settings: AppConfig


@dataclass(frozen=True)
class _ProcessPayload:
    request: _Request
    settings: AppConfig
    expected_backend_id: str
    expected_profile: str
    expected_backend_model_id: str | None


def test_prepare_in_process_operation_preserves_injected_model_and_backend() -> None:
    """Helper should keep injected model/backend and build float32 audio payload."""
    loaded_model = object()
    backend = object()
    calls: dict[str, object] = {}

    def _validate_loaded_model(candidate: object) -> None:
        calls["compatible"] = candidate
        calls["warned"] = candidate

    prepared = prepare_in_process_operation(
        request=_Request(file_path="sample.wav"),
        settings=cast(AppConfig, SimpleNamespace()),
        runtime_config=cast(ProfileRuntimeConfig, SimpleNamespace()),
        loaded_model=loaded_model,
        backend=backend,
        load_accurate_model=lambda _settings: (_ for _ in ()).throw(
            AssertionError("model loader should not run when model is injected")
        ),
        validate_loaded_model=_validate_loaded_model,
        read_audio_file=lambda _path: (
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            16_000,
        ),
        build_backend_for_profile=lambda _settings: (_ for _ in ()).throw(
            AssertionError("backend builder should not run when backend is injected")
        ),
        model_unavailable_error_factory=RuntimeError,
        model_load_error_factory=RuntimeError,
    )

    assert prepared.loaded_model is loaded_model
    assert prepared.backend is backend
    assert prepared.sample_rate == 16_000
    assert prepared.audio.dtype == np.float32
    assert calls["compatible"] is loaded_model
    assert calls["warned"] is loaded_model


def test_prepare_process_operation_builds_runtime_payload_and_warms_backend() -> None:
    """Process helper should validate model, cast audio, and warm backend runtime."""

    settings = config.reload_settings()
    runtime_config = cast(ProfileRuntimeConfig, SimpleNamespace(mode="accurate"))
    loaded_model = object()
    backend = object()
    calls: dict[str, object] = {}

    def _resolve_runtime_config(
        active_settings: AppConfig,
        expected_profile: str,
    ) -> ProfileRuntimeConfig:
        calls["runtime_inputs"] = (active_settings, expected_profile)
        return runtime_config

    def _load_accurate_model(payload: _ProcessPayload) -> object:
        calls["loaded_from"] = payload.expected_backend_id
        return loaded_model

    def _validate_loaded_model(candidate: object, payload: _ProcessPayload) -> None:
        calls["validated"] = (candidate, payload.expected_profile)

    def _build_backend(payload: _ProcessPayload) -> object:
        calls["backend_for"] = payload.expected_profile
        return backend

    def _prepare_backend_runtime(candidate: object) -> None:
        calls["prepared_backend"] = candidate

    prepared = prepare_process_operation(
        _ProcessPayload(
            request=_Request(file_path="sample.wav"),
            settings=settings,
            expected_backend_id="hf_whisper",
            expected_profile="accurate",
            expected_backend_model_id="openai/whisper-large-v3",
        ),
        resolve_runtime_config=_resolve_runtime_config,
        load_accurate_model=_load_accurate_model,
        validate_loaded_model=_validate_loaded_model,
        read_audio_file=lambda _path: (
            np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            16_000,
        ),
        build_backend_for_payload=_build_backend,
        prepare_accurate_backend_runtime=_prepare_backend_runtime,
        model_unavailable_error_factory=RuntimeError,
        model_load_error_factory=RuntimeError,
    )

    assert prepared.loaded_model is loaded_model
    assert prepared.backend is backend
    assert prepared.sample_rate == 16_000
    assert prepared.audio.dtype == np.float32
    assert calls["runtime_inputs"] == (settings, "accurate")
    assert calls["loaded_from"] == "hf_whisper"
    assert calls["validated"] == (loaded_model, "accurate")
    assert calls["backend_for"] == "accurate"
    assert calls["prepared_backend"] is backend


def test_prepare_in_process_operation_maps_missing_model_error() -> None:
    """Missing model should map to unavailable error contract."""

    class _UnavailableError(RuntimeError):
        pass

    with pytest.raises(_UnavailableError, match="Train first"):
        _ = prepare_in_process_operation(
            request=_Request(file_path="sample.wav"),
            settings=cast(AppConfig, SimpleNamespace()),
            runtime_config=cast(ProfileRuntimeConfig, SimpleNamespace()),
            loaded_model=None,
            backend=object(),
            load_accurate_model=lambda _settings: (_ for _ in ()).throw(
                FileNotFoundError("Train first")
            ),
            validate_loaded_model=lambda _candidate: None,
            read_audio_file=lambda _path: (
                np.asarray([0.1], dtype=np.float32),
                16_000,
            ),
            build_backend_for_profile=lambda _settings: object(),
            model_unavailable_error_factory=_UnavailableError,
            model_load_error_factory=RuntimeError,
        )


def test_prepare_in_process_operation_maps_invalid_model_error() -> None:
    """Invalid model payload should map to model-load error contract."""

    class _ModelLoadError(RuntimeError):
        pass

    with pytest.raises(_ModelLoadError, match="Failed to load accurate-profile"):
        _ = prepare_in_process_operation(
            request=_Request(file_path="sample.wav"),
            settings=cast(AppConfig, SimpleNamespace()),
            runtime_config=cast(ProfileRuntimeConfig, SimpleNamespace()),
            loaded_model=None,
            backend=object(),
            load_accurate_model=lambda _settings: (_ for _ in ()).throw(
                ValueError("invalid artifact")
            ),
            validate_loaded_model=lambda _candidate: None,
            read_audio_file=lambda _path: (
                np.asarray([0.1], dtype=np.float32),
                16_000,
            ),
            build_backend_for_profile=lambda _settings: object(),
            model_unavailable_error_factory=RuntimeError,
            model_load_error_factory=_ModelLoadError,
        )


def test_prepare_process_operation_maps_missing_model_error() -> None:
    """Process helper should map missing model files to unavailable errors."""

    class _UnavailableError(RuntimeError):
        pass

    with pytest.raises(_UnavailableError, match="Train first"):
        _ = prepare_process_operation(
            _ProcessPayload(
                request=_Request(file_path="sample.wav"),
                settings=config.reload_settings(),
                expected_backend_id="hf_whisper",
                expected_profile="accurate",
                expected_backend_model_id=None,
            ),
            resolve_runtime_config=lambda _settings, _profile: cast(
                ProfileRuntimeConfig,
                SimpleNamespace(),
            ),
            load_accurate_model=lambda _payload: (_ for _ in ()).throw(
                FileNotFoundError("Train first")
            ),
            validate_loaded_model=lambda _candidate, _payload: None,
            read_audio_file=lambda _path: (
                np.asarray([0.1], dtype=np.float32),
                16_000,
            ),
            build_backend_for_payload=lambda _payload: object(),
            prepare_accurate_backend_runtime=lambda _backend: None,
            model_unavailable_error_factory=_UnavailableError,
            model_load_error_factory=RuntimeError,
        )


def test_prepare_process_operation_maps_invalid_model_error() -> None:
    """Process helper should map invalid artifacts to model-load errors."""

    class _ModelLoadError(RuntimeError):
        pass

    with pytest.raises(_ModelLoadError, match="Failed to load accurate-profile"):
        _ = prepare_process_operation(
            _ProcessPayload(
                request=_Request(file_path="sample.wav"),
                settings=config.reload_settings(),
                expected_backend_id="hf_whisper",
                expected_profile="accurate",
                expected_backend_model_id=None,
            ),
            resolve_runtime_config=lambda _settings, _profile: cast(
                ProfileRuntimeConfig,
                SimpleNamespace(),
            ),
            load_accurate_model=lambda _payload: (_ for _ in ()).throw(
                ValueError("invalid artifact")
            ),
            validate_loaded_model=lambda _candidate, _payload: None,
            read_audio_file=lambda _path: (
                np.asarray([0.1], dtype=np.float32),
                16_000,
            ),
            build_backend_for_payload=lambda _payload: object(),
            prepare_accurate_backend_runtime=lambda _backend: None,
            model_unavailable_error_factory=RuntimeError,
            model_load_error_factory=_ModelLoadError,
        )


def test_run_inference_operation_process_isolation_delegates() -> None:
    """Accurate helper should delegate isolated attempts to the process runner."""
    payload = cast(_Payload, SimpleNamespace())

    def _run_once(
        *,
        loaded_model: object,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: ProfileRuntimeConfig,
    ) -> str:
        del loaded_model, backend, audio, sample_rate, runtime_config
        return "never"

    result = run_inference_operation(
        enforce_timeout=True,
        use_process_isolation=True,
        process_payload=payload,
        prepared_operation=None,
        active_backend=None,
        timeout_seconds=7.0,
        expected_profile="accurate",
        logger=logging.getLogger(__name__),
        inference_phase_name="phase",
        log_phase_started=lambda *_args, **_kwargs: 1.0,
        log_phase_completed=lambda *_args, **_kwargs: None,
        log_phase_failed=lambda *_args, **_kwargs: None,
        run_with_process_timeout=lambda candidate: (
            "process_ok" if candidate is payload else "unexpected"
        ),
        run_accurate_inference_once=_run_once,
        run_with_timeout=lambda **_kwargs: "never",
        timeout_error_factory=TimeoutError,
        runtime_error_factory=RuntimeError,
    )

    assert result == "process_ok"


def test_run_inference_operation_uses_active_backend_for_inprocess_retry() -> None:
    """Accurate helper should run in-process attempts with the current active backend."""
    prepared_backend = object()
    active_backend = object()
    prepared = PreparedAccurateOperation[object, object](
        loaded_model=object(),
        backend=prepared_backend,
        audio=np.asarray([0.1, 0.2], dtype=np.float32),
        sample_rate=16_000,
        runtime_config=cast(ProfileRuntimeConfig, SimpleNamespace()),
    )
    captured: dict[str, object] = {}

    def _run_once(
        *,
        loaded_model: object,
        backend: object,
        audio: np.ndarray,
        sample_rate: int,
        runtime_config: ProfileRuntimeConfig,
    ) -> str:
        captured["loaded_model"] = loaded_model
        captured["backend"] = backend
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        captured["runtime_config"] = runtime_config
        return "inprocess_ok"

    result = run_inference_operation(
        enforce_timeout=False,
        use_process_isolation=False,
        process_payload=None,
        prepared_operation=prepared,
        active_backend=active_backend,
        timeout_seconds=7.0,
        expected_profile="accurate",
        logger=logging.getLogger(__name__),
        inference_phase_name="phase",
        log_phase_started=lambda *_args, **_kwargs: 1.0,
        log_phase_completed=lambda *_args, **_kwargs: None,
        log_phase_failed=lambda *_args, **_kwargs: None,
        run_with_process_timeout=lambda _candidate: "never",
        run_accurate_inference_once=_run_once,
        run_with_timeout=lambda **kwargs: kwargs["operation"](),
        timeout_error_factory=TimeoutError,
        runtime_error_factory=RuntimeError,
    )

    assert result == "inprocess_ok"
    assert captured["loaded_model"] is prepared.loaded_model
    assert captured["backend"] is active_backend
    assert cast(np.ndarray, captured["audio"]).dtype == np.float32
    assert captured["sample_rate"] == 16_000
    assert captured["runtime_config"] is prepared.runtime_config


def test_prepare_retry_state_builds_process_payload_when_isolated() -> None:
    """Retry-state helper should build process payload in isolation mode only."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    payload = _Payload(request=request, settings=settings)

    state, prepared, setup_started_at = prepare_retry_state(
        use_process_isolation=True,
        request=request,
        settings=settings,
        runtime_config=settings.accurate_runtime,
        loaded_model=None,
        backend=None,
        logger=logging.getLogger("ser.tests.accurate_worker_operation"),
        profile="accurate",
        setup_phase_name="setup",
        log_phase_started=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup phase should not start in isolation mode")
        ),
        log_phase_failed=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup phase should not fail in isolation mode")
        ),
        process_payload=payload,
        prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("in-process setup should not run in isolation mode")
        ),
    )

    assert setup_started_at is None
    assert prepared is None
    assert state.process_payload is payload
    assert state.active_backend is None


def test_prepare_retry_state_logs_failure_for_inprocess_setup_errors() -> None:
    """Retry-state helper should log setup failure when in-process setup raises."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )
    phase_events: list[tuple[str, object]] = []
    logger = logging.getLogger("ser.tests.accurate_worker_operation")

    def _log_phase_started(*_args: object, **_kwargs: object) -> float:
        return 3.5

    def _log_phase_failed(*_args: object, **kwargs: object) -> float:
        phase_events.append(("failed", kwargs.get("started_at")))
        return 0.0

    with pytest.raises(RuntimeError, match="setup failed"):
        _ = prepare_retry_state(
            use_process_isolation=False,
            request=request,
            settings=settings,
            runtime_config=settings.accurate_runtime,
            loaded_model=None,
            backend=None,
            logger=logger,
            profile="accurate",
            setup_phase_name="setup",
            log_phase_started=_log_phase_started,
            log_phase_failed=_log_phase_failed,
            process_payload=None,
            prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("setup failed")
            ),
        )

    assert phase_events == [("failed", 3.5)]


def test_prepare_retry_state_requires_process_payload_when_isolated() -> None:
    """Isolation mode should fail fast when the process payload invariant is broken."""
    settings = config.reload_settings()
    request = InferenceRequest(
        file_path="sample.wav",
        language="en",
        save_transcript=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Accurate process payload is missing for isolated execution",
    ):
        _ = prepare_retry_state(
            use_process_isolation=True,
            request=request,
            settings=settings,
            runtime_config=settings.accurate_runtime,
            loaded_model=None,
            backend=None,
            logger=logging.getLogger("ser.tests.accurate_worker_operation"),
            profile="accurate",
            setup_phase_name="setup",
            log_phase_started=lambda *_args, **_kwargs: None,
            log_phase_failed=lambda *_args, **_kwargs: None,
            process_payload=None,
            prepare_in_process_operation=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("in-process setup should not run in isolation mode")
            ),
        )


def test_finalize_in_process_setup_prepares_backend_and_logs_completion() -> None:
    """Finalize helper should warm backend runtime and emit setup completion event."""
    state = AccurateRetryOperationState[_Payload, object](active_backend=object())
    phase_events: list[str] = []
    logger = logging.getLogger("ser.tests.accurate_worker_operation")

    def _prepare_backend_runtime(_backend: object) -> None:
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
        setup_started_at=1.0,
        logger=logger,
        profile="accurate",
        setup_phase_name="setup",
        prepare_accurate_backend_runtime=_prepare_backend_runtime,
        log_phase_completed=_log_phase_completed,
        log_phase_failed=_log_phase_failed,
        runtime_error_factory=RuntimeError,
    )

    assert phase_events == ["prepared", "completed"]


def test_finalize_in_process_setup_requires_active_backend() -> None:
    """Finalize helper should enforce in-process backend availability."""
    state = AccurateRetryOperationState[_Payload, object]()

    with pytest.raises(RuntimeError, match="Accurate backend is missing"):
        finalize_in_process_setup(
            use_process_isolation=False,
            state=state,
            setup_started_at=None,
            logger=logging.getLogger("ser.tests.accurate_worker_operation"),
            profile="accurate",
            setup_phase_name="setup",
            prepare_accurate_backend_runtime=lambda _backend: None,
            log_phase_completed=lambda *_args, **_kwargs: 0.0,
            log_phase_failed=lambda *_args, **_kwargs: 0.0,
            runtime_error_factory=RuntimeError,
        )


def test_run_process_operation_delegates_to_compute_phase() -> None:
    """Process runner should forward prepared fields to compute callback."""

    loaded_model = object()
    backend = object()
    runtime_config = cast(ProfileRuntimeConfig, SimpleNamespace(mode="accurate"))
    observed: dict[str, object] = {}

    result = run_process_operation(
        PreparedAccurateOperation(
            loaded_model=loaded_model,
            backend=backend,
            audio=np.asarray([0.1, 0.2], dtype=np.float32),
            sample_rate=16_000,
            runtime_config=runtime_config,
        ),
        run_accurate_inference_once=lambda model, runtime_backend, audio, sample_rate, resolved_runtime_config: observed.setdefault(
            "call",
            (
                model,
                runtime_backend,
                audio.dtype,
                sample_rate,
                resolved_runtime_config,
            ),
        ),
    )

    assert result == (
        loaded_model,
        backend,
        np.dtype(np.float32),
        16_000,
        runtime_config,
    )
