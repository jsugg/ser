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
    finalize_in_process_setup,
    prepare_in_process_operation,
    prepare_retry_state,
)
from ser.runtime.contracts import InferenceRequest


@dataclass(frozen=True)
class _Request:
    file_path: str


@dataclass(frozen=True)
class _Payload:
    request: InferenceRequest
    settings: AppConfig


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
