"""Retry fallback helpers for accurate-profile inference operations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, ProfileRuntimeConfig


class _TorchRuntimeLike(Protocol):
    """Minimal torch-runtime contract needed for device fallback checks."""

    @property
    def device(self) -> str: ...


class _SettingsLike(Protocol):
    """Settings contract that exposes torch runtime selectors."""

    @property
    def torch_runtime(self) -> _TorchRuntimeLike: ...


class _PayloadLike(Protocol):
    """Serializable payload contract that carries settings snapshot."""

    @property
    def settings(self) -> _SettingsLike: ...


class _RequestLike(Protocol):
    """Minimal request contract carrying the audio file path."""

    @property
    def file_path(self) -> str: ...


_PayloadT = TypeVar("_PayloadT", bound=_PayloadLike)
_BackendT = TypeVar("_BackendT")
_LoadedModelT = TypeVar("_LoadedModelT")


@dataclass(slots=True)
class AccurateRetryOperationState(Generic[_PayloadT, _BackendT]):
    """Mutable retry state for one accurate inference request."""

    process_payload: _PayloadT | None = None
    active_backend: _BackendT | None = None
    cpu_fallback_applied: bool = False


@dataclass(frozen=True, slots=True)
class PreparedAccurateOperation(Generic[_LoadedModelT, _BackendT]):
    """Setup-complete payload for one in-process accurate inference operation."""

    loaded_model: _LoadedModelT
    backend: _BackendT
    audio: NDArray[np.float32]
    sample_rate: int
    runtime_config: ProfileRuntimeConfig


def prepare_retry_state(
    *,
    use_process_isolation: bool,
    request: _RequestLike,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    process_payload: _PayloadT | None,
    prepare_in_process_operation: Callable[
        ...,
        PreparedAccurateOperation[_LoadedModelT, _BackendT],
    ],
) -> tuple[
    AccurateRetryOperationState[_PayloadT, _BackendT],
    PreparedAccurateOperation[_LoadedModelT, _BackendT] | None,
    float | None,
]:
    """Builds initial retry state for accurate runtime and performs untimed setup."""
    retry_state = AccurateRetryOperationState[_PayloadT, _BackendT]()
    setup_started_at: float | None = None
    if use_process_isolation:
        if process_payload is None:
            raise RuntimeError(
                "Accurate process payload is missing for isolated execution."
            )
        retry_state.process_payload = process_payload
        return retry_state, None, setup_started_at

    setup_started_at = log_phase_started(
        logger,
        phase_name=setup_phase_name,
        profile=profile,
    )
    prepared_operation: PreparedAccurateOperation[_LoadedModelT, _BackendT] | None = (
        None
    )
    try:
        prepared_operation = prepare_in_process_operation(
            request=request,
            settings=settings,
            runtime_config=runtime_config,
            loaded_model=loaded_model,
            backend=backend,
        )
        retry_state.active_backend = prepared_operation.backend
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    return retry_state, prepared_operation, setup_started_at


def finalize_in_process_setup(
    *,
    use_process_isolation: bool,
    state: AccurateRetryOperationState[_PayloadT, _BackendT],
    setup_started_at: float | None,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    prepare_accurate_backend_runtime: Callable[[_BackendT], None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    runtime_error_factory: Callable[[str], Exception],
) -> None:
    """Prepares in-process backend runtime and emits setup phase diagnostics."""
    if use_process_isolation:
        return
    if state.active_backend is None:
        raise runtime_error_factory(
            "Accurate backend is missing for in-process inference."
        )
    try:
        prepare_accurate_backend_runtime(state.active_backend)
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    if setup_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=setup_phase_name,
            started_at=setup_started_at,
            profile=profile,
        )


def prepare_in_process_operation(
    *,
    request: _RequestLike,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    load_accurate_model: Callable[[AppConfig], _LoadedModelT],
    validate_loaded_model: Callable[[_LoadedModelT], None],
    read_audio_file: Callable[[str], tuple[NDArray[np.float32], int]],
    build_backend_for_profile: Callable[[AppConfig], _BackendT],
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedAccurateOperation[_LoadedModelT, _BackendT]:
    """Performs untimed setup for one in-process accurate operation."""
    active_loaded_model: _LoadedModelT
    if loaded_model is None:
        try:
            active_loaded_model = load_accurate_model(settings)
        except FileNotFoundError as err:
            raise model_unavailable_error_factory(str(err)) from err
        except ValueError as err:
            raise model_load_error_factory(
                "Failed to load accurate-profile model artifact from configured paths."
            ) from err
    else:
        active_loaded_model = loaded_model

    validate_loaded_model(active_loaded_model)
    audio, sample_rate = read_audio_file(request.file_path)
    audio_array = np.asarray(audio, dtype=np.float32)
    active_backend = (
        backend if backend is not None else build_backend_for_profile(settings)
    )
    return PreparedAccurateOperation(
        loaded_model=active_loaded_model,
        backend=active_backend,
        audio=audio_array,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
    )


def build_transient_failure_handler(
    *,
    state: AccurateRetryOperationState[_PayloadT, _BackendT],
    use_process_isolation: bool,
    injected_backend: _BackendT | None,
    expected_backend_id: str,
    policy_device: str,
    logger: logging.Logger,
    should_retry_on_cpu_after_transient_failure: Callable[[str], bool],
    summarize_transient_failure: Callable[[str], str],
    process_payload_cpu_fallback: Callable[[_PayloadT], _PayloadT],
    in_process_cpu_backend_builder: Callable[[], _BackendT],
    prepare_accurate_backend_runtime: Callable[[_BackendT], None],
    runtime_error_factory: Callable[[str], Exception],
) -> Callable[[Exception, int, int], None]:
    """Builds retry callback that applies one-time CPU fallback for MPS OOM errors."""

    def on_transient_failure(
        err: Exception,
        _attempt: int,
        _transient_failures: int,
    ) -> None:
        if state.cpu_fallback_applied:
            return
        if expected_backend_id != "hf_whisper":
            return
        error_text = str(err)
        if not should_retry_on_cpu_after_transient_failure(error_text):
            return
        if not use_process_isolation and injected_backend is not None:
            logger.info(
                "Accurate inference detected MPS OOM but cannot switch "
                "to CPU when a custom backend instance is injected."
            )
            return

        current_device = (
            state.process_payload.settings.torch_runtime.device
            if use_process_isolation and state.process_payload is not None
            else policy_device
        )
        if current_device.strip().lower() not in {"auto", "mps"}:
            return
        logger.info(
            "Accurate inference will retry on CPU after MPS OOM (%s).",
            summarize_transient_failure(error_text),
        )
        state.cpu_fallback_applied = True
        try:
            if use_process_isolation:
                if state.process_payload is None:
                    raise runtime_error_factory(
                        "Accurate process payload is missing for CPU fallback."
                    )
                state.process_payload = process_payload_cpu_fallback(
                    state.process_payload
                )
                return
            active_backend = in_process_cpu_backend_builder()
            prepare_accurate_backend_runtime(active_backend)
            state.active_backend = active_backend
        except Exception as swap_err:
            state.cpu_fallback_applied = False
            logger.warning(
                "Accurate inference CPU fallback setup failed; "
                "continuing with existing retry policy: %s",
                swap_err,
            )

    return on_transient_failure


__all__ = [
    "AccurateRetryOperationState",
    "PreparedAccurateOperation",
    "build_transient_failure_handler",
    "finalize_in_process_setup",
    "prepare_in_process_operation",
    "prepare_retry_state",
]
