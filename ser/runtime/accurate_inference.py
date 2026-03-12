"""Accurate-profile inference runner with bounded retries and timeout guards."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ser._internal.runtime import accurate_public_boundary as _boundary_support
from ser._internal.runtime.single_flight import SingleFlightRegistry
from ser._internal.runtime.worker_lifecycle import (
    is_setup_complete_message as _is_setup_complete_message_impl,
)
from ser._internal.runtime.worker_lifecycle import (
    parse_worker_completion_message as _parse_worker_completion_message_impl,
)
from ser._internal.runtime.worker_lifecycle import raise_worker_error as _raise_worker_error_impl
from ser._internal.runtime.worker_lifecycle import recv_worker_message as _recv_worker_message_impl
from ser._internal.runtime.worker_lifecycle import (
    run_process_setup_compute_handshake as _run_process_setup_compute_handshake_impl,
)
from ser._internal.runtime.worker_lifecycle import run_with_timeout as _run_with_timeout_impl
from ser._internal.runtime.worker_lifecycle import (
    terminate_worker_process as _terminate_worker_process_impl,
)
from ser.config import AppConfig, ProfileRuntimeConfig
from ser.models.emotion_model import LoadedModel, load_model
from ser.models.profile_runtime import resolve_accurate_model_id
from ser.repr import (
    Emotion2VecBackend,
    FeatureBackend,
    WhisperBackend,
)
from ser.runtime.accurate_backend_runtime import (
    runtime_config_for_profile as _runtime_config_for_profile_impl,
)
from ser.runtime.accurate_runtime_support import (
    build_cpu_settings_snapshot as _build_cpu_settings_snapshot_impl,
)
from ser.runtime.accurate_runtime_support import (
    build_process_settings_snapshot as _build_process_settings_snapshot_impl,
)
from ser.runtime.accurate_worker_lifecycle import (
    is_setup_complete_message as _is_setup_complete_message_orchestration,
)
from ser.runtime.accurate_worker_lifecycle import (
    parse_worker_completion_message as _parse_worker_completion_message_orchestration,
)
from ser.runtime.accurate_worker_lifecycle import (
    raise_worker_error as _raise_worker_error_orchestration,
)
from ser.runtime.accurate_worker_lifecycle import (
    recv_worker_message as _recv_worker_message_orchestration,
)
from ser.runtime.accurate_worker_lifecycle import (
    run_with_process_timeout as _run_with_process_timeout_orchestration_impl,
)
from ser.runtime.accurate_worker_lifecycle import (
    run_worker_entry as _run_worker_entry_orchestration,
)
from ser.runtime.accurate_worker_lifecycle import (
    terminate_worker_process as _terminate_worker_process_orchestration,
)
from ser.runtime.accurate_worker_operation import (
    AccurateRetryOperationState,
    PreparedAccurateOperation,
)
from ser.runtime.accurate_worker_operation import prepare_retry_state as _prepare_retry_state_impl
from ser.runtime.accurate_worker_operation import (
    run_inference_operation as _run_inference_operation_impl,
)
from ser.runtime.contracts import InferenceRequest
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.schema import InferenceResult
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type FeatureMatrix = NDArray[np.float64]
type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage
type _PreparedAccurateOperation = PreparedAccurateOperation[LoadedModel, FeatureBackend]

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_REGISTRY = SingleFlightRegistry()


@dataclass(frozen=True)
class AccurateProcessPayload:
    """Serializable payload for one process-isolated accurate inference attempt."""

    request: InferenceRequest
    settings: AppConfig
    expected_backend_id: str
    expected_profile: str
    expected_backend_model_id: str | None


class AccurateModelUnavailableError(FileNotFoundError):
    """Raised when a compatible accurate-profile model artifact is unavailable."""


class AccurateRuntimeDependencyError(RuntimeError):
    """Raised when accurate optional runtime dependencies are missing."""


class AccurateModelLoadError(RuntimeError):
    """Raised when accurate model artifact loading fails unexpectedly."""


class AccurateInferenceTimeoutError(TimeoutError):
    """Raised when accurate inference exceeds configured timeout budget."""


class AccurateInferenceExecutionError(RuntimeError):
    """Raised when accurate inference exhausts retries without recovery."""


class AccurateTransientBackendError(RuntimeError):
    """Raised for retryable accurate backend encoding failures."""


_WORKER_ERROR_FACTORIES: dict[str, Callable[[str], Exception]] = {
    "ValueError": ValueError,
    "AccurateRuntimeDependencyError": AccurateRuntimeDependencyError,
    "AccurateTransientBackendError": AccurateTransientBackendError,
    "AccurateModelUnavailableError": AccurateModelUnavailableError,
    "AccurateModelLoadError": AccurateModelLoadError,
    "AccurateInferenceTimeoutError": AccurateInferenceTimeoutError,
    "RuntimeError": RuntimeError,
}


def run_accurate_inference(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    backend: FeatureBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
    expected_backend_id: str = "hf_whisper",
    expected_profile: str = "accurate",
    expected_backend_model_id: str | None = None,
) -> InferenceResult:
    """Runs accurate-profile inference with bounded retries and timeout budget.

    Args:
        request: Runtime inference request payload.
        settings: Active application settings.
        loaded_model: Optional preloaded model artifact for repeated inference calls.
        backend: Optional preinitialized feature backend for repeated inference calls.
        enforce_timeout: Whether to apply timeout wrapper for each inference attempt.
        allow_retries: Whether to apply configured retry budget for retryable errors.
        expected_backend_id: Expected backend id in model artifact metadata.
        expected_profile: Expected profile identifier in model artifact metadata.
        expected_backend_model_id: Expected backend model id in model artifact metadata.

    Returns:
        Detailed inference result with frame and segment predictions.

    Raises:
        AccurateModelUnavailableError: If no compatible accurate artifact exists.
        AccurateRuntimeDependencyError: If required accurate dependencies are missing.
        AccurateModelLoadError: If model loading fails for non-compatibility reasons.
        AccurateInferenceTimeoutError: If all attempts timed out.
        AccurateInferenceExecutionError: If transient backend failures exhaust retries.
        ValueError: If feature dimensions are incompatible with loaded artifact.
    """
    runtime_config = _runtime_config_for_profile_impl(
        settings=settings,
        expected_profile=expected_profile,
        unsupported_profile_error=AccurateModelUnavailableError,
    )
    resolved_expected_backend_model_id = expected_backend_model_id
    if resolved_expected_backend_model_id is None and expected_backend_id == "hf_whisper":
        resolved_expected_backend_model_id = resolve_accurate_model_id(settings)
    use_process_isolation = (
        enforce_timeout
        and loaded_model is None
        and backend is None
        and settings.runtime_flags.profile_pipeline
        and runtime_config.process_isolation
    )
    process_payload: AccurateProcessPayload | None = None
    if use_process_isolation:
        process_payload = AccurateProcessPayload(
            request=request,
            settings=_build_process_settings_snapshot_impl(settings),
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=resolved_expected_backend_model_id,
        )
    cpu_settings = _build_cpu_settings_snapshot_impl(settings)
    cpu_backend_builder: Callable[[], FeatureBackend] = partial(
        _build_backend_for_profile,
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=resolved_expected_backend_model_id,
        settings=cpu_settings,
    )

    retry_state, prepared_operation, setup_started_at = _prepare_retry_state_impl(
        use_process_isolation=use_process_isolation,
        request=request,
        settings=settings,
        runtime_config=runtime_config,
        loaded_model=loaded_model,
        backend=backend,
        logger=logger,
        profile=expected_profile,
        setup_phase_name=PHASE_EMOTION_SETUP,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
        process_payload=process_payload,
        prepare_in_process_operation=partial(
            _prepare_in_process_accurate_operation,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=resolved_expected_backend_model_id,
        ),
    )
    with _SINGLE_FLIGHT_REGISTRY.lock(
        profile=expected_profile,
        backend_model_id=resolved_expected_backend_model_id,
    ):
        return _execute_accurate_inference_with_retry(
            use_process_isolation=use_process_isolation,
            retry_state=retry_state,
            prepared_operation=prepared_operation,
            setup_started_at=setup_started_at,
            settings=settings,
            runtime_config=runtime_config,
            backend=backend,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            allow_retries=allow_retries,
            enforce_timeout=enforce_timeout,
            cpu_backend_builder=cpu_backend_builder,
        )


def _execute_accurate_inference_with_retry(
    *,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[AccurateProcessPayload, FeatureBackend],
    prepared_operation: _PreparedAccurateOperation | None,
    setup_started_at: float | None,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    backend: FeatureBackend | None,
    expected_backend_id: str,
    expected_profile: str,
    allow_retries: bool,
    enforce_timeout: bool,
    cpu_backend_builder: Callable[[], FeatureBackend],
) -> InferenceResult:
    """Finalizes setup and executes accurate inference under retry policy."""
    return _boundary_support.execute_accurate_inference_with_retry(
        use_process_isolation=use_process_isolation,
        retry_state=retry_state,
        prepared_operation=prepared_operation,
        setup_started_at=setup_started_at,
        settings=settings,
        backend=backend,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        allow_retries=allow_retries,
        enforce_timeout=enforce_timeout,
        cpu_backend_builder=cpu_backend_builder,
        logger=logger,
        run_accurate_retryable_operation=_run_accurate_retryable_operation,
        retry_delay_seconds=_retry_delay_seconds,
        process_payload_cpu_fallback=_payload_with_cpu_settings,
        timeout_error_type=AccurateInferenceTimeoutError,
        runtime_dependency_error_type=AccurateRuntimeDependencyError,
        inference_execution_error_type=AccurateInferenceExecutionError,
        transient_backend_error_type=AccurateTransientBackendError,
    )


def _run_accurate_retryable_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[AccurateProcessPayload, FeatureBackend],
    prepared_operation: _PreparedAccurateOperation | None,
    timeout_seconds: float,
    expected_profile: str,
) -> InferenceResult:
    """Runs one accurate inference attempt using the current retry state."""
    return _boundary_support.run_accurate_retryable_operation(
        enforce_timeout=enforce_timeout,
        use_process_isolation=use_process_isolation,
        retry_state=retry_state,
        prepared_operation=prepared_operation,
        timeout_seconds=timeout_seconds,
        expected_profile=expected_profile,
        logger=logger,
        run_with_process_timeout=_run_with_process_timeout,
        run_accurate_inference_once=_run_accurate_inference_once,
        run_with_timeout=_run_with_timeout_impl,
        run_inference_operation=_run_inference_operation_impl,
        timeout_error_factory=AccurateInferenceTimeoutError,
    )


def _payload_with_cpu_settings(
    payload: AccurateProcessPayload,
) -> AccurateProcessPayload:
    """Returns one process payload updated to use CPU torch selectors."""
    return _boundary_support.payload_with_cpu_settings(payload)


def _run_accurate_inference_once(
    *,
    loaded_model: LoadedModel,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: ProfileRuntimeConfig,
) -> InferenceResult:
    """Runs one accurate inference attempt without retry control."""
    return _boundary_support.run_accurate_inference_once(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
        logger=logger,
        dependency_error_factory=AccurateRuntimeDependencyError,
        transient_error_factory=AccurateTransientBackendError,
    )


def _run_with_process_timeout(
    payload: AccurateProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    return _run_with_process_timeout_orchestration_impl(
        payload=payload,
        timeout_seconds=timeout_seconds,
        get_context=mp.get_context,
        logger=logger,
        setup_phase_name=PHASE_EMOTION_SETUP,
        inference_phase_name=PHASE_EMOTION_INFERENCE,
        log_phase_started=log_phase_started,
        log_phase_completed=log_phase_completed,
        log_phase_failed=log_phase_failed,
        run_process_setup_compute_handshake=_run_process_setup_compute_handshake_impl,
        worker_target=_worker_entry,
        recv_worker_message=_recv_worker_message,
        is_setup_complete_message=_is_setup_complete_message,
        terminate_worker_process=_terminate_worker_process,
        timeout_error_factory=AccurateInferenceTimeoutError,
        execution_error_factory=AccurateInferenceExecutionError,
        worker_label="Accurate inference",
        process_join_grace_seconds=_TERMINATE_GRACE_SECONDS,
        parse_worker_completion_message=_parse_worker_completion_message,
    )


def _recv_worker_message(
    connection: Connection,
    *,
    stage: str,
) -> tuple[object, ...]:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_orchestration(
        connection=connection,
        stage=stage,
        impl=_recv_worker_message_impl,
        worker_label="Accurate inference",
        error_factory=AccurateInferenceExecutionError,
    )


def _is_setup_complete_message(message: tuple[object, ...]) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_orchestration(
        message=message,
        impl=_is_setup_complete_message_impl,
        worker_label="Accurate inference",
        error_factory=AccurateInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: tuple[object, ...]) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    return _parse_worker_completion_message_orchestration(
        worker_message=worker_message,
        impl=_parse_worker_completion_message_impl,
        worker_label="Accurate inference",
        error_factory=AccurateInferenceExecutionError,
        raise_worker_error=_raise_worker_error,
        result_type=InferenceResult,
    )


def _worker_entry(
    payload: AccurateProcessPayload,
    connection: Connection,
) -> None:
    """Executes one inference operation inside child process."""
    _run_worker_entry_orchestration(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_process_operation,
        run_process_operation=_run_process_operation,
    )


def _prepare_in_process_accurate_operation(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    loaded_model: LoadedModel | None,
    backend: FeatureBackend | None,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
) -> _PreparedAccurateOperation:
    """Prepares one in-process accurate operation using runtime-specific contracts."""
    return cast(
        _PreparedAccurateOperation,
        _boundary_support.prepare_in_process_accurate_operation(
            request=request,
            settings=settings,
            runtime_config=runtime_config,
            loaded_model=loaded_model,
            backend=backend,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=expected_backend_model_id,
            load_model_fn=load_model,
            read_audio_file_fn=read_audio_file,
            build_backend_for_profile_fn=_build_backend_for_profile,
            logger=logger,
            model_unavailable_error_factory=AccurateModelUnavailableError,
            model_load_error_factory=AccurateModelLoadError,
        ),
    )


def _prepare_process_operation(
    payload: AccurateProcessPayload,
) -> _PreparedAccurateOperation:
    """Performs untimed setup for one process-isolated accurate operation."""
    return cast(
        _PreparedAccurateOperation,
        _boundary_support.prepare_process_operation(
            payload,
            load_model_fn=load_model,
            read_audio_file_fn=read_audio_file,
            build_backend_for_profile_fn=_build_backend_for_profile,
            logger=logger,
            model_unavailable_error_factory=AccurateModelUnavailableError,
            model_load_error_factory=AccurateModelLoadError,
            runtime_dependency_error_factory=AccurateRuntimeDependencyError,
            transient_error_factory=AccurateTransientBackendError,
        ),
    )


def _run_process_operation(prepared: _PreparedAccurateOperation) -> InferenceResult:
    """Runs one accurate compute phase inside isolated worker process."""
    return _boundary_support.run_process_operation(
        prepared,
        run_accurate_inference_once=lambda **kwargs: _run_accurate_inference_once(**kwargs),
    )


def _build_backend_for_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
) -> FeatureBackend:
    """Builds a feature backend aligned with profile/backend runtime expectations."""
    return _boundary_support.build_backend_for_profile(
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=expected_backend_model_id,
        settings=settings,
        whisper_backend_factory=WhisperBackend,
        emotion2vec_backend_factory=Emotion2VecBackend,
        unsupported_backend_error=AccurateModelUnavailableError,
    )


def _terminate_worker_process(process: BaseProcess) -> None:
    """Terminates a timed-out worker process with kill fallback."""
    _terminate_worker_process_orchestration(
        process=process,
        impl=_terminate_worker_process_impl,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        kill_grace_seconds=_KILL_GRACE_SECONDS,
    )


def _raise_worker_error(error_type: str, message: str) -> None:
    """Rehydrates child-process errors into runtime-domain exceptions."""
    _raise_worker_error_orchestration(
        error_type=error_type,
        message=message,
        impl=_raise_worker_error_impl,
        known_error_factories=_WORKER_ERROR_FACTORIES,
        unknown_error_factory=AccurateInferenceExecutionError,
        worker_label="Accurate inference",
    )


def _retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    return _boundary_support.retry_delay_seconds(
        base_delay=base_delay,
        attempt=attempt,
    )
