"""Medium-profile inference runner with encode-once/pool-many semantics."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ser._internal.runtime import medium_public_boundary as _boundary_support
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
from ser.config import AppConfig, MediumRuntimeConfig
from ser.models.emotion_model import LoadedModel, load_model
from ser.models.profile_runtime import resolve_medium_model_id
from ser.repr import XLSRBackend
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime import medium_worker_operation as medium_worker_operation_helpers
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_execution_context import MediumExecutionContext as _MediumExecutionContext
from ser.runtime.medium_retry_operation import (
    run_medium_inference_with_retry_policy as _run_medium_retry_policy_impl,
)
from ser.runtime.medium_runtime_support import (
    build_cpu_medium_backend_for_settings as _build_cpu_medium_backend_for_settings_impl,
)
from ser.runtime.medium_runtime_support import (
    build_runtime_settings_snapshot as _build_runtime_settings_snapshot_impl,
)
from ser.runtime.medium_worker_lifecycle import (
    is_setup_complete_message as _is_setup_complete_message_orchestration,
)
from ser.runtime.medium_worker_lifecycle import (
    parse_worker_completion_message as _parse_worker_completion_message_orchestration,
)
from ser.runtime.medium_worker_lifecycle import (
    raise_worker_error as _raise_worker_error_orchestration,
)
from ser.runtime.medium_worker_lifecycle import (
    recv_worker_message as _recv_worker_message_orchestration,
)
from ser.runtime.medium_worker_lifecycle import (
    run_with_process_timeout as _run_with_process_timeout_orchestration,
)
from ser.runtime.medium_worker_lifecycle import run_worker_entry as _run_worker_entry_orchestration
from ser.runtime.medium_worker_lifecycle import (
    terminate_worker_process as _terminate_worker_process_orchestration,
)
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

type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_REGISTRY = SingleFlightRegistry()


@dataclass(frozen=True)
class MediumProcessPayload:
    """Serializable payload for one process-isolated medium inference attempt."""

    request: InferenceRequest
    settings: AppConfig
    expected_backend_model_id: str


type _PreparedMediumOperation = medium_worker_operation_helpers.PreparedMediumOperation[
    LoadedModel,
    XLSRBackend,
]
type _MediumRetryOperationState = (
    medium_worker_operation_helpers.MediumRetryOperationState[
        MediumProcessPayload,
        LoadedModel,
        XLSRBackend,
    ]
)
type _PreparedMediumExecutionContext = _MediumExecutionContext[
    MediumProcessPayload,
    LoadedModel,
    XLSRBackend,
]


class MediumModelUnavailableError(FileNotFoundError):
    """Raised when a compatible medium-profile model artifact is unavailable."""


class MediumRuntimeDependencyError(RuntimeError):
    """Raised when medium optional runtime dependencies are missing."""


class MediumModelLoadError(RuntimeError):
    """Raised when medium model artifact loading fails unexpectedly."""


class MediumInferenceTimeoutError(TimeoutError):
    """Raised when medium inference exceeds configured timeout budget."""


class MediumInferenceExecutionError(RuntimeError):
    """Raised when medium inference exhausts retries without recovery."""


class MediumTransientBackendError(RuntimeError):
    """Raised for retryable medium backend encoding failures."""


_WORKER_ERROR_FACTORIES: dict[str, Callable[[str], Exception]] = {
    "ValueError": ValueError,
    "MediumRuntimeDependencyError": MediumRuntimeDependencyError,
    "MediumTransientBackendError": MediumTransientBackendError,
    "MediumModelUnavailableError": MediumModelUnavailableError,
    "MediumModelLoadError": MediumModelLoadError,
    "MediumInferenceTimeoutError": MediumInferenceTimeoutError,
    "RuntimeError": RuntimeError,
}


def run_medium_inference(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    backend: XLSRBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
) -> InferenceResult:
    """Runs medium-profile inference with bounded retries and timeout budget.

    Args:
        request: Runtime inference request payload.
        settings: Active application settings.
        loaded_model: Optional preloaded model artifact for repeated inference calls.
        backend: Optional preinitialized XLSR backend for repeated inference calls.
        enforce_timeout: Whether to apply timeout wrapper for each inference attempt.
        allow_retries: Whether to apply configured retry budget for retryable errors.

    Returns:
        Detailed inference result with frame and segment predictions.

    Raises:
        MediumModelUnavailableError: If no compatible medium artifact exists.
        MediumRuntimeDependencyError: If required medium dependencies are missing.
        MediumModelLoadError: If model loading fails for non-compatibility reasons.
        MediumInferenceTimeoutError: If all attempts timed out.
        MediumInferenceExecutionError: If transient backend failures exhaust retries.
        ValueError: If feature dimensions are incompatible with loaded artifact.
    """
    execution_context = _prepare_execution_context(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
    )
    expected_backend_model_id = execution_context.expected_backend_model_id

    with _SINGLE_FLIGHT_REGISTRY.lock(
        profile="medium",
        backend_model_id=expected_backend_model_id,
    ):
        return _execute_medium_inference_with_retry(
            execution_context=execution_context,
            settings=settings,
            injected_backend=backend,
            enforce_timeout=enforce_timeout,
            allow_retries=allow_retries,
            expected_backend_model_id=expected_backend_model_id,
        )


def _run_with_timeout(
    *,
    operation: Callable[[], InferenceResult],
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one in-process medium inference attempt under timeout budget."""
    return _run_with_timeout_impl(
        operation=operation,
        timeout_seconds=timeout_seconds,
        timeout_error_factory=MediumInferenceTimeoutError,
        timeout_label="Medium inference",
    )


def _run_medium_inference_once(
    *,
    loaded_model: LoadedModel,
    backend: XLSRBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: MediumRuntimeConfig,
) -> InferenceResult:
    """Runs one medium inference attempt without retry control."""
    return _boundary_support.run_medium_inference_once(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
        logger=logger,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )


def _run_with_process_timeout(
    payload: MediumProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    return _run_with_process_timeout_orchestration(
        payload=payload,
        profile="medium",
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
        timeout_error_factory=MediumInferenceTimeoutError,
        execution_error_factory=MediumInferenceExecutionError,
        worker_label="Medium inference",
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
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
    )


def _is_setup_complete_message(message: tuple[object, ...]) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_orchestration(
        message=message,
        impl=_is_setup_complete_message_impl,
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: tuple[object, ...]) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    return _parse_worker_completion_message_orchestration(
        worker_message=worker_message,
        impl=_parse_worker_completion_message_impl,
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
        raise_worker_error=_raise_worker_error,
        result_type=InferenceResult,
    )


def _worker_entry(
    payload: MediumProcessPayload,
    connection: Connection,
) -> None:
    """Executes one medium inference operation inside a child process."""
    _run_worker_entry_orchestration(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_process_operation,
        run_process_operation=_run_process_operation,
    )


def _prepare_in_process_operation(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: LoadedModel | None,
    backend: XLSRBackend | None,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
) -> _PreparedMediumOperation:
    """Performs untimed setup for one in-process medium operation."""
    return cast(
        _PreparedMediumOperation,
        _boundary_support.prepare_in_process_operation(
            request=request,
            settings=settings,
            loaded_model=loaded_model,
            backend=backend,
            expected_backend_model_id=expected_backend_model_id,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
            load_model_fn=load_model,
            read_audio_file_fn=read_audio_file,
            backend_factory=XLSRBackend,
            logger=logger,
            model_unavailable_error_factory=MediumModelUnavailableError,
            model_load_error_factory=MediumModelLoadError,
        ),
    )


def _prepare_process_operation(
    payload: MediumProcessPayload,
) -> _PreparedMediumOperation:
    """Performs untimed setup for one process-isolated medium operation."""
    return cast(
        _PreparedMediumOperation,
        _boundary_support.prepare_process_operation(
            payload,
            load_model_fn=load_model,
            read_audio_file_fn=read_audio_file,
            backend_factory=XLSRBackend,
            resolve_runtime_policy=lambda settings: _resolve_medium_feature_runtime_policy(
                settings=settings
            ),
            logger=logger,
            model_unavailable_error_factory=MediumModelUnavailableError,
            model_load_error_factory=MediumModelLoadError,
            prepare_medium_backend_runtime=lambda active_backend: _prepare_medium_backend_runtime(
                backend=active_backend
            ),
        ),
    )


def _run_process_operation(prepared: _PreparedMediumOperation) -> InferenceResult:
    """Runs one medium compute phase inside isolated worker process."""
    return _boundary_support.run_process_operation(
        prepared,
        run_medium_inference_once=lambda **kwargs: _run_medium_inference_once(**kwargs),
    )


def _prepare_execution_context(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: LoadedModel | None,
    backend: XLSRBackend | None,
    enforce_timeout: bool,
) -> _PreparedMediumExecutionContext:
    """Resolves pre-lock runtime context for medium inference execution."""
    return cast(
        _PreparedMediumExecutionContext,
        _boundary_support.prepare_execution_context(
            request=request,
            settings=settings,
            loaded_model=loaded_model,
            backend=backend,
            enforce_timeout=enforce_timeout,
            resolve_medium_model_id=resolve_medium_model_id,
            resolve_runtime_policy=lambda active_settings: _resolve_medium_feature_runtime_policy(
                settings=active_settings
            ),
            prepare_retry_state=medium_worker_operation_helpers.prepare_retry_state,
            prepare_in_process_operation=_prepare_in_process_operation,
            build_process_payload=lambda backend_model_id, policy_device, policy_dtype: (
                MediumProcessPayload(
                    request=request,
                    settings=_build_runtime_settings_snapshot_impl(
                        settings,
                        runtime_device=policy_device,
                        runtime_dtype=policy_dtype,
                    ),
                    expected_backend_model_id=backend_model_id,
                )
            ),
            logger=logger,
        ),
    )


def _execute_medium_inference_with_retry(
    *,
    execution_context: _PreparedMediumExecutionContext,
    settings: AppConfig,
    injected_backend: XLSRBackend | None,
    enforce_timeout: bool,
    allow_retries: bool,
    expected_backend_model_id: str,
) -> InferenceResult:
    """Executes medium inference inside the single-flight lock."""
    return _boundary_support.execute_medium_inference_with_retry(
        execution_context=execution_context,
        settings=settings,
        injected_backend=injected_backend,
        enforce_timeout=enforce_timeout,
        allow_retries=allow_retries,
        expected_backend_model_id=expected_backend_model_id,
        logger=logger,
        run_with_process_timeout=lambda payload, timeout_seconds: _run_with_process_timeout(
            payload,
            timeout_seconds=timeout_seconds,
        ),
        run_process_operation=_run_process_operation,
        run_with_timeout=lambda operation, timeout_seconds: _run_with_timeout(
            operation=operation,
            timeout_seconds=timeout_seconds,
        ),
        prepare_medium_backend_runtime=lambda active_backend: _prepare_medium_backend_runtime(
            backend=active_backend
        ),
        cpu_backend_builder=lambda: _build_cpu_medium_backend_for_settings_impl(
            settings=settings,
            expected_backend_model_id=expected_backend_model_id,
            backend_factory=XLSRBackend,
        ),
        timeout_error_type=MediumInferenceTimeoutError,
        transient_error_type=MediumTransientBackendError,
        runtime_dependency_error_type=MediumRuntimeDependencyError,
        execution_error_type=MediumInferenceExecutionError,
        run_retry_policy_impl=_run_medium_retry_policy_impl,
        retry_delay_seconds=_retry_delay_seconds,
        should_retry_on_cpu_after_transient_failure=_should_retry_on_cpu_after_transient_failure,
        summarize_transient_failure=_summarize_transient_failure,
    )


def _resolve_medium_feature_runtime_policy(
    *,
    settings: AppConfig,
) -> FeatureRuntimePolicy:
    """Resolves backend-aware feature runtime selectors for medium profile."""
    return _boundary_support.resolve_medium_feature_runtime_policy(
        settings=settings,
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
        unknown_error_factory=MediumInferenceExecutionError,
        worker_label="Medium inference",
    )


def _retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    return _boundary_support.retry_delay_seconds(
        base_delay=base_delay,
        attempt=attempt,
    )


def _should_retry_on_cpu_after_transient_failure(err: Exception) -> bool:
    """Returns whether one transient failure should trigger CPU fallback retry."""
    return _boundary_support.should_retry_on_cpu_after_transient_failure(err)


def _summarize_transient_failure(err: Exception) -> str:
    """Builds one compact summary line for medium transient fallback logs."""
    return _boundary_support.summarize_transient_failure(err)


def _is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    return _boundary_support.is_dependency_error(err)


def _prepare_medium_backend_runtime(*, backend: XLSRBackend) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    _boundary_support.prepare_medium_backend_runtime(
        backend=backend,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )
