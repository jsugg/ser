"""Accurate-profile inference runner with bounded retries and timeout guards."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

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
from ser.models.profile_runtime import (
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
)
from ser.repr import (
    Emotion2VecBackend,
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    WhisperBackend,
)
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.runtime import mps_oom as mps_oom_helpers
from ser.runtime.accurate_backend_runtime import (
    build_backend_for_profile as _build_backend_for_profile_impl,
)
from ser.runtime.accurate_backend_runtime import (
    runtime_config_for_profile as _runtime_config_for_profile_impl,
)
from ser.runtime.accurate_execution import (
    pooling_windows_from_encoded_frames as _pooling_windows_from_encoded_frames_impl,
)
from ser.runtime.accurate_execution import (
    run_accurate_inference_once as _run_accurate_inference_once_impl,
)
from ser.runtime.accurate_execution_flow import (
    execute_accurate_inference_with_retry as _execute_accurate_inference_with_retry_impl,
)
from ser.runtime.accurate_execution_flow import (
    run_accurate_retryable_operation as _run_accurate_retryable_operation_impl,
)
from ser.runtime.accurate_model_contract import (
    validate_accurate_loaded_model_runtime_contract as _validate_accurate_loaded_model_runtime_contract_impl,
)
from ser.runtime.accurate_operation_setup import (
    prepare_in_process_operation as _prepare_in_process_operation_orchestration,
)
from ser.runtime.accurate_operation_setup import (
    prepare_process_operation as _prepare_process_operation_orchestration,
)
from ser.runtime.accurate_operation_setup import (
    run_process_operation as _run_process_operation_orchestration,
)
from ser.runtime.accurate_prediction import (
    confidence_and_probabilities as _confidence_and_probabilities_impl,
)
from ser.runtime.accurate_prediction import predict_labels as _predict_labels_impl
from ser.runtime.accurate_retry_operation import (
    run_accurate_inference_with_retry_policy as _run_accurate_retry_policy_impl,
)
from ser.runtime.accurate_runtime_support import (
    build_cpu_settings_snapshot as _build_cpu_settings_snapshot_impl,
)
from ser.runtime.accurate_runtime_support import (
    build_process_settings_snapshot as _build_process_settings_snapshot_impl,
)
from ser.runtime.accurate_runtime_support import (
    encode_accurate_sequence as _encode_accurate_sequence_impl,
)
from ser.runtime.accurate_runtime_support import (
    prepare_accurate_backend_runtime as _prepare_accurate_backend_runtime_support_impl,
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
    build_transient_failure_handler,
)
from ser.runtime.accurate_worker_operation import (
    finalize_in_process_setup as _finalize_in_process_setup_impl,
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
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.retry_primitives import (
    jittered_retry_delay_seconds as _jittered_retry_delay_seconds_impl,
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
type _PreparedAccurateOperation = PreparedAccurateOperation[
    LoadedModel,
    FeatureBackend,
]

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
    return cast(
        InferenceResult,
        _execute_accurate_inference_with_retry_impl(
            use_process_isolation=use_process_isolation,
            retry_state=retry_state,
            prepared_operation=prepared_operation,
            setup_started_at=setup_started_at,
            settings=settings,
            timeout_seconds=runtime_config.timeout_seconds,
            backend=backend,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            allow_retries=allow_retries,
            logger=logger,
            setup_phase_name=PHASE_EMOTION_SETUP,
            finalize_in_process_setup=_finalize_in_process_setup_impl,
            prepare_accurate_backend_runtime=lambda active_backend: (
                _prepare_accurate_backend_runtime_support_impl(
                    active_backend,
                    dependency_error_factory=AccurateRuntimeDependencyError,
                    transient_error_factory=AccurateTransientBackendError,
                )
            ),
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            build_transient_failure_handler=build_transient_failure_handler,
            should_retry_on_cpu_after_transient_failure=(
                mps_oom_helpers.is_mps_out_of_memory_error
            ),
            summarize_transient_failure=mps_oom_helpers.summarize_mps_oom_memory,
            process_payload_cpu_fallback=_payload_with_cpu_settings,
            cpu_backend_builder=cpu_backend_builder,
            run_retry_policy=_run_accurate_retry_policy_impl,
            retry_delay_seconds=_retry_delay_seconds,
            run_with_retry_policy=run_with_retry_policy,
            passthrough_error_types=(
                AccurateRuntimeDependencyError,
                ValueError,
                AccurateInferenceExecutionError,
            ),
            run_accurate_retryable_operation=_run_accurate_retryable_operation,
            timeout_error_type=AccurateInferenceTimeoutError,
            transient_error_type=AccurateTransientBackendError,
            transient_exhausted_error=lambda _err: AccurateInferenceExecutionError(
                "Accurate inference exhausted retry budget after backend failures."
            ),
            runtime_error_factory=lambda _err: AccurateInferenceExecutionError(
                "Accurate inference failed with a non-retryable runtime error."
            ),
            enforce_timeout=enforce_timeout,
        ),
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
    return cast(
        InferenceResult,
        _run_accurate_retryable_operation_impl(
            enforce_timeout=enforce_timeout,
            use_process_isolation=use_process_isolation,
            retry_state=retry_state,
            prepared_operation=prepared_operation,
            timeout_seconds=timeout_seconds,
            expected_profile=expected_profile,
            logger=logger,
            inference_phase_name=PHASE_EMOTION_INFERENCE,
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_with_process_timeout=_run_with_process_timeout,
            run_accurate_inference_once=_run_accurate_inference_once,
            run_with_timeout=_run_with_timeout_impl,
            run_inference_operation=_run_inference_operation_impl,
            timeout_error_factory=AccurateInferenceTimeoutError,
            runtime_error_factory=RuntimeError,
        ),
    )


def _payload_with_cpu_settings(
    payload: AccurateProcessPayload,
) -> AccurateProcessPayload:
    """Returns one process payload updated to use CPU torch selectors."""
    return replace(
        payload,
        settings=_build_cpu_settings_snapshot_impl(payload.settings),
    )


def _run_accurate_inference_once(
    *,
    loaded_model: LoadedModel,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: ProfileRuntimeConfig,
) -> InferenceResult:
    """Runs one accurate inference attempt without retry control."""
    encoded = _encode_accurate_sequence_impl(
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        dependency_error_factory=AccurateRuntimeDependencyError,
        transient_error_factory=AccurateTransientBackendError,
    )
    return _run_accurate_inference_once_impl(
        loaded_model=loaded_model,
        encoded=encoded,
        runtime_config=runtime_config,
        predict_labels=lambda model, features: _predict_labels_impl(
            model=model,
            features=features,
        ),
        confidence_and_probabilities=lambda model, features, expected_rows: (
            _confidence_and_probabilities_impl(
                model=model,
                features=features,
                expected_rows=expected_rows,
                logger=logger,
            )
        ),
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
) -> WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_orchestration(
        connection=connection,
        stage=stage,
        impl=_recv_worker_message_impl,
        worker_label="Accurate inference",
        error_factory=AccurateInferenceExecutionError,
    )


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_orchestration(
        message=message,
        impl=_is_setup_complete_message_impl,
        worker_label="Accurate inference",
        error_factory=AccurateInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
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
    return _prepare_in_process_operation_orchestration(
        request=request,
        settings=settings,
        runtime_config=runtime_config,
        loaded_model=loaded_model,
        backend=backend,
        load_accurate_model=lambda active_settings: load_model(
            settings=active_settings,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=expected_backend_model_id,
        ),
        validate_loaded_model=lambda active_loaded_model: (
            _validate_accurate_loaded_model_runtime_contract_impl(
                active_loaded_model,
                settings=settings,
                expected_backend_id=expected_backend_id,
                expected_profile=expected_profile,
                expected_backend_model_id=expected_backend_model_id,
                unavailable_error_factory=AccurateModelUnavailableError,
                logger=logger,
                resolve_runtime_policy=resolve_feature_runtime_policy,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=settings.audio_read,
        ),
        build_backend_for_profile=lambda active_settings: _build_backend_for_profile(
            expected_backend_id=expected_backend_id,
            expected_backend_model_id=expected_backend_model_id,
            settings=active_settings,
        ),
        model_unavailable_error_factory=AccurateModelUnavailableError,
        model_load_error_factory=AccurateModelLoadError,
    )


def _prepare_process_operation(
    payload: AccurateProcessPayload,
) -> _PreparedAccurateOperation:
    """Performs untimed setup for one process-isolated accurate operation."""
    return _prepare_process_operation_orchestration(
        payload=payload,
        resolve_runtime_config=lambda settings, expected_profile: (
            _runtime_config_for_profile_impl(
                settings=settings,
                expected_profile=expected_profile,
                unsupported_profile_error=AccurateModelUnavailableError,
            )
        ),
        load_accurate_model=lambda active_payload: load_model(
            settings=active_payload.settings,
            expected_backend_id=active_payload.expected_backend_id,
            expected_profile=active_payload.expected_profile,
            expected_backend_model_id=active_payload.expected_backend_model_id,
        ),
        validate_loaded_model=lambda loaded_model, active_payload: (
            _validate_accurate_loaded_model_runtime_contract_impl(
                loaded_model,
                settings=active_payload.settings,
                expected_backend_id=active_payload.expected_backend_id,
                expected_profile=active_payload.expected_profile,
                expected_backend_model_id=active_payload.expected_backend_model_id,
                unavailable_error_factory=AccurateModelUnavailableError,
                logger=logger,
                resolve_runtime_policy=resolve_feature_runtime_policy,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=payload.settings.audio_read,
        ),
        build_backend_for_payload=lambda active_payload: _build_backend_for_profile(
            expected_backend_id=active_payload.expected_backend_id,
            expected_backend_model_id=active_payload.expected_backend_model_id,
            settings=active_payload.settings,
        ),
        prepare_accurate_backend_runtime=lambda backend: (
            _prepare_accurate_backend_runtime_support_impl(
                backend,
                dependency_error_factory=AccurateRuntimeDependencyError,
                transient_error_factory=AccurateTransientBackendError,
            )
        ),
        model_unavailable_error_factory=AccurateModelUnavailableError,
        model_load_error_factory=AccurateModelLoadError,
    )


def _run_process_operation(prepared: _PreparedAccurateOperation) -> InferenceResult:
    """Runs one accurate compute phase inside isolated worker process."""
    return _run_process_operation_orchestration(
        prepared,
        run_accurate_inference_once=lambda loaded_model, backend, audio, sample_rate, runtime_config: (
            _run_accurate_inference_once(
                loaded_model=loaded_model,
                backend=backend,
                audio=audio,
                sample_rate=sample_rate,
                runtime_config=runtime_config,
            )
        ),
    )


def _build_backend_for_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
) -> FeatureBackend:
    """Builds a feature backend aligned with profile/backend runtime expectations."""
    return _build_backend_for_profile_impl(
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=expected_backend_model_id,
        settings=settings,
        resolve_accurate_model_id=resolve_accurate_model_id,
        resolve_accurate_research_model_id=resolve_accurate_research_model_id,
        resolve_runtime_policy=resolve_feature_runtime_policy,
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
    return _jittered_retry_delay_seconds_impl(
        base_delay=base_delay,
        attempt=attempt,
    )


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    runtime_config: ProfileRuntimeConfig,
) -> list[PoolingWindow]:
    """Creates accurate temporal pooling windows from configured window policy."""
    return _pooling_windows_from_encoded_frames_impl(
        encoded,
        runtime_config=runtime_config,
    )
