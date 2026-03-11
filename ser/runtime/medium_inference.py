"""Medium-profile inference runner with encode-once/pool-many semantics."""

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
from ser.config import AppConfig, MediumRuntimeConfig
from ser.models.emotion_model import LoadedModel, load_model
from ser.models.profile_runtime import resolve_medium_model_id
from ser.repr import EncodedSequence, PoolingWindow, XLSRBackend
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime import medium_execution as medium_execution_helpers
from ser.runtime import medium_retry_policy as medium_retry_policy_helpers
from ser.runtime import medium_worker_operation as medium_worker_operation_helpers
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_backend_runtime import (
    prepare_medium_backend_runtime as _prepare_medium_backend_runtime_impl,
)
from ser.runtime.medium_backend_runtime import (
    resolve_medium_feature_runtime_policy as _resolve_medium_feature_runtime_policy_impl,
)
from ser.runtime.medium_execution_context import MediumExecutionContext as _MediumExecutionContext
from ser.runtime.medium_execution_context import (
    prepare_execution_context as _prepare_execution_context_impl,
)
from ser.runtime.medium_execution_flow import (
    execute_medium_inference_with_retry as _execute_medium_inference_with_retry_impl,
)
from ser.runtime.medium_prediction import (
    confidence_and_probabilities as _confidence_and_probabilities_impl,
)
from ser.runtime.medium_prediction import predict_labels as _predict_labels_impl
from ser.runtime.medium_process_operation import (
    prepare_process_operation as _prepare_process_operation_impl,
)
from ser.runtime.medium_process_operation import (
    run_process_operation as _run_process_operation_impl,
)
from ser.runtime.medium_retry_operation import (
    run_medium_inference_with_retry_policy as _run_medium_retry_policy_impl,
)
from ser.runtime.medium_runtime_support import (
    build_cpu_medium_backend_for_settings as _build_cpu_medium_backend_for_settings_impl,
)
from ser.runtime.medium_runtime_support import (
    build_cpu_settings_snapshot as _build_cpu_settings_snapshot_impl,
)
from ser.runtime.medium_runtime_support import (
    build_medium_backend_for_settings as _build_medium_backend_for_settings_impl,
)
from ser.runtime.medium_runtime_support import (
    build_runtime_settings_snapshot as _build_runtime_settings_snapshot_impl,
)
from ser.runtime.medium_runtime_support import (
    encode_medium_sequence as _encode_medium_sequence_impl,
)
from ser.runtime.medium_runtime_support import (
    ensure_medium_loaded_model_compatibility as _ensure_medium_loaded_model_compatibility_impl,
)
from ser.runtime.medium_runtime_support import (
    warn_on_medium_runtime_selector_mismatch as _warn_on_medium_runtime_selector_mismatch_impl,
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
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.schema import InferenceResult
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type FeatureMatrix = NDArray[np.float64]
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
    encoded = _encode_medium_sequence_impl(
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )
    return medium_execution_helpers.run_medium_inference_once(
        loaded_model=loaded_model,
        encoded=encoded,
        runtime_config=runtime_config,
        predict_labels=lambda model, features: _predict_labels(model, features),
        confidence_and_probabilities=lambda model, features, expected_rows: (
            _confidence_and_probabilities(
                model,
                features,
                expected_rows=expected_rows,
            )
        ),
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
) -> WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_orchestration(
        connection=connection,
        stage=stage,
        impl=_recv_worker_message_impl,
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
    )


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_orchestration(
        message=message,
        impl=_is_setup_complete_message_impl,
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
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
    return medium_worker_operation_helpers.prepare_in_process_operation(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        expected_backend_model_id=expected_backend_model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        load_medium_model=lambda settings, expected_backend_model_id: load_model(
            settings=settings,
            expected_backend_id="hf_xlsr",
            expected_profile="medium",
            expected_backend_model_id=expected_backend_model_id,
        ),
        ensure_medium_compatible_model=lambda loaded_model, expected_backend_model_id: (
            _ensure_medium_loaded_model_compatibility_impl(
                loaded_model,
                expected_backend_model_id=expected_backend_model_id,
                unavailable_error_factory=MediumModelUnavailableError,
            )
        ),
        warn_on_runtime_selector_mismatch=lambda loaded_model, runtime_device, runtime_dtype: (
            _warn_on_medium_runtime_selector_mismatch_impl(
                loaded_model=loaded_model,
                profile="medium",
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                logger=logger,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=settings.audio_read,
        ),
        build_medium_backend=lambda settings, expected_backend_model_id, runtime_device, runtime_dtype: (
            _build_medium_backend_for_settings_impl(
                settings=settings,
                expected_backend_model_id=expected_backend_model_id,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                backend_factory=XLSRBackend,
            )
        ),
        model_unavailable_error_factory=MediumModelUnavailableError,
        model_load_error_factory=MediumModelLoadError,
    )


def _prepare_process_operation(
    payload: MediumProcessPayload,
) -> _PreparedMediumOperation:
    """Performs untimed setup for one process-isolated medium operation."""
    return _prepare_process_operation_impl(
        payload=payload,
        load_medium_model=lambda settings, expected_backend_model_id: load_model(
            settings=settings,
            expected_backend_id="hf_xlsr",
            expected_profile="medium",
            expected_backend_model_id=expected_backend_model_id,
        ),
        ensure_medium_compatible_model=lambda loaded_model, expected_backend_model_id: (
            _ensure_medium_loaded_model_compatibility_impl(
                loaded_model,
                expected_backend_model_id=expected_backend_model_id,
                unavailable_error_factory=MediumModelUnavailableError,
            )
        ),
        resolve_runtime_policy=lambda settings: _resolve_medium_feature_runtime_policy(
            settings=settings
        ),
        warn_on_runtime_selector_mismatch=lambda loaded_model, runtime_device, runtime_dtype: (
            _warn_on_medium_runtime_selector_mismatch_impl(
                loaded_model=loaded_model,
                profile="medium",
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                logger=logger,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=payload.settings.audio_read,
        ),
        build_medium_backend=lambda settings, expected_backend_model_id, runtime_device, runtime_dtype: (
            _build_medium_backend_for_settings_impl(
                settings=settings,
                expected_backend_model_id=expected_backend_model_id,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                backend_factory=XLSRBackend,
            )
        ),
        prepare_medium_backend_runtime=lambda backend: _prepare_medium_backend_runtime(
            backend=backend
        ),
        model_unavailable_error_factory=MediumModelUnavailableError,
        model_load_error_factory=MediumModelLoadError,
    )


def _run_process_operation(prepared: _PreparedMediumOperation) -> InferenceResult:
    """Runs one medium compute phase inside isolated worker process."""
    return _run_process_operation_impl(
        prepared,
        run_medium_inference_once=lambda loaded_model, backend, audio, sample_rate, runtime_config: (
            _run_medium_inference_once(
                loaded_model=loaded_model,
                backend=backend,
                audio=audio,
                sample_rate=sample_rate,
                runtime_config=runtime_config,
            )
        ),
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
    return _prepare_execution_context_impl(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        resolve_medium_model_id=resolve_medium_model_id,
        resolve_runtime_policy=lambda settings: _resolve_medium_feature_runtime_policy(
            settings=settings
        ),
        log_selector_adjustment=lambda device, dtype, reason: logger.info(
            "Medium feature runtime policy adjusted selectors " "(device=%s, dtype=%s, reason=%s).",
            device,
            dtype,
            reason,
        ),
        prepare_retry_state=medium_worker_operation_helpers.prepare_retry_state,
        build_process_payload=lambda expected_backend_model_id, policy_device, policy_dtype: (
            MediumProcessPayload(
                request=request,
                settings=_build_runtime_settings_snapshot_impl(
                    settings,
                    runtime_device=policy_device,
                    runtime_dtype=policy_dtype,
                ),
                expected_backend_model_id=expected_backend_model_id,
            )
        ),
        prepare_in_process_operation=_prepare_in_process_operation,
        logger=logger,
        profile="medium",
        setup_phase_name=PHASE_EMOTION_SETUP,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
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
    return cast(
        InferenceResult,
        _execute_medium_inference_with_retry_impl(
            execution_context=execution_context,
            injected_backend=injected_backend,
            enforce_timeout=enforce_timeout,
            allow_retries=allow_retries,
            logger=logger,
            profile="medium",
            setup_phase_name=PHASE_EMOTION_SETUP,
            inference_phase_name=PHASE_EMOTION_INFERENCE,
            finalize_in_process_setup=medium_worker_operation_helpers.finalize_in_process_setup,
            prepare_medium_backend_runtime=lambda active_backend: (
                _prepare_medium_backend_runtime(backend=active_backend)
            ),
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_inference_operation=medium_worker_operation_helpers.run_inference_operation,
            run_with_process_timeout=lambda payload, timeout_seconds: (
                _run_with_process_timeout(
                    payload,
                    timeout_seconds=timeout_seconds,
                )
            ),
            run_process_operation=_run_process_operation,
            run_with_timeout=lambda operation, timeout_seconds: _run_with_timeout(
                operation=operation,
                timeout_seconds=timeout_seconds,
            ),
            build_transient_failure_handler=medium_worker_operation_helpers.build_transient_failure_handler,
            should_retry_on_cpu_after_transient_failure=(
                _should_retry_on_cpu_after_transient_failure
            ),
            summarize_transient_failure=_summarize_transient_failure,
            process_payload_cpu_fallback=lambda payload: replace(
                payload,
                settings=_build_cpu_settings_snapshot_impl(payload.settings),
            ),
            in_process_cpu_backend_builder=lambda: _build_cpu_medium_backend_for_settings_impl(
                settings=settings,
                expected_backend_model_id=expected_backend_model_id,
                backend_factory=XLSRBackend,
            ),
            replace_prepared_backend=lambda prepared, active_backend: replace(
                prepared,
                backend=active_backend,
            ),
            run_retry_policy=_run_medium_retry_policy_impl,
            retry_delay_seconds=_retry_delay_seconds,
            timeout_error_type=MediumInferenceTimeoutError,
            transient_error_type=MediumTransientBackendError,
            transient_exhausted_error=lambda err: MediumInferenceExecutionError(
                "Medium inference exhausted retry budget after backend failures."
            ),
            run_with_retry_policy=run_with_retry_policy,
            passthrough_error_types=(
                MediumRuntimeDependencyError,
                ValueError,
                MediumInferenceExecutionError,
            ),
            runtime_error_factory=lambda _err: MediumInferenceExecutionError(
                "Medium inference failed with a non-retryable runtime error."
            ),
        ),
    )


def _resolve_medium_feature_runtime_policy(
    *,
    settings: AppConfig,
) -> FeatureRuntimePolicy:
    """Resolves backend-aware feature runtime selectors for medium profile."""
    return _resolve_medium_feature_runtime_policy_impl(
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
    return medium_retry_policy_helpers.retry_delay_seconds(
        base_delay=base_delay,
        attempt=attempt,
    )


def _should_retry_on_cpu_after_transient_failure(err: Exception) -> bool:
    """Returns whether one transient failure should trigger CPU fallback retry."""
    return medium_retry_policy_helpers.should_retry_on_cpu_after_transient_failure(err)


def _summarize_transient_failure(err: Exception) -> str:
    """Builds one compact summary line for medium transient fallback logs."""
    return medium_retry_policy_helpers.summarize_transient_failure(err)


def _is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    return medium_retry_policy_helpers.is_dependency_error(err)


def _prepare_medium_backend_runtime(*, backend: XLSRBackend) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    _prepare_medium_backend_runtime_impl(
        backend=backend,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    runtime_config: MediumRuntimeConfig,
) -> list[PoolingWindow]:
    """Creates medium temporal pooling windows from configured window policy."""
    return medium_execution_helpers.pooling_windows_from_encoded_frames(
        encoded,
        runtime_config=runtime_config,
    )


def _predict_labels(model: object, features: FeatureMatrix) -> list[str]:
    """Runs model prediction and validates row-aligned label output."""
    return _predict_labels_impl(
        model=model,
        features=features,
    )


def _confidence_and_probabilities(
    model: object,
    features: FeatureMatrix,
    *,
    expected_rows: int,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability mappings."""
    return _confidence_and_probabilities_impl(
        model=model,
        features=features,
        expected_rows=expected_rows,
        logger=logger,
    )
