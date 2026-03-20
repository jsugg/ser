"""Internal support owners for accurate inference public-boundary wrappers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, Literal, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ser._internal.runtime.process_timeout import (
    run_with_process_timeout as _run_with_process_timeout_impl,
)
from ser._internal.runtime.single_flight import SingleFlightRegistry
from ser._internal.runtime.worker_bindings import (
    is_setup_complete_message as _is_setup_complete_message_binding,
)
from ser._internal.runtime.worker_bindings import (
    parse_worker_completion_message as _parse_worker_completion_message_binding,
)
from ser._internal.runtime.worker_bindings import raise_worker_error as _raise_worker_error_binding
from ser._internal.runtime.worker_bindings import (
    recv_worker_message as _recv_worker_message_binding,
)
from ser._internal.runtime.worker_bindings import run_worker_entry as _run_worker_entry_binding
from ser._internal.runtime.worker_bindings import (
    terminate_worker_process as _terminate_worker_process_binding,
)
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
from ser.models.emotion_model import load_model
from ser.models.profile_runtime import (
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
)
from ser.repr import Emotion2VecBackend, FeatureBackend, WhisperBackend
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.runtime import mps_oom as mps_oom_helpers
from ser.runtime.accurate_backend_runtime import (
    build_backend_for_profile as _build_backend_for_profile_impl,
)
from ser.runtime.accurate_backend_runtime import (
    runtime_config_for_profile as _runtime_config_for_profile_impl,
)
from ser.runtime.accurate_execution import LoadedModelLike as _AccurateLoadedModelLike
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
    ArtifactMetadataCarrier,
)
from ser.runtime.accurate_model_contract import (
    validate_accurate_loaded_model_runtime_contract as _validate_accurate_loaded_model_runtime_contract_impl,
)
from ser.runtime.accurate_operation_setup import _PayloadLike as _AccuratePayloadLike
from ser.runtime.accurate_operation_setup import _RequestLike as _AccurateRequestLike
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


class _AccurateLoadedModel(ArtifactMetadataCarrier, _AccurateLoadedModelLike, Protocol):
    """Loaded-model contract required by accurate public-boundary helpers."""


_AccurateLoadedModelT = TypeVar("_AccurateLoadedModelT", bound=_AccurateLoadedModel)
_AccuratePayloadT = TypeVar("_AccuratePayloadT", bound=_AccuratePayloadLike)

type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_REGISTRY = SingleFlightRegistry()
_WORKER_LOGGER = logging.getLogger("ser.runtime.accurate_inference")


@dataclass(frozen=True)
class AccurateProcessPayload:
    """Serializable payload for one process-isolated accurate inference attempt."""

    request: InferenceRequest
    settings: AppConfig
    expected_backend_id: str
    expected_profile: str
    expected_backend_model_id: str | None


class AccurateModelUnavailableError(FileNotFoundError):
    """Spawn-safe accurate worker error marker for unavailable model artifacts."""


class AccurateRuntimeDependencyError(RuntimeError):
    """Spawn-safe accurate worker error marker for missing runtime dependencies."""


class AccurateModelLoadError(RuntimeError):
    """Spawn-safe accurate worker error marker for model load failures."""


class AccurateTransientBackendError(RuntimeError):
    """Spawn-safe accurate worker error marker for transient backend failures."""


@dataclass(frozen=True)
class _AccurateBoundaryDependencies:
    """Precomputed collaborators and execution plan for accurate boundary orchestration."""

    runtime_config: ProfileRuntimeConfig
    resolved_expected_backend_model_id: str | None
    use_process_isolation: bool
    process_payload: AccurateProcessPayload | None
    cpu_backend_builder: Callable[[], FeatureBackend]
    prepare_in_process_operation: Callable[
        ...,
        PreparedAccurateOperation[_AccurateLoadedModel, FeatureBackend],
    ]
    run_with_process_timeout: Callable[..., InferenceResult]
    run_inference_once: Callable[..., InferenceResult]


def _build_backend_for_worker_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
) -> FeatureBackend:
    """Builds one accurate worker backend with spawn-safe error types."""
    return build_backend_for_profile(
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=expected_backend_model_id,
        settings=settings,
        whisper_backend_factory=WhisperBackend,
        emotion2vec_backend_factory=Emotion2VecBackend,
        unsupported_backend_error=AccurateModelUnavailableError,
    )


def _prepare_accurate_process_operation(
    payload: AccurateProcessPayload,
) -> PreparedAccurateOperation[_AccurateLoadedModel, FeatureBackend]:
    """Builds one accurate worker operation using module-level collaborators."""
    return prepare_process_operation(
        payload,
        load_model_fn=load_model,
        read_audio_file_fn=read_audio_file,
        build_backend_for_profile_fn=_build_backend_for_worker_profile,
        logger=_WORKER_LOGGER,
        model_unavailable_error_factory=AccurateModelUnavailableError,
        model_load_error_factory=AccurateModelLoadError,
        runtime_dependency_error_factory=AccurateRuntimeDependencyError,
        transient_error_factory=AccurateTransientBackendError,
    )


def _run_accurate_process_inference_once(
    *,
    loaded_model: _AccurateLoadedModel,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: ProfileRuntimeConfig,
) -> InferenceResult:
    """Runs one accurate worker-process inference attempt."""
    return run_accurate_inference_once(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
        logger=_WORKER_LOGGER,
        dependency_error_factory=AccurateRuntimeDependencyError,
        transient_error_factory=AccurateTransientBackendError,
    )


def _run_accurate_process_operation(
    prepared: PreparedAccurateOperation[_AccurateLoadedModel, FeatureBackend],
) -> InferenceResult:
    """Runs one accurate compute phase inside the spawned worker process."""
    return run_process_operation(
        prepared,
        run_accurate_inference_once=_run_accurate_process_inference_once,
    )


def _accurate_worker_entry(payload: AccurateProcessPayload, connection: Connection) -> None:
    """Executes one spawned accurate worker using module-level collaborators only."""
    _run_worker_entry_binding(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_accurate_process_operation,
        run_process_operation=_run_accurate_process_operation,
    )


def run_accurate_inference_once(
    *,
    loaded_model: _AccurateLoadedModel,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: ProfileRuntimeConfig,
    logger: logging.Logger,
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> InferenceResult:
    """Runs one accurate inference attempt without retry control."""
    encoded = _encode_accurate_sequence_impl(
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        dependency_error_factory=dependency_error_factory,
        transient_error_factory=transient_error_factory,
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


def _build_accurate_boundary_dependencies(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: _AccurateLoadedModel | None,
    backend: FeatureBackend | None,
    enforce_timeout: bool,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
    logger: logging.Logger,
    model_unavailable_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    model_load_error_type: type[Exception],
    timeout_error_type: type[Exception],
    inference_execution_error_type: type[Exception],
    transient_backend_error_type: type[Exception],
) -> _AccurateBoundaryDependencies:
    """Builds accurate boundary collaborators and execution plan once per invocation."""
    worker_error_factories: dict[str, Callable[[str], Exception]] = {
        "ValueError": ValueError,
        runtime_dependency_error_type.__name__: runtime_dependency_error_type,
        transient_backend_error_type.__name__: transient_backend_error_type,
        model_unavailable_error_type.__name__: model_unavailable_error_type,
        model_load_error_type.__name__: model_load_error_type,
        timeout_error_type.__name__: timeout_error_type,
        "RuntimeError": RuntimeError,
    }

    def _run_with_process_timeout(
        payload: AccurateProcessPayload,
        *,
        timeout_seconds: float,
    ) -> InferenceResult:
        return _run_with_process_timeout_impl(
            payload=payload,
            resolve_profile=lambda active_payload: active_payload.expected_profile,
            timeout_seconds=timeout_seconds,
            get_context=mp.get_context,
            logger=logger,
            setup_phase_name=PHASE_EMOTION_SETUP,
            inference_phase_name=PHASE_EMOTION_INFERENCE,
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_process_setup_compute_handshake=_run_process_setup_compute_handshake_impl,
            worker_target=_accurate_worker_entry,
            recv_worker_message=_recv_worker_message,
            is_setup_complete_message=_is_setup_complete_message,
            terminate_worker_process=_terminate_worker_process,
            timeout_error_factory=timeout_error_type,
            execution_error_factory=inference_execution_error_type,
            worker_label="Accurate inference",
            process_join_grace_seconds=_TERMINATE_GRACE_SECONDS,
            parse_worker_completion_message=_parse_worker_completion_message,
        )

    def _recv_worker_message(
        connection: Connection,
        *,
        stage: str,
    ) -> WorkerMessage:
        return _recv_worker_message_binding(
            connection=connection,
            stage=stage,
            impl=_recv_worker_message_impl,
            worker_label="Accurate inference",
            error_factory=inference_execution_error_type,
        )

    def _is_setup_complete_message(message: WorkerMessage) -> bool:
        return _is_setup_complete_message_binding(
            message=message,
            impl=_is_setup_complete_message_impl,
            worker_label="Accurate inference",
            error_factory=inference_execution_error_type,
        )

    def _terminate_worker_process(process: BaseProcess) -> None:
        _terminate_worker_process_binding(
            process=process,
            impl=_terminate_worker_process_impl,
            terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
            kill_grace_seconds=_KILL_GRACE_SECONDS,
        )

    def _raise_worker_error(error_type: str, message: str) -> None:
        _raise_worker_error_binding(
            error_type=error_type,
            message=message,
            impl=_raise_worker_error_impl,
            known_error_factories=worker_error_factories,
            unknown_error_factory=inference_execution_error_type,
            worker_label="Accurate inference",
        )

    def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
        return _parse_worker_completion_message_binding(
            worker_message=worker_message,
            impl=_parse_worker_completion_message_impl,
            worker_label="Accurate inference",
            error_factory=inference_execution_error_type,
            raise_worker_error=_raise_worker_error,
            result_type=InferenceResult,
        )

    runtime_config = _runtime_config_for_profile_impl(
        settings=settings,
        expected_profile=expected_profile,
        unsupported_profile_error=model_unavailable_error_type,
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
    process_payload = (
        AccurateProcessPayload(
            request=request,
            settings=_build_process_settings_snapshot_impl(settings),
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=resolved_expected_backend_model_id,
        )
        if use_process_isolation
        else None
    )

    def _build_backend_for_profile(
        *,
        expected_backend_id: str,
        expected_backend_model_id: str | None,
        settings: AppConfig,
        expected_profile: str | None = None,
    ) -> FeatureBackend:
        del expected_profile
        return build_backend_for_profile(
            expected_backend_id=expected_backend_id,
            expected_backend_model_id=expected_backend_model_id,
            settings=settings,
            whisper_backend_factory=WhisperBackend,
            emotion2vec_backend_factory=Emotion2VecBackend,
            unsupported_backend_error=model_unavailable_error_type,
        )

    def _prepare_in_process_accurate_operation(
        *,
        request: _AccurateRequestLike,
        settings: AppConfig,
        runtime_config: ProfileRuntimeConfig,
        loaded_model: _AccurateLoadedModel | None,
        backend: FeatureBackend | None,
        expected_backend_id: str,
        expected_profile: str,
        expected_backend_model_id: str | None,
    ) -> PreparedAccurateOperation[_AccurateLoadedModel, FeatureBackend]:
        return prepare_in_process_accurate_operation(
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
            model_unavailable_error_factory=model_unavailable_error_type,
            model_load_error_factory=model_load_error_type,
        )

    def _run_accurate_inference_once(
        *,
        loaded_model: _AccurateLoadedModel,
        backend: FeatureBackend,
        audio: NDArray[np.float32],
        sample_rate: int,
        runtime_config: ProfileRuntimeConfig,
    ) -> InferenceResult:
        return run_accurate_inference_once(
            loaded_model=loaded_model,
            backend=backend,
            audio=audio,
            sample_rate=sample_rate,
            runtime_config=runtime_config,
            logger=logger,
            dependency_error_factory=runtime_dependency_error_type,
            transient_error_factory=transient_backend_error_type,
        )

    cpu_settings = _build_cpu_settings_snapshot_impl(settings)
    cpu_backend_builder: Callable[[], FeatureBackend] = partial(
        _build_backend_for_profile,
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=resolved_expected_backend_model_id,
        settings=cpu_settings,
        expected_profile=expected_profile,
    )
    return _AccurateBoundaryDependencies(
        runtime_config=runtime_config,
        resolved_expected_backend_model_id=resolved_expected_backend_model_id,
        use_process_isolation=use_process_isolation,
        process_payload=process_payload,
        cpu_backend_builder=cpu_backend_builder,
        prepare_in_process_operation=_prepare_in_process_accurate_operation,
        run_with_process_timeout=_run_with_process_timeout,
        run_inference_once=_run_accurate_inference_once,
    )


def run_accurate_inference_from_public_boundary(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: _AccurateLoadedModel | None = None,
    backend: FeatureBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
    expected_backend_id: str = "hf_whisper",
    expected_profile: str = "accurate",
    expected_backend_model_id: str | None = None,
    logger: logging.Logger,
    model_unavailable_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    model_load_error_type: type[Exception],
    timeout_error_type: type[Exception],
    inference_execution_error_type: type[Exception],
    transient_backend_error_type: type[Exception],
) -> InferenceResult:
    """Runs accurate inference through the internal public-boundary owner."""
    dependencies = _build_accurate_boundary_dependencies(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
        logger=logger,
        model_unavailable_error_type=model_unavailable_error_type,
        runtime_dependency_error_type=runtime_dependency_error_type,
        model_load_error_type=model_load_error_type,
        timeout_error_type=timeout_error_type,
        inference_execution_error_type=inference_execution_error_type,
        transient_backend_error_type=transient_backend_error_type,
    )

    retry_state, prepared_operation, setup_started_at = _prepare_retry_state_impl(
        use_process_isolation=dependencies.use_process_isolation,
        request=request,
        settings=settings,
        runtime_config=dependencies.runtime_config,
        loaded_model=loaded_model,
        backend=backend,
        logger=logger,
        profile=expected_profile,
        setup_phase_name=PHASE_EMOTION_SETUP,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
        process_payload=dependencies.process_payload,
        prepare_in_process_operation=partial(
            dependencies.prepare_in_process_operation,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=dependencies.resolved_expected_backend_model_id,
        ),
    )
    with _SINGLE_FLIGHT_REGISTRY.lock(
        profile=expected_profile,
        backend_model_id=dependencies.resolved_expected_backend_model_id,
    ):
        return execute_accurate_inference_with_retry(
            use_process_isolation=dependencies.use_process_isolation,
            retry_state=retry_state,
            prepared_operation=prepared_operation,
            setup_started_at=setup_started_at,
            settings=settings,
            backend=backend,
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            allow_retries=allow_retries,
            enforce_timeout=enforce_timeout,
            cpu_backend_builder=dependencies.cpu_backend_builder,
            logger=logger,
            run_accurate_retryable_operation=lambda **kwargs: run_accurate_retryable_operation(
                logger=logger,
                run_with_process_timeout=dependencies.run_with_process_timeout,
                run_accurate_inference_once=dependencies.run_inference_once,
                run_with_timeout=_run_with_timeout_impl,
                run_inference_operation=_run_inference_operation_impl,
                timeout_error_factory=timeout_error_type,
                **kwargs,
            ),
            retry_delay_seconds=retry_delay_seconds,
            process_payload_cpu_fallback=payload_with_cpu_settings,
            timeout_error_type=timeout_error_type,
            runtime_dependency_error_type=runtime_dependency_error_type,
            inference_execution_error_type=inference_execution_error_type,
            transient_backend_error_type=transient_backend_error_type,
        )


def run_accurate_retryable_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[_AccuratePayloadT, FeatureBackend],
    prepared_operation: PreparedAccurateOperation[_AccurateLoadedModelT, FeatureBackend] | None,
    timeout_seconds: float,
    expected_profile: str,
    logger: logging.Logger,
    run_with_process_timeout: Callable[..., InferenceResult],
    run_accurate_inference_once: Callable[..., InferenceResult],
    run_with_timeout: Callable[..., InferenceResult],
    run_inference_operation: Callable[..., InferenceResult],
    timeout_error_factory: Callable[[str], Exception],
) -> InferenceResult:
    """Runs one accurate inference attempt using the current retry state."""
    return _run_accurate_retryable_operation_impl(
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
        run_with_process_timeout=run_with_process_timeout,
        run_accurate_inference_once=run_accurate_inference_once,
        run_with_timeout=run_with_timeout,
        run_inference_operation=run_inference_operation,
        timeout_error_factory=timeout_error_factory,
        runtime_error_factory=RuntimeError,
    )


def execute_accurate_inference_with_retry(
    *,
    use_process_isolation: bool,
    retry_state: AccurateRetryOperationState[_AccuratePayloadT, FeatureBackend],
    prepared_operation: PreparedAccurateOperation[_AccurateLoadedModelT, FeatureBackend] | None,
    setup_started_at: float | None,
    settings: AppConfig,
    backend: FeatureBackend | None,
    expected_backend_id: str,
    expected_profile: str,
    allow_retries: bool,
    enforce_timeout: bool,
    cpu_backend_builder: Callable[[], FeatureBackend],
    logger: logging.Logger,
    run_accurate_retryable_operation: Callable[..., InferenceResult],
    retry_delay_seconds: Callable[..., float],
    process_payload_cpu_fallback: Callable[[_AccuratePayloadT], _AccuratePayloadT],
    timeout_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    inference_execution_error_type: type[Exception],
    transient_backend_error_type: type[Exception],
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
            timeout_seconds=settings.accurate_runtime.timeout_seconds,
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
                    dependency_error_factory=runtime_dependency_error_type,
                    transient_error_factory=transient_backend_error_type,
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
            process_payload_cpu_fallback=process_payload_cpu_fallback,
            cpu_backend_builder=cpu_backend_builder,
            run_retry_policy=_run_accurate_retry_policy_impl,
            retry_delay_seconds=retry_delay_seconds,
            run_with_retry_policy=run_with_retry_policy,
            passthrough_error_types=(
                runtime_dependency_error_type,
                ValueError,
                inference_execution_error_type,
            ),
            run_accurate_retryable_operation=run_accurate_retryable_operation,
            timeout_error_type=timeout_error_type,
            transient_error_type=transient_backend_error_type,
            transient_exhausted_error=lambda _err: inference_execution_error_type(
                "Accurate inference exhausted retry budget after backend failures."
            ),
            runtime_error_factory=lambda _err: inference_execution_error_type(
                "Accurate inference failed with a non-retryable runtime error."
            ),
            enforce_timeout=enforce_timeout,
        ),
    )


def prepare_in_process_accurate_operation(
    *,
    request: _AccurateRequestLike,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    loaded_model: _AccurateLoadedModelT | None,
    backend: FeatureBackend | None,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
    load_model_fn: Callable[..., _AccurateLoadedModelT],
    read_audio_file_fn: Callable[..., tuple[NDArray[np.float32], int]],
    build_backend_for_profile_fn: Callable[..., FeatureBackend],
    logger: logging.Logger,
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedAccurateOperation[_AccurateLoadedModelT, FeatureBackend]:
    """Prepares one in-process accurate operation using runtime contracts."""
    return _prepare_in_process_operation_orchestration(
        request=request,
        settings=settings,
        runtime_config=runtime_config,
        loaded_model=loaded_model,
        backend=backend,
        load_accurate_model=lambda active_settings: load_model_fn(
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
                unavailable_error_factory=model_unavailable_error_factory,
                logger=logger,
                resolve_runtime_policy=resolve_feature_runtime_policy,
            )
        ),
        read_audio_file=partial(
            read_audio_file_fn,
            audio_read_config=settings.audio_read,
        ),
        build_backend_for_profile=lambda active_settings: build_backend_for_profile_fn(
            expected_backend_id=expected_backend_id,
            expected_backend_model_id=expected_backend_model_id,
            settings=active_settings,
        ),
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def prepare_process_operation(
    payload: _AccuratePayloadLike,
    *,
    load_model_fn: Callable[..., _AccurateLoadedModelT],
    read_audio_file_fn: Callable[..., tuple[NDArray[np.float32], int]],
    build_backend_for_profile_fn: Callable[..., FeatureBackend],
    logger: logging.Logger,
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
    runtime_dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> PreparedAccurateOperation[_AccurateLoadedModelT, FeatureBackend]:
    """Performs untimed setup for one process-isolated accurate operation."""
    return _prepare_process_operation_orchestration(
        payload=payload,
        resolve_runtime_config=lambda settings, expected_profile: (
            _runtime_config_for_profile_impl(
                settings=settings,
                expected_profile=expected_profile,
                unsupported_profile_error=model_unavailable_error_factory,
            )
        ),
        load_accurate_model=lambda active_payload: load_model_fn(
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
                unavailable_error_factory=model_unavailable_error_factory,
                logger=logger,
                resolve_runtime_policy=resolve_feature_runtime_policy,
            )
        ),
        read_audio_file=partial(
            read_audio_file_fn,
            audio_read_config=payload.settings.audio_read,
        ),
        build_backend_for_payload=lambda active_payload: build_backend_for_profile_fn(
            expected_backend_id=active_payload.expected_backend_id,
            expected_backend_model_id=active_payload.expected_backend_model_id,
            settings=active_payload.settings,
        ),
        prepare_accurate_backend_runtime=lambda backend: (
            _prepare_accurate_backend_runtime_support_impl(
                backend,
                dependency_error_factory=runtime_dependency_error_factory,
                transient_error_factory=transient_error_factory,
            )
        ),
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def run_process_operation(
    prepared: PreparedAccurateOperation[_AccurateLoadedModelT, FeatureBackend],
    *,
    run_accurate_inference_once: Callable[..., InferenceResult],
) -> InferenceResult:
    """Runs one accurate compute phase inside the isolated worker process."""
    return _run_process_operation_orchestration(
        prepared,
        run_accurate_inference_once=lambda loaded_model, backend, audio, sample_rate, runtime_config: (
            run_accurate_inference_once(
                loaded_model=loaded_model,
                backend=backend,
                audio=audio,
                sample_rate=sample_rate,
                runtime_config=runtime_config,
            )
        ),
    )


def build_backend_for_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
    whisper_backend_factory: Callable[..., FeatureBackend],
    emotion2vec_backend_factory: Callable[..., FeatureBackend],
    unsupported_backend_error: Callable[[str], Exception],
) -> FeatureBackend:
    """Builds one feature backend aligned with accurate runtime expectations."""
    return _build_backend_for_profile_impl(
        expected_backend_id=expected_backend_id,
        expected_backend_model_id=expected_backend_model_id,
        settings=settings,
        resolve_accurate_model_id=resolve_accurate_model_id,
        resolve_accurate_research_model_id=resolve_accurate_research_model_id,
        resolve_runtime_policy=resolve_feature_runtime_policy,
        whisper_backend_factory=whisper_backend_factory,
        emotion2vec_backend_factory=emotion2vec_backend_factory,
        unsupported_backend_error=unsupported_backend_error,
    )


def retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    return _jittered_retry_delay_seconds_impl(
        base_delay=base_delay,
        attempt=attempt,
    )


def payload_with_cpu_settings(payload: _AccuratePayloadT) -> _AccuratePayloadT:
    """Returns one process payload updated to use CPU torch selectors."""
    return cast(
        _AccuratePayloadT,
        replace(
            cast(Any, payload),
            settings=_build_cpu_settings_snapshot_impl(payload.settings),
        ),
    )


__all__ = [
    "run_accurate_inference_from_public_boundary",
    "build_backend_for_profile",
    "execute_accurate_inference_with_retry",
    "payload_with_cpu_settings",
    "prepare_in_process_accurate_operation",
    "prepare_process_operation",
    "retry_delay_seconds",
    "run_accurate_inference_once",
    "run_accurate_retryable_operation",
    "run_process_operation",
]
