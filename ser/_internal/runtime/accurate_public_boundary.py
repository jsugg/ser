"""Internal support owners for accurate inference public-boundary wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.models.profile_runtime import (
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
)
from ser.repr import FeatureBackend
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


class _AccurateLoadedModel(ArtifactMetadataCarrier, _AccurateLoadedModelLike, Protocol):
    """Loaded-model contract required by accurate public-boundary helpers."""


_AccurateLoadedModelT = TypeVar("_AccurateLoadedModelT", bound=_AccurateLoadedModel)
_AccuratePayloadT = TypeVar("_AccuratePayloadT", bound=_AccuratePayloadLike)


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
