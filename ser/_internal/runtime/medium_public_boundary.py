"""Internal support owners for medium inference public-boundary wrappers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, MediumRuntimeConfig
from ser.repr import XLSRBackend
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
from ser.runtime.medium_execution import LoadedModelLike as _MediumExecutionLoadedModelLike
from ser.runtime.medium_execution_context import (
    MediumExecutionContext,
)
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
from ser.runtime.medium_process_operation import _PayloadLike as _MediumPayloadLike
from ser.runtime.medium_process_operation import (
    prepare_process_operation as _prepare_process_operation_impl,
)
from ser.runtime.medium_process_operation import (
    run_process_operation as _run_process_operation_impl,
)
from ser.runtime.medium_runtime_support import LoadedModelLike as _MediumRuntimeLoadedModelLike
from ser.runtime.medium_runtime_support import (
    build_cpu_settings_snapshot as _build_cpu_settings_snapshot_impl,
)
from ser.runtime.medium_runtime_support import (
    build_medium_backend_for_settings as _build_medium_backend_for_settings_impl,
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
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.schema import InferenceResult


class _MediumLoadedModel(_MediumExecutionLoadedModelLike, _MediumRuntimeLoadedModelLike, Protocol):
    """Loaded-model contract required by medium public-boundary helpers."""


_MediumLoadedModelT = TypeVar("_MediumLoadedModelT", bound=_MediumLoadedModel)
_MediumPayloadT = TypeVar("_MediumPayloadT", bound=_MediumPayloadLike)


def run_medium_inference_once(
    *,
    loaded_model: _MediumLoadedModel,
    backend: XLSRBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: MediumRuntimeConfig,
    logger: logging.Logger,
    is_dependency_error: Callable[[RuntimeError], bool],
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> InferenceResult:
    """Runs one medium inference attempt without retry control."""
    encoded = _encode_medium_sequence_impl(
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        is_dependency_error=is_dependency_error,
        dependency_error_factory=dependency_error_factory,
        transient_error_factory=transient_error_factory,
    )
    return medium_execution_helpers.run_medium_inference_once(
        loaded_model=loaded_model,
        encoded=encoded,
        runtime_config=runtime_config,
        predict_labels=lambda model, features: _predict_labels_impl(model=model, features=features),
        confidence_and_probabilities=lambda model, features, expected_rows: (
            _confidence_and_probabilities_impl(
                model=model,
                features=features,
                expected_rows=expected_rows,
                logger=logger,
            )
        ),
    )


def prepare_in_process_operation(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: _MediumLoadedModelT | None,
    backend: XLSRBackend | None,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    load_model_fn: Callable[..., _MediumLoadedModelT],
    read_audio_file_fn: Callable[..., tuple[NDArray[np.float32], int]],
    backend_factory: Callable[..., XLSRBackend],
    logger: logging.Logger,
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> medium_worker_operation_helpers.PreparedMediumOperation[_MediumLoadedModelT, XLSRBackend]:
    """Performs untimed setup for one in-process medium operation."""
    return medium_worker_operation_helpers.prepare_in_process_operation(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        expected_backend_model_id=expected_backend_model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        load_medium_model=lambda active_settings, backend_model_id: load_model_fn(
            settings=active_settings,
            expected_backend_id="hf_xlsr",
            expected_profile="medium",
            expected_backend_model_id=backend_model_id,
        ),
        ensure_medium_compatible_model=lambda active_loaded_model, backend_model_id: (
            _ensure_medium_loaded_model_compatibility_impl(
                active_loaded_model,
                expected_backend_model_id=backend_model_id,
                unavailable_error_factory=model_unavailable_error_factory,
            )
        ),
        warn_on_runtime_selector_mismatch=lambda active_loaded_model, device, dtype: (
            _warn_on_medium_runtime_selector_mismatch_impl(
                loaded_model=active_loaded_model,
                profile="medium",
                runtime_device=device,
                runtime_dtype=dtype,
                logger=logger,
            )
        ),
        read_audio_file=partial(
            read_audio_file_fn,
            audio_read_config=settings.audio_read,
        ),
        build_medium_backend=lambda active_settings, backend_model_id, device, dtype: (
            _build_medium_backend_for_settings_impl(
                settings=active_settings,
                expected_backend_model_id=backend_model_id,
                runtime_device=device,
                runtime_dtype=dtype,
                backend_factory=backend_factory,
            )
        ),
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def prepare_process_operation(
    payload: _MediumPayloadLike,
    *,
    load_model_fn: Callable[..., _MediumLoadedModelT],
    read_audio_file_fn: Callable[..., tuple[NDArray[np.float32], int]],
    backend_factory: Callable[..., XLSRBackend],
    resolve_runtime_policy: Callable[[AppConfig], FeatureRuntimePolicy],
    logger: logging.Logger,
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
    prepare_medium_backend_runtime: Callable[[XLSRBackend], None],
) -> medium_worker_operation_helpers.PreparedMediumOperation[_MediumLoadedModelT, XLSRBackend]:
    """Performs untimed setup for one process-isolated medium operation."""
    return _prepare_process_operation_impl(
        payload=payload,
        load_medium_model=lambda settings, backend_model_id: load_model_fn(
            settings=settings,
            expected_backend_id="hf_xlsr",
            expected_profile="medium",
            expected_backend_model_id=backend_model_id,
        ),
        ensure_medium_compatible_model=lambda active_loaded_model, backend_model_id: (
            _ensure_medium_loaded_model_compatibility_impl(
                active_loaded_model,
                expected_backend_model_id=backend_model_id,
                unavailable_error_factory=model_unavailable_error_factory,
            )
        ),
        resolve_runtime_policy=resolve_runtime_policy,
        warn_on_runtime_selector_mismatch=lambda active_loaded_model, device, dtype: (
            _warn_on_medium_runtime_selector_mismatch_impl(
                loaded_model=active_loaded_model,
                profile="medium",
                runtime_device=device,
                runtime_dtype=dtype,
                logger=logger,
            )
        ),
        read_audio_file=partial(
            read_audio_file_fn,
            audio_read_config=payload.settings.audio_read,
        ),
        build_medium_backend=lambda settings, backend_model_id, device, dtype: (
            _build_medium_backend_for_settings_impl(
                settings=settings,
                expected_backend_model_id=backend_model_id,
                runtime_device=device,
                runtime_dtype=dtype,
                backend_factory=backend_factory,
            )
        ),
        prepare_medium_backend_runtime=prepare_medium_backend_runtime,
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def run_process_operation(
    prepared: medium_worker_operation_helpers.PreparedMediumOperation[
        _MediumLoadedModelT, XLSRBackend
    ],
    *,
    run_medium_inference_once: Callable[..., InferenceResult],
) -> InferenceResult:
    """Runs one medium compute phase inside the isolated worker process."""
    return _run_process_operation_impl(
        prepared,
        run_medium_inference_once=lambda loaded_model, backend, audio, sample_rate, runtime_config: (
            run_medium_inference_once(
                loaded_model=loaded_model,
                backend=backend,
                audio=audio,
                sample_rate=sample_rate,
                runtime_config=runtime_config,
            )
        ),
    )


def prepare_execution_context(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: _MediumLoadedModelT | None,
    backend: XLSRBackend | None,
    enforce_timeout: bool,
    resolve_medium_model_id: Callable[[AppConfig], str],
    resolve_runtime_policy: Callable[[AppConfig], FeatureRuntimePolicy],
    prepare_retry_state: Callable[
        ...,
        tuple[
            medium_worker_operation_helpers.MediumRetryOperationState[
                _MediumPayloadT,
                _MediumLoadedModelT,
                XLSRBackend,
            ],
            float | None,
        ],
    ],
    prepare_in_process_operation: Callable[
        ...,
        medium_worker_operation_helpers.PreparedMediumOperation[_MediumLoadedModelT, XLSRBackend],
    ],
    build_process_payload: Callable[[str, str, str], _MediumPayloadT],
    logger: logging.Logger,
) -> MediumExecutionContext[_MediumPayloadT, _MediumLoadedModelT, XLSRBackend]:
    """Resolves pre-lock runtime context for medium inference execution."""
    return _prepare_execution_context_impl(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        resolve_medium_model_id=resolve_medium_model_id,
        resolve_runtime_policy=resolve_runtime_policy,
        log_selector_adjustment=lambda device, dtype, reason: logger.info(
            "Medium feature runtime policy adjusted selectors (device=%s, dtype=%s, reason=%s).",
            device,
            dtype,
            reason,
        ),
        prepare_retry_state=prepare_retry_state,
        build_process_payload=build_process_payload,
        prepare_in_process_operation=prepare_in_process_operation,
        logger=logger,
        profile="medium",
        setup_phase_name=PHASE_EMOTION_SETUP,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
    )


def execute_medium_inference_with_retry(
    *,
    execution_context: MediumExecutionContext[_MediumPayloadT, _MediumLoadedModelT, XLSRBackend],
    settings: AppConfig,
    injected_backend: XLSRBackend | None,
    enforce_timeout: bool,
    allow_retries: bool,
    expected_backend_model_id: str,
    logger: logging.Logger,
    run_with_process_timeout: Callable[[_MediumPayloadT, float], InferenceResult],
    run_process_operation: Callable[
        [medium_worker_operation_helpers.PreparedMediumOperation[_MediumLoadedModelT, XLSRBackend]],
        InferenceResult,
    ],
    run_with_timeout: Callable[[Callable[[], InferenceResult], float], InferenceResult],
    prepare_medium_backend_runtime: Callable[[XLSRBackend], None],
    cpu_backend_builder: Callable[[], XLSRBackend],
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    execution_error_type: type[Exception],
    run_retry_policy_impl: Callable[..., InferenceResult],
    retry_delay_seconds: Callable[..., float],
    should_retry_on_cpu_after_transient_failure: Callable[[Exception], bool],
    summarize_transient_failure: Callable[[Exception], str],
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
            prepare_medium_backend_runtime=prepare_medium_backend_runtime,
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_inference_operation=medium_worker_operation_helpers.run_inference_operation,
            run_with_process_timeout=run_with_process_timeout,
            run_process_operation=run_process_operation,
            run_with_timeout=run_with_timeout,
            build_transient_failure_handler=medium_worker_operation_helpers.build_transient_failure_handler,
            should_retry_on_cpu_after_transient_failure=should_retry_on_cpu_after_transient_failure,
            summarize_transient_failure=summarize_transient_failure,
            process_payload_cpu_fallback=lambda payload: cast(
                _MediumPayloadT,
                replace(
                    cast(Any, payload),
                    settings=_build_cpu_settings_snapshot_impl(payload.settings),
                ),
            ),
            in_process_cpu_backend_builder=cpu_backend_builder,
            replace_prepared_backend=lambda prepared, active_backend: replace(
                prepared,
                backend=active_backend,
            ),
            run_retry_policy=run_retry_policy_impl,
            retry_delay_seconds=retry_delay_seconds,
            timeout_error_type=timeout_error_type,
            transient_error_type=transient_error_type,
            transient_exhausted_error=lambda _err: execution_error_type(
                "Medium inference exhausted retry budget after backend failures."
            ),
            run_with_retry_policy=run_with_retry_policy,
            passthrough_error_types=(
                runtime_dependency_error_type,
                ValueError,
                execution_error_type,
            ),
            runtime_error_factory=lambda _err: execution_error_type(
                "Medium inference failed with a non-retryable runtime error."
            ),
        ),
    )


def resolve_medium_feature_runtime_policy(
    *,
    settings: AppConfig,
) -> FeatureRuntimePolicy:
    """Resolves backend-aware feature runtime selectors for medium profile."""
    return _resolve_medium_feature_runtime_policy_impl(
        settings=settings,
    )


def retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    return medium_retry_policy_helpers.retry_delay_seconds(
        base_delay=base_delay,
        attempt=attempt,
    )


def should_retry_on_cpu_after_transient_failure(err: Exception) -> bool:
    """Returns whether one transient failure should trigger CPU fallback retry."""
    return medium_retry_policy_helpers.should_retry_on_cpu_after_transient_failure(err)


def summarize_transient_failure(err: Exception) -> str:
    """Builds one compact summary line for medium transient fallback logs."""
    return medium_retry_policy_helpers.summarize_transient_failure(err)


def is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    return medium_retry_policy_helpers.is_dependency_error(err)


def prepare_medium_backend_runtime(
    *,
    backend: XLSRBackend,
    is_dependency_error: Callable[[RuntimeError], bool],
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    _prepare_medium_backend_runtime_impl(
        backend=backend,
        is_dependency_error=is_dependency_error,
        dependency_error_factory=dependency_error_factory,
        transient_error_factory=transient_error_factory,
    )


__all__ = [
    "execute_medium_inference_with_retry",
    "is_dependency_error",
    "prepare_execution_context",
    "prepare_in_process_operation",
    "prepare_medium_backend_runtime",
    "prepare_process_operation",
    "resolve_medium_feature_runtime_policy",
    "retry_delay_seconds",
    "run_medium_inference_once",
    "run_process_operation",
    "should_retry_on_cpu_after_transient_failure",
    "summarize_transient_failure",
]
