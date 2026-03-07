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
from ser._internal.runtime.worker_lifecycle import (
    raise_worker_error as _raise_worker_error_impl,
)
from ser._internal.runtime.worker_lifecycle import (
    recv_worker_message as _recv_worker_message_impl,
)
from ser._internal.runtime.worker_lifecycle import (
    run_process_setup_compute_handshake as _run_process_setup_compute_handshake_impl,
)
from ser._internal.runtime.worker_lifecycle import (
    run_with_timeout as _run_with_timeout_impl,
)
from ser._internal.runtime.worker_lifecycle import (
    terminate_worker_process as _terminate_worker_process_impl,
)
from ser.config import AppConfig, MediumRuntimeConfig
from ser.models.emotion_model import LoadedModel, load_model
from ser.models.profile_runtime import resolve_medium_model_id
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import EncodedSequence, PoolingWindow, XLSRBackend
from ser.repr.runtime_policy import FeatureRuntimePolicy
from ser.runtime import medium_process_timeout as medium_process_timeout_helpers
from ser.runtime import medium_retry_policy as medium_retry_policy_helpers
from ser.runtime import medium_worker_operation as medium_worker_operation_helpers
from ser.runtime.contracts import InferenceRequest
from ser.runtime.medium_backend_runtime import (
    build_medium_backend as _build_medium_backend_impl,
)
from ser.runtime.medium_backend_runtime import (
    prepare_medium_backend_runtime as _prepare_medium_backend_runtime_impl,
)
from ser.runtime.medium_backend_runtime import (
    resolve_medium_feature_runtime_policy as _resolve_medium_feature_runtime_policy_impl,
)
from ser.runtime.medium_backend_runtime import (
    settings_with_torch_runtime as _settings_with_torch_runtime_impl,
)
from ser.runtime.medium_backend_runtime import (
    warn_on_runtime_selector_mismatch as _warn_on_runtime_selector_mismatch_impl,
)
from ser.runtime.medium_prediction import (
    confidence_and_probabilities as _confidence_and_probabilities_impl,
)
from ser.runtime.medium_prediction import predict_labels as _predict_labels_impl
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.postprocessing import (
    build_segment_postprocessing_config,
    postprocess_frame_predictions,
)
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
)
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
    runtime_config = settings.medium_runtime
    expected_backend_model_id = resolve_medium_model_id(settings)
    runtime_policy = _resolve_medium_feature_runtime_policy(settings=settings)
    policy_device = runtime_policy.device
    policy_dtype = runtime_policy.dtype
    if (
        policy_device != settings.torch_runtime.device
        or policy_dtype != settings.torch_runtime.dtype
    ):
        logger.info(
            "Medium feature runtime policy adjusted selectors "
            "(device=%s, dtype=%s, reason=%s).",
            policy_device,
            policy_dtype,
            runtime_policy.reason,
        )
    use_process_isolation = (
        enforce_timeout
        and loaded_model is None
        and backend is None
        and settings.runtime_flags.profile_pipeline
        and runtime_config.process_isolation
    )

    retry_state: _MediumRetryOperationState
    setup_started_at: float | None
    retry_state, setup_started_at = medium_worker_operation_helpers.prepare_retry_state(
        use_process_isolation=use_process_isolation,
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        expected_backend_model_id=expected_backend_model_id,
        policy_device=policy_device,
        policy_dtype=policy_dtype,
        logger=logger,
        profile="medium",
        setup_phase_name=PHASE_EMOTION_SETUP,
        log_phase_started=log_phase_started,
        log_phase_failed=log_phase_failed,
        build_process_payload=lambda: MediumProcessPayload(
            request=request,
            settings=_settings_with_torch_runtime(
                _build_process_settings_snapshot(settings),
                device=policy_device,
                dtype=policy_dtype,
            ),
            expected_backend_model_id=expected_backend_model_id,
        ),
        prepare_in_process_operation=_prepare_in_process_operation,
    )

    with _SINGLE_FLIGHT_REGISTRY.lock(
        profile="medium",
        backend_model_id=expected_backend_model_id,
    ):
        medium_worker_operation_helpers.finalize_in_process_setup(
            use_process_isolation=use_process_isolation,
            state=retry_state,
            setup_started_at=setup_started_at,
            logger=logger,
            profile="medium",
            setup_phase_name=PHASE_EMOTION_SETUP,
            prepare_medium_backend_runtime=lambda active_backend: (
                _prepare_medium_backend_runtime(backend=active_backend)
            ),
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            runtime_error_factory=RuntimeError,
        )

        def operation() -> InferenceResult:
            return medium_worker_operation_helpers.run_inference_operation(
                enforce_timeout=enforce_timeout,
                use_process_isolation=use_process_isolation,
                process_payload=retry_state.process_payload,
                prepared_operation=retry_state.prepared_operation,
                timeout_seconds=runtime_config.timeout_seconds,
                logger=logger,
                profile="medium",
                inference_phase_name=PHASE_EMOTION_INFERENCE,
                log_phase_started=log_phase_started,
                log_phase_completed=log_phase_completed,
                log_phase_failed=log_phase_failed,
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
                runtime_error_factory=RuntimeError,
            )

        on_transient_failure = (
            medium_worker_operation_helpers.build_transient_failure_handler(
                state=retry_state,
                use_process_isolation=use_process_isolation,
                injected_backend=backend,
                policy_device=policy_device,
                logger=logger,
                should_retry_on_cpu_after_transient_failure=(
                    _should_retry_on_cpu_after_transient_failure
                ),
                summarize_transient_failure=_summarize_transient_failure,
                process_payload_cpu_fallback=lambda payload: replace(
                    payload,
                    settings=_settings_with_torch_runtime(
                        payload.settings,
                        device="cpu",
                        dtype="float32",
                    ),
                ),
                in_process_cpu_backend_builder=lambda: _build_medium_backend(
                    settings=_settings_with_torch_runtime(
                        settings,
                        device="cpu",
                        dtype="float32",
                    ),
                    expected_backend_model_id=expected_backend_model_id,
                    runtime_device="cpu",
                    runtime_dtype="float32",
                ),
                prepare_medium_backend_runtime=lambda active_backend: (
                    _prepare_medium_backend_runtime(backend=active_backend)
                ),
                replace_prepared_backend=lambda prepared, active_backend: replace(
                    prepared,
                    backend=active_backend,
                ),
                runtime_error_factory=RuntimeError,
            )
        )

        try:
            return run_with_retry_policy(
                operation=operation,
                runtime_config=runtime_config,
                allow_retries=allow_retries,
                profile_label="Medium",
                timeout_error_type=MediumInferenceTimeoutError,
                transient_error_type=MediumTransientBackendError,
                transient_exhausted_error=lambda err: MediumInferenceExecutionError(
                    "Medium inference exhausted retry budget after backend failures."
                ),
                retry_delay_seconds=_retry_delay_seconds,
                logger=logger,
                on_transient_failure=on_transient_failure,
            )
        except MediumRuntimeDependencyError:
            raise
        except ValueError:
            raise
        except MediumInferenceExecutionError:
            raise
        except RuntimeError as err:
            raise MediumInferenceExecutionError(
                "Medium inference failed with a non-retryable runtime error."
            ) from err


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
    encoded = _encode_medium_sequence(
        backend=backend, audio=audio, sample_rate=sample_rate
    )
    windows = _pooling_windows_from_encoded_frames(
        encoded,
        runtime_config=runtime_config,
    )
    feature_matrix = mean_std_pool(encoded, windows)

    expected_size = loaded_model.expected_feature_size
    if expected_size is not None and int(feature_matrix.shape[1]) != expected_size:
        raise ValueError(
            "Feature vector size mismatch for medium-profile model. "
            f"Expected {expected_size}, got {int(feature_matrix.shape[1])}."
        )

    predictions = _predict_labels(loaded_model.model, feature_matrix)
    confidences, probabilities = _confidence_and_probabilities(
        loaded_model.model,
        feature_matrix,
        expected_rows=len(windows),
    )

    frame_predictions = [
        FramePrediction(
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
            emotion=predictions[index],
            confidence=confidences[index],
            probabilities=probabilities[index],
        )
        for index, window in enumerate(windows)
    ]
    return InferenceResult(
        schema_version=OUTPUT_SCHEMA_VERSION,
        segments=postprocess_frame_predictions(
            frame_predictions,
            config=build_segment_postprocessing_config(runtime_config),
        ),
        frames=frame_predictions,
    )


def _run_with_process_timeout(
    payload: MediumProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    return medium_process_timeout_helpers.run_with_process_timeout(
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
    return cast(
        WorkerMessage,
        _recv_worker_message_impl(
            connection=connection,
            stage=stage,
            worker_label="Medium inference",
            error_factory=MediumInferenceExecutionError,
        ),
    )


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_impl(
        message=message,
        worker_label="Medium inference",
        error_factory=MediumInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    return _parse_worker_completion_message_impl(
        worker_message=worker_message,
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
    try:
        prepared_operation = _prepare_process_operation(payload)
        connection.send(("phase", "setup_complete"))
        result = _run_process_operation(prepared_operation)
        connection.send(("ok", result))
    except BaseException as err:
        connection.send(("err", type(err).__name__, str(err)))
    finally:
        connection.close()


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
            _ensure_medium_compatible_model(
                loaded_model,
                expected_backend_model_id=expected_backend_model_id,
            )
        ),
        warn_on_runtime_selector_mismatch=lambda loaded_model, runtime_device, runtime_dtype: (
            _warn_on_runtime_selector_mismatch(
                loaded_model=loaded_model,
                profile="medium",
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=settings.audio_read,
        ),
        build_medium_backend=lambda settings, expected_backend_model_id, runtime_device, runtime_dtype: (
            _build_medium_backend(
                settings=settings,
                expected_backend_model_id=expected_backend_model_id,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
        ),
        model_unavailable_error_factory=MediumModelUnavailableError,
        model_load_error_factory=MediumModelLoadError,
    )


def _prepare_process_operation(
    payload: MediumProcessPayload,
) -> _PreparedMediumOperation:
    """Performs untimed setup for one process-isolated medium operation."""
    return medium_worker_operation_helpers.prepare_process_operation(
        payload=payload,
        load_medium_model=lambda settings, expected_backend_model_id: load_model(
            settings=settings,
            expected_backend_id="hf_xlsr",
            expected_profile="medium",
            expected_backend_model_id=expected_backend_model_id,
        ),
        ensure_medium_compatible_model=lambda loaded_model, expected_backend_model_id: (
            _ensure_medium_compatible_model(
                loaded_model,
                expected_backend_model_id=expected_backend_model_id,
            )
        ),
        resolve_runtime_policy=lambda settings: _resolve_medium_feature_runtime_policy(
            settings=settings
        ),
        warn_on_runtime_selector_mismatch=lambda loaded_model, runtime_device, runtime_dtype: (
            _warn_on_runtime_selector_mismatch(
                loaded_model=loaded_model,
                profile="medium",
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
            )
        ),
        read_audio_file=partial(
            read_audio_file,
            audio_read_config=payload.settings.audio_read,
        ),
        build_medium_backend=lambda settings, expected_backend_model_id, runtime_device, runtime_dtype: (
            _build_medium_backend(
                settings=settings,
                expected_backend_model_id=expected_backend_model_id,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
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
    return medium_worker_operation_helpers.run_process_operation(
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


def _build_process_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Builds a process-safe settings snapshot for spawn-based workers."""
    return replace(settings, emotions=dict(settings.emotions))


def _settings_with_torch_runtime(
    settings: AppConfig,
    *,
    device: str,
    dtype: str,
) -> AppConfig:
    """Returns process-safe settings with one normalized torch runtime selector pair."""
    return _settings_with_torch_runtime_impl(
        settings,
        device=device,
        dtype=dtype,
    )


def _resolve_medium_feature_runtime_policy(
    *,
    settings: AppConfig,
) -> FeatureRuntimePolicy:
    """Resolves backend-aware feature runtime selectors for medium profile."""
    return _resolve_medium_feature_runtime_policy_impl(
        settings=settings,
    )


def _build_medium_backend(
    *,
    settings: AppConfig,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
) -> XLSRBackend:
    """Builds one XLS-R backend with explicit runtime selectors."""
    return _build_medium_backend_impl(
        settings=settings,
        expected_backend_model_id=expected_backend_model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        backend_factory=XLSRBackend,
    )


def _terminate_worker_process(process: BaseProcess) -> None:
    """Terminates a timed-out worker process with kill fallback."""
    _terminate_worker_process_impl(
        process=process,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        kill_grace_seconds=_KILL_GRACE_SECONDS,
    )


def _raise_worker_error(error_type: str, message: str) -> None:
    """Rehydrates child-process errors into runtime-domain exceptions."""
    _raise_worker_error_impl(
        error_type=error_type,
        message=message,
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


def _encode_medium_sequence(
    *,
    backend: XLSRBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
) -> EncodedSequence:
    """Encodes medium audio sequence and maps dependency/transient failures."""
    try:
        return backend.encode_sequence(audio, sample_rate)
    except RuntimeError as err:
        if _is_dependency_error(err):
            raise MediumRuntimeDependencyError(str(err)) from err
        raise MediumTransientBackendError(str(err)) from err


def _prepare_medium_backend_runtime(*, backend: XLSRBackend) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    _prepare_medium_backend_runtime_impl(
        backend=backend,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )


def _ensure_medium_compatible_model(
    loaded_model: LoadedModel,
    *,
    expected_backend_model_id: str,
) -> None:
    """Validates that loaded artifact metadata is compatible with medium runtime."""
    medium_worker_operation_helpers.ensure_medium_compatible_model(
        loaded_model,
        expected_backend_model_id=expected_backend_model_id,
        unavailable_error_factory=MediumModelUnavailableError,
    )


def _warn_on_runtime_selector_mismatch(
    *,
    loaded_model: LoadedModel,
    profile: str,
    runtime_device: str,
    runtime_dtype: str,
) -> None:
    """Warns when artifact runtime selectors differ from current runtime settings."""
    _warn_on_runtime_selector_mismatch_impl(
        loaded_model=loaded_model,
        profile=profile,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        logger=logger,
    )


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    runtime_config: MediumRuntimeConfig,
) -> list[PoolingWindow]:
    """Creates medium temporal pooling windows from configured window policy."""
    return temporal_pooling_windows(
        encoded,
        window_size_seconds=runtime_config.pool_window_size_seconds,
        window_stride_seconds=runtime_config.pool_window_stride_seconds,
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
