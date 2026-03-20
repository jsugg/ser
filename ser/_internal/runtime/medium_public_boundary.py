"""Internal support owners for medium inference public-boundary wrappers."""

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
from ser.config import AppConfig, MediumRuntimeConfig
from ser.models.emotion_model import LoadedModel as EmotionLoadedModel
from ser.models.emotion_model import load_model
from ser.models.profile_runtime import resolve_medium_model_id
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
from ser.runtime.medium_retry_operation import (
    run_medium_inference_with_retry_policy as _run_medium_retry_policy_impl,
)
from ser.runtime.medium_runtime_support import LoadedModelLike as _MediumRuntimeLoadedModelLike
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
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.schema import InferenceResult
from ser.utils.audio_utils import read_audio_file


class _MediumLoadedModel(_MediumExecutionLoadedModelLike, _MediumRuntimeLoadedModelLike, Protocol):
    """Loaded-model contract required by medium public-boundary helpers."""


_MediumLoadedModelT = TypeVar("_MediumLoadedModelT", bound=_MediumLoadedModel)
_MediumPayloadT = TypeVar("_MediumPayloadT", bound=_MediumPayloadLike)

type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_REGISTRY = SingleFlightRegistry()
_WORKER_LOGGER = logging.getLogger("ser.runtime.medium_inference")


@dataclass(frozen=True)
class MediumProcessPayload:
    """Serializable payload for one process-isolated medium inference attempt."""

    request: InferenceRequest
    settings: AppConfig
    expected_backend_model_id: str


class MediumModelUnavailableError(FileNotFoundError):
    """Spawn-safe medium worker error marker for unavailable model artifacts."""


class MediumRuntimeDependencyError(RuntimeError):
    """Spawn-safe medium worker error marker for missing runtime dependencies."""


class MediumModelLoadError(RuntimeError):
    """Spawn-safe medium worker error marker for model load failures."""


class MediumTransientBackendError(RuntimeError):
    """Spawn-safe medium worker error marker for transient backend failures."""


@dataclass(frozen=True)
class _MediumBoundaryDependencies:
    """Precomputed collaborators and execution plan for medium boundary orchestration."""

    execution_context: MediumExecutionContext[MediumProcessPayload, _MediumLoadedModel, XLSRBackend]
    expected_backend_model_id: str
    run_with_timeout: Callable[..., InferenceResult]
    run_with_process_timeout: Callable[..., InferenceResult]
    run_inference_once: Callable[..., InferenceResult]


def _prepare_medium_process_operation(
    payload: MediumProcessPayload,
) -> medium_worker_operation_helpers.PreparedMediumOperation[EmotionLoadedModel, XLSRBackend]:
    """Builds one medium worker operation using only module-level collaborators."""
    return prepare_process_operation(
        payload,
        load_model_fn=load_model,
        read_audio_file_fn=read_audio_file,
        backend_factory=XLSRBackend,
        resolve_runtime_policy=lambda settings: resolve_medium_feature_runtime_policy(
            settings=settings
        ),
        logger=_WORKER_LOGGER,
        model_unavailable_error_factory=MediumModelUnavailableError,
        model_load_error_factory=MediumModelLoadError,
        prepare_medium_backend_runtime=lambda active_backend: prepare_medium_backend_runtime(
            backend=active_backend,
            is_dependency_error=is_dependency_error,
            dependency_error_factory=MediumRuntimeDependencyError,
            transient_error_factory=MediumTransientBackendError,
        ),
    )


def _run_medium_process_inference_once(
    *,
    loaded_model: _MediumLoadedModel,
    backend: XLSRBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: MediumRuntimeConfig,
) -> InferenceResult:
    """Runs one medium worker-process inference attempt."""
    return run_medium_inference_once(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
        logger=_WORKER_LOGGER,
        is_dependency_error=is_dependency_error,
        dependency_error_factory=MediumRuntimeDependencyError,
        transient_error_factory=MediumTransientBackendError,
    )


def _run_medium_process_operation(
    prepared: medium_worker_operation_helpers.PreparedMediumOperation[
        EmotionLoadedModel, XLSRBackend
    ],
) -> InferenceResult:
    """Runs one medium compute phase inside the spawned worker process."""
    return run_process_operation(
        prepared,
        run_medium_inference_once=_run_medium_process_inference_once,
    )


def _medium_worker_entry(payload: MediumProcessPayload, connection: Connection) -> None:
    """Executes one spawned medium worker using module-level collaborators only."""
    _run_worker_entry_binding(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_medium_process_operation,
        run_process_operation=_run_medium_process_operation,
    )


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


def _build_medium_boundary_dependencies(
    *,
    request: InferenceRequest,
    settings: AppConfig,
    loaded_model: _MediumLoadedModel | None,
    backend: XLSRBackend | None,
    enforce_timeout: bool,
    logger: logging.Logger,
    model_unavailable_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    model_load_error_type: type[Exception],
    timeout_error_type: type[Exception],
    execution_error_type: type[Exception],
    transient_error_type: type[Exception],
) -> _MediumBoundaryDependencies:
    """Builds medium boundary collaborators and execution plan once per invocation."""
    worker_error_factories: dict[str, Callable[[str], Exception]] = {
        "ValueError": ValueError,
        runtime_dependency_error_type.__name__: runtime_dependency_error_type,
        transient_error_type.__name__: transient_error_type,
        model_unavailable_error_type.__name__: model_unavailable_error_type,
        model_load_error_type.__name__: model_load_error_type,
        timeout_error_type.__name__: timeout_error_type,
        "RuntimeError": RuntimeError,
    }

    def _run_with_timeout(
        operation: Callable[[], InferenceResult],
        timeout_seconds: float,
    ) -> InferenceResult:
        return _run_with_timeout_impl(
            operation=operation,
            timeout_seconds=timeout_seconds,
            timeout_error_factory=timeout_error_type,
            timeout_label="Medium inference",
        )

    def _run_with_process_timeout(
        payload: MediumProcessPayload,
        timeout_seconds: float,
    ) -> InferenceResult:
        return _run_with_process_timeout_impl(
            payload=payload,
            resolve_profile=lambda _payload: "medium",
            timeout_seconds=timeout_seconds,
            get_context=mp.get_context,
            logger=logger,
            setup_phase_name=PHASE_EMOTION_SETUP,
            inference_phase_name=PHASE_EMOTION_INFERENCE,
            log_phase_started=log_phase_started,
            log_phase_completed=log_phase_completed,
            log_phase_failed=log_phase_failed,
            run_process_setup_compute_handshake=_run_process_setup_compute_handshake_impl,
            worker_target=_medium_worker_entry,
            recv_worker_message=_recv_worker_message,
            is_setup_complete_message=_is_setup_complete_message,
            terminate_worker_process=_terminate_worker_process,
            timeout_error_factory=timeout_error_type,
            execution_error_factory=execution_error_type,
            worker_label="Medium inference",
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
            worker_label="Medium inference",
            error_factory=execution_error_type,
        )

    def _is_setup_complete_message(message: WorkerMessage) -> bool:
        return _is_setup_complete_message_binding(
            message=message,
            impl=_is_setup_complete_message_impl,
            worker_label="Medium inference",
            error_factory=execution_error_type,
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
            unknown_error_factory=execution_error_type,
            worker_label="Medium inference",
        )

    def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
        return _parse_worker_completion_message_binding(
            worker_message=worker_message,
            impl=_parse_worker_completion_message_impl,
            worker_label="Medium inference",
            error_factory=execution_error_type,
            raise_worker_error=_raise_worker_error,
            result_type=InferenceResult,
        )

    def _prepare_in_process_operation(
        *,
        request: InferenceRequest,
        settings: AppConfig,
        loaded_model: _MediumLoadedModel | None,
        backend: XLSRBackend | None,
        expected_backend_model_id: str,
        runtime_device: str,
        runtime_dtype: str,
    ) -> medium_worker_operation_helpers.PreparedMediumOperation[_MediumLoadedModel, XLSRBackend]:
        return prepare_in_process_operation(
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
            model_unavailable_error_factory=model_unavailable_error_type,
            model_load_error_factory=model_load_error_type,
        )

    def _prepare_execution_context(
        *,
        request: InferenceRequest,
        settings: AppConfig,
        loaded_model: _MediumLoadedModel | None,
        backend: XLSRBackend | None,
        enforce_timeout: bool,
    ) -> MediumExecutionContext[MediumProcessPayload, _MediumLoadedModel, XLSRBackend]:
        return prepare_execution_context(
            request=request,
            settings=settings,
            loaded_model=loaded_model,
            backend=backend,
            enforce_timeout=enforce_timeout,
            resolve_medium_model_id=resolve_medium_model_id,
            resolve_runtime_policy=lambda active_settings: resolve_medium_feature_runtime_policy(
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
        )

    def _run_medium_inference_once(
        *,
        loaded_model: _MediumLoadedModel,
        backend: XLSRBackend,
        audio: NDArray[np.float32],
        sample_rate: int,
        runtime_config: MediumRuntimeConfig,
    ) -> InferenceResult:
        return run_medium_inference_once(
            loaded_model=loaded_model,
            backend=backend,
            audio=audio,
            sample_rate=sample_rate,
            runtime_config=runtime_config,
            logger=logger,
            is_dependency_error=is_dependency_error,
            dependency_error_factory=runtime_dependency_error_type,
            transient_error_factory=transient_error_type,
        )

    execution_context = _prepare_execution_context(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
    )
    return _MediumBoundaryDependencies(
        execution_context=execution_context,
        expected_backend_model_id=execution_context.expected_backend_model_id,
        run_with_timeout=_run_with_timeout,
        run_with_process_timeout=_run_with_process_timeout,
        run_inference_once=_run_medium_inference_once,
    )


def run_medium_inference_from_public_boundary(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: _MediumLoadedModel | None = None,
    backend: XLSRBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
    logger: logging.Logger,
    model_unavailable_error_type: type[Exception],
    runtime_dependency_error_type: type[Exception],
    model_load_error_type: type[Exception],
    timeout_error_type: type[Exception],
    execution_error_type: type[Exception],
    transient_error_type: type[Exception],
) -> InferenceResult:
    """Runs medium inference through the internal public-boundary owner."""
    dependencies = _build_medium_boundary_dependencies(
        request=request,
        settings=settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        logger=logger,
        model_unavailable_error_type=model_unavailable_error_type,
        runtime_dependency_error_type=runtime_dependency_error_type,
        model_load_error_type=model_load_error_type,
        timeout_error_type=timeout_error_type,
        execution_error_type=execution_error_type,
        transient_error_type=transient_error_type,
    )

    with _SINGLE_FLIGHT_REGISTRY.lock(
        profile="medium",
        backend_model_id=dependencies.expected_backend_model_id,
    ):
        return execute_medium_inference_with_retry(
            execution_context=dependencies.execution_context,
            settings=settings,
            injected_backend=backend,
            enforce_timeout=enforce_timeout,
            allow_retries=allow_retries,
            expected_backend_model_id=dependencies.expected_backend_model_id,
            logger=logger,
            run_with_process_timeout=dependencies.run_with_process_timeout,
            run_process_operation=lambda prepared: run_process_operation(
                prepared,
                run_medium_inference_once=dependencies.run_inference_once,
            ),
            run_with_timeout=dependencies.run_with_timeout,
            prepare_medium_backend_runtime=lambda active_backend: prepare_medium_backend_runtime(
                backend=active_backend,
                is_dependency_error=is_dependency_error,
                dependency_error_factory=runtime_dependency_error_type,
                transient_error_factory=transient_error_type,
            ),
            cpu_backend_builder=lambda: _build_medium_backend_for_settings_impl(
                settings=_build_cpu_settings_snapshot_impl(settings),
                expected_backend_model_id=dependencies.expected_backend_model_id,
                runtime_device="cpu",
                runtime_dtype="float32",
                backend_factory=XLSRBackend,
            ),
            timeout_error_type=timeout_error_type,
            transient_error_type=transient_error_type,
            runtime_dependency_error_type=runtime_dependency_error_type,
            execution_error_type=execution_error_type,
            run_retry_policy_impl=_run_medium_retry_policy_impl,
            retry_delay_seconds=retry_delay_seconds,
            should_retry_on_cpu_after_transient_failure=should_retry_on_cpu_after_transient_failure,
            summarize_transient_failure=summarize_transient_failure,
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
    "run_medium_inference_from_public_boundary",
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
