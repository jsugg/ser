"""Fast-profile inference runner with shared runtime policy semantics."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Literal

from ser._internal.runtime.process_timeout import (
    run_with_process_timeout as _run_with_process_timeout_orchestration,
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
from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel, load_model, predict_emotions_detailed
from ser.runtime.contracts import InferenceRequest
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.schema import InferenceResult
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
class FastProcessPayload:
    """Serializable payload for one process-isolated fast inference attempt."""

    request: InferenceRequest
    settings: AppConfig


@dataclass(frozen=True)
class _PreparedFastOperation:
    """Holds setup-complete data for one fast worker compute phase."""

    loaded_model: LoadedModel
    request: InferenceRequest


class FastModelUnavailableError(FileNotFoundError):
    """Raised when a compatible fast-profile model artifact is unavailable."""


class FastModelLoadError(RuntimeError):
    """Raised when fast model artifact loading fails unexpectedly."""


class FastInferenceTimeoutError(TimeoutError):
    """Raised when fast inference exceeds configured timeout budget."""


class FastInferenceExecutionError(RuntimeError):
    """Raised when fast inference exhausts retries without recovery."""


class FastTransientBackendError(RuntimeError):
    """Raised for retryable fast backend failures."""


_WORKER_ERROR_FACTORIES: dict[str, Callable[[str], Exception]] = {
    "ValueError": ValueError,
    "FastModelUnavailableError": FastModelUnavailableError,
    "FastModelLoadError": FastModelLoadError,
    "FastInferenceTimeoutError": FastInferenceTimeoutError,
    "FastTransientBackendError": FastTransientBackendError,
    "RuntimeError": RuntimeError,
}


def run_fast_inference(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
) -> InferenceResult:
    """Runs fast-profile inference with shared runtime timeout/retry policy."""
    runtime_config = settings.fast_runtime
    use_process_isolation = (
        enforce_timeout
        and loaded_model is None
        and settings.runtime_flags.profile_pipeline
        and runtime_config.process_isolation
    )

    process_payload: FastProcessPayload | None = None
    active_loaded_model: LoadedModel | None = None
    setup_started_at: float | None = None
    if use_process_isolation:
        process_payload = FastProcessPayload(
            request=request,
            settings=_build_process_settings_snapshot(settings),
        )
    else:
        setup_started_at = log_phase_started(
            logger,
            phase_name=PHASE_EMOTION_SETUP,
            profile="fast",
        )
        try:
            active_loaded_model = _load_fast_model(settings, loaded_model=loaded_model)
        except Exception:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_SETUP,
                started_at=setup_started_at,
                profile="fast",
            )
            raise
        log_phase_completed(
            logger,
            phase_name=PHASE_EMOTION_SETUP,
            started_at=setup_started_at,
            profile="fast",
        )
        setup_started_at = None

    with _SINGLE_FLIGHT_REGISTRY.lock(profile="fast", backend_model_id=None):

        def operation() -> InferenceResult:
            if enforce_timeout:
                if use_process_isolation:
                    if process_payload is None:
                        raise RuntimeError(
                            "Fast process payload is missing for isolated execution."
                        )
                    return _run_with_process_timeout(
                        process_payload,
                        timeout_seconds=runtime_config.timeout_seconds,
                    )
                inference_started_at = log_phase_started(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    profile="fast",
                )
                try:
                    result = _run_with_timeout_impl(
                        operation=lambda: _run_fast_inference_once(
                            request=request,
                            loaded_model=active_loaded_model,
                            settings=settings,
                        ),
                        timeout_seconds=runtime_config.timeout_seconds,
                        timeout_error_factory=FastInferenceTimeoutError,
                        timeout_label="Fast inference",
                    )
                except Exception:
                    log_phase_failed(
                        logger,
                        phase_name=PHASE_EMOTION_INFERENCE,
                        started_at=inference_started_at,
                        profile="fast",
                    )
                    raise
                log_phase_completed(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    started_at=inference_started_at,
                    profile="fast",
                )
                return result
            inference_started_at = log_phase_started(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                profile="fast",
            )
            try:
                result = _run_fast_inference_once(
                    request=request,
                    loaded_model=active_loaded_model,
                    settings=settings,
                )
            except Exception:
                log_phase_failed(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    started_at=inference_started_at,
                    profile="fast",
                )
                raise
            log_phase_completed(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                started_at=inference_started_at,
                profile="fast",
            )
            return result

        try:
            return run_with_retry_policy(
                operation=operation,
                runtime_config=runtime_config,
                allow_retries=allow_retries,
                profile_label="Fast",
                timeout_error_type=FastInferenceTimeoutError,
                transient_error_type=FastTransientBackendError,
                transient_exhausted_error=lambda _err: FastInferenceExecutionError(
                    "Fast inference exhausted retry budget after backend failures."
                ),
                retry_delay_seconds=_retry_delay_seconds,
                logger=logger,
            )
        except ValueError:
            raise
        except FastInferenceExecutionError:
            raise
        except RuntimeError as err:
            raise FastInferenceExecutionError(
                "Fast inference failed with a non-retryable runtime error."
            ) from err


def _load_fast_model(
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None,
) -> LoadedModel:
    """Loads and validates fast model metadata when model is not injected."""
    if loaded_model is None:
        try:
            return load_model(
                settings=settings,
                expected_backend_id="handcrafted",
                expected_profile="fast",
            )
        except FileNotFoundError as err:
            raise FastModelUnavailableError(str(err)) from err
        except ValueError as err:
            raise FastModelLoadError(
                "Failed to load fast-profile model artifact from configured paths."
            ) from err
    _ensure_fast_compatible_model(loaded_model)
    return loaded_model


def _run_fast_inference_once(
    *,
    request: InferenceRequest,
    loaded_model: LoadedModel | None,
    settings: AppConfig,
) -> InferenceResult:
    """Runs one fast inference attempt without retry control."""
    active_loaded_model = _load_fast_model(settings, loaded_model=loaded_model)
    return predict_emotions_detailed(
        request.file_path,
        loaded_model=active_loaded_model,
    )


def _run_with_process_timeout(
    payload: FastProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    return _run_with_process_timeout_orchestration(
        payload=payload,
        resolve_profile=lambda _payload: "fast",
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
        timeout_error_factory=FastInferenceTimeoutError,
        execution_error_factory=FastInferenceExecutionError,
        worker_label="Fast inference",
        process_join_grace_seconds=_TERMINATE_GRACE_SECONDS,
        parse_worker_completion_message=_parse_worker_completion_message,
    )


def _recv_worker_message(
    connection: Connection,
    *,
    stage: str,
) -> WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    return _recv_worker_message_binding(
        connection=connection,
        stage=stage,
        impl=_recv_worker_message_impl,
        worker_label="Fast inference",
        error_factory=FastInferenceExecutionError,
    )


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    return _is_setup_complete_message_binding(
        message=message,
        impl=_is_setup_complete_message_impl,
        worker_label="Fast inference",
        error_factory=FastInferenceExecutionError,
    )


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    return _parse_worker_completion_message_binding(
        worker_message=worker_message,
        impl=_parse_worker_completion_message_impl,
        worker_label="Fast inference",
        error_factory=FastInferenceExecutionError,
        raise_worker_error=_raise_worker_error,
        result_type=InferenceResult,
    )


def _worker_entry(payload: FastProcessPayload, connection: Connection) -> None:
    """Executes one fast inference operation inside child process."""
    _run_worker_entry_binding(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_process_operation,
        run_process_operation=_run_process_operation,
    )


def _prepare_process_operation(payload: FastProcessPayload) -> _PreparedFastOperation:
    """Performs untimed setup for one process-isolated fast operation."""
    loaded_model = _load_fast_model(payload.settings, loaded_model=None)
    return _PreparedFastOperation(loaded_model=loaded_model, request=payload.request)


def _run_process_operation(prepared: _PreparedFastOperation) -> InferenceResult:
    """Runs one fast compute phase inside isolated worker process."""
    return predict_emotions_detailed(
        prepared.request.file_path,
        loaded_model=prepared.loaded_model,
    )


def _build_process_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Builds a process-safe settings snapshot for spawn-based workers."""
    return replace(settings, emotions=dict(settings.emotions))


def _ensure_fast_compatible_model(loaded_model: LoadedModel) -> None:
    """Validates that loaded artifact metadata is compatible with fast runtime."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise FastModelUnavailableError(
            "Fast profile requires a v2 model artifact metadata envelope. "
            "Train a fast-profile model before inference."
        )
    if metadata.get("backend_id") != "handcrafted":
        raise FastModelUnavailableError(
            "No fast-profile model artifact is available. "
            f"Found backend_id={metadata.get('backend_id')!r}; expected 'handcrafted'."
        )
    if metadata.get("profile") != "fast":
        raise FastModelUnavailableError(
            "No fast-profile model artifact is available. "
            f"Found profile={metadata.get('profile')!r}; expected 'fast'."
        )


def _terminate_worker_process(process: BaseProcess) -> None:
    """Terminates a timed-out worker process with kill fallback."""
    _terminate_worker_process_binding(
        process=process,
        impl=_terminate_worker_process_impl,
        terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
        kill_grace_seconds=_KILL_GRACE_SECONDS,
    )


def _raise_worker_error(error_type: str, message: str) -> None:
    """Rehydrates child-process errors into runtime-domain exceptions."""
    _raise_worker_error_binding(
        error_type=error_type,
        message=message,
        impl=_raise_worker_error_impl,
        known_error_factories=_WORKER_ERROR_FACTORIES,
        unknown_error_factory=FastInferenceExecutionError,
        worker_label="Fast inference",
    )


def _retry_delay_seconds(base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay for one retry attempt."""
    if base_delay <= 0.0:
        return 0.0
    return base_delay * float(attempt)
