"""Internal support owner for fast inference public-boundary wrappers."""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Literal

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
    """Spawn-safe fast worker error marker for unavailable model artifacts."""


class FastModelLoadError(RuntimeError):
    """Spawn-safe fast worker error marker for model load failures."""


def _load_fast_model_for_worker(
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None,
) -> LoadedModel:
    """Loads one fast model for the spawned worker process."""
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
    _ensure_fast_compatible_model(loaded_model, FastModelUnavailableError)
    return loaded_model


def _prepare_fast_process_operation(payload: FastProcessPayload) -> _PreparedFastOperation:
    """Builds one fast worker operation using module-level collaborators."""
    active_loaded_model = _load_fast_model_for_worker(payload.settings, loaded_model=None)
    return _PreparedFastOperation(loaded_model=active_loaded_model, request=payload.request)


def _run_fast_process_operation(prepared: _PreparedFastOperation) -> InferenceResult:
    """Runs one fast compute phase inside the spawned worker process."""
    return predict_emotions_detailed(
        prepared.request.file_path,
        loaded_model=prepared.loaded_model,
    )


def _fast_worker_entry(payload: FastProcessPayload, connection: Connection) -> None:
    """Executes one spawned fast worker using module-level collaborators only."""
    _run_worker_entry_binding(
        payload=payload,
        connection=connection,
        prepare_process_operation=_prepare_fast_process_operation,
        run_process_operation=_run_fast_process_operation,
    )


def run_fast_inference_from_public_boundary(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
    logger: logging.Logger,
    model_unavailable_error_type: type[Exception],
    model_load_error_type: type[Exception],
    timeout_error_type: type[Exception],
    execution_error_type: type[Exception],
    transient_error_type: type[Exception],
) -> InferenceResult:
    """Runs fast inference through the internal public-boundary owner."""

    worker_error_factories: dict[str, Callable[[str], Exception]] = {
        "ValueError": ValueError,
        model_unavailable_error_type.__name__: model_unavailable_error_type,
        model_load_error_type.__name__: model_load_error_type,
        timeout_error_type.__name__: timeout_error_type,
        transient_error_type.__name__: transient_error_type,
        "RuntimeError": RuntimeError,
    }

    def _retry_delay_seconds(base_delay: float, attempt: int) -> float:
        if base_delay <= 0.0:
            return 0.0
        return base_delay * float(attempt)

    def _load_fast_model(
        settings: AppConfig,
        *,
        loaded_model: LoadedModel | None,
    ) -> LoadedModel:
        if loaded_model is None:
            try:
                return load_model(
                    settings=settings,
                    expected_backend_id="handcrafted",
                    expected_profile="fast",
                )
            except FileNotFoundError as err:
                raise model_unavailable_error_type(str(err)) from err
            except ValueError as err:
                raise model_load_error_type(
                    "Failed to load fast-profile model artifact from configured paths."
                ) from err
        _ensure_fast_compatible_model(loaded_model, model_unavailable_error_type)
        return loaded_model

    def _run_fast_inference_once(
        *,
        request: InferenceRequest,
        loaded_model: LoadedModel | None,
        settings: AppConfig,
    ) -> InferenceResult:
        active_loaded_model = _load_fast_model(settings, loaded_model=loaded_model)
        return predict_emotions_detailed(
            request.file_path,
            loaded_model=active_loaded_model,
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
            worker_label="Fast inference",
            error_factory=execution_error_type,
        )

    def _is_setup_complete_message(message: WorkerMessage) -> bool:
        return _is_setup_complete_message_binding(
            message=message,
            impl=_is_setup_complete_message_impl,
            worker_label="Fast inference",
            error_factory=execution_error_type,
        )

    def _raise_worker_error(error_type: str, message: str) -> None:
        _raise_worker_error_binding(
            error_type=error_type,
            message=message,
            impl=_raise_worker_error_impl,
            known_error_factories=worker_error_factories,
            unknown_error_factory=execution_error_type,
            worker_label="Fast inference",
        )

    def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
        return _parse_worker_completion_message_binding(
            worker_message=worker_message,
            impl=_parse_worker_completion_message_impl,
            worker_label="Fast inference",
            error_factory=execution_error_type,
            raise_worker_error=_raise_worker_error,
            result_type=InferenceResult,
        )

    def _terminate_worker_process(process: BaseProcess) -> None:
        _terminate_worker_process_binding(
            process=process,
            impl=_terminate_worker_process_impl,
            terminate_grace_seconds=_TERMINATE_GRACE_SECONDS,
            kill_grace_seconds=_KILL_GRACE_SECONDS,
        )

    def _run_with_process_timeout(
        payload: FastProcessPayload,
        *,
        timeout_seconds: float,
    ) -> InferenceResult:
        return _run_with_process_timeout_impl(
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
            worker_target=_fast_worker_entry,
            recv_worker_message=_recv_worker_message,
            is_setup_complete_message=_is_setup_complete_message,
            terminate_worker_process=_terminate_worker_process,
            timeout_error_factory=timeout_error_type,
            execution_error_factory=execution_error_type,
            worker_label="Fast inference",
            process_join_grace_seconds=_TERMINATE_GRACE_SECONDS,
            parse_worker_completion_message=_parse_worker_completion_message,
        )

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
            settings=replace(settings, emotions=dict(settings.emotions)),
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
                        timeout_error_factory=timeout_error_type,
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
                timeout_error_type=timeout_error_type,
                transient_error_type=transient_error_type,
                transient_exhausted_error=lambda _err: execution_error_type(
                    "Fast inference exhausted retry budget after backend failures."
                ),
                retry_delay_seconds=_retry_delay_seconds,
                logger=logger,
            )
        except ValueError:
            raise
        except execution_error_type:
            raise
        except RuntimeError as err:
            raise execution_error_type(
                "Fast inference failed with a non-retryable runtime error."
            ) from err


def _ensure_fast_compatible_model(
    loaded_model: LoadedModel,
    unavailable_error_factory: Callable[[str], Exception] | type[Exception],
) -> None:
    """Validates that loaded artifact metadata is compatible with fast runtime."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise unavailable_error_factory(
            "Fast profile requires a v2 model artifact metadata envelope. "
            "Train a fast-profile model before inference."
        )
    if metadata.get("backend_id") != "handcrafted":
        raise unavailable_error_factory(
            "No fast-profile model artifact is available. "
            f"Found backend_id={metadata.get('backend_id')!r}; expected 'handcrafted'."
        )
    if metadata.get("profile") != "fast":
        raise unavailable_error_factory(
            "No fast-profile model artifact is available. "
            f"Found profile={metadata.get('profile')!r}; expected 'fast'."
        )


__all__ = ["run_fast_inference_from_public_boundary"]
