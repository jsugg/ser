"""Fast-profile inference runner with shared runtime policy semantics."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Lock
from typing import Literal, cast

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

type InferenceRunner = Callable[[], InferenceResult]
type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_LOCKS: dict[tuple[str, str], Lock] = {}
_SINGLE_FLIGHT_GUARD = Lock()


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

    with _single_flight_lock(profile="fast", backend_model_id=None):

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
                    result = _run_with_timeout(
                        lambda: _run_fast_inference_once(
                            request=request,
                            loaded_model=active_loaded_model,
                            settings=settings,
                        ),
                        timeout_seconds=runtime_config.timeout_seconds,
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


def _run_with_timeout(
    operation: InferenceRunner,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs an inference operation with best-effort timeout enforcement."""
    if timeout_seconds <= 0.0:
        return operation()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(operation)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as err:
        future.cancel()
        raise FastInferenceTimeoutError(
            f"Fast inference exceeded timeout budget ({timeout_seconds:.2f}s)."
        ) from err
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_with_process_timeout(
    payload: FastProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    setup_started_at = log_phase_started(
        logger,
        phase_name=PHASE_EMOTION_SETUP,
        profile="fast",
    )
    setup_completed = False
    inference_started_at: float | None = None
    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(payload, child_conn),
        daemon=False,
    )
    process.start()
    child_conn.close()
    try:
        try:
            setup_message = _recv_worker_message(parent_conn, stage="setup")
            if _is_setup_complete_message(setup_message):
                setup_completed = True
                log_phase_completed(
                    logger,
                    phase_name=PHASE_EMOTION_SETUP,
                    started_at=setup_started_at,
                    profile="fast",
                )
                inference_started_at = log_phase_started(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    profile="fast",
                )
                if timeout_seconds <= 0.0:
                    worker_message = _recv_worker_message(parent_conn, stage="compute")
                else:
                    if not parent_conn.poll(timeout_seconds):
                        if process.is_alive():
                            _terminate_worker_process(process)
                            raise FastInferenceTimeoutError(
                                f"Fast inference exceeded timeout budget ({timeout_seconds:.2f}s)."
                            )
                        raise FastInferenceExecutionError(
                            "Fast inference worker exited before sending compute result."
                        )
                    worker_message = _recv_worker_message(parent_conn, stage="compute")
            else:
                worker_message = setup_message
            if process.is_alive():
                process.join(timeout=_TERMINATE_GRACE_SECONDS)
            if process.exitcode not in (0, None) and not parent_conn.poll():
                raise FastInferenceExecutionError(
                    "Fast inference worker exited without a result payload."
                )
        finally:
            parent_conn.close()
            if process.is_alive():
                _terminate_worker_process(process)
    except Exception:
        if inference_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                started_at=inference_started_at,
                profile="fast",
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_SETUP,
                started_at=setup_started_at,
                profile="fast",
            )
        raise

    try:
        result = _parse_worker_completion_message(worker_message)
    except Exception:
        if inference_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                started_at=inference_started_at,
                profile="fast",
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_SETUP,
                started_at=setup_started_at,
                profile="fast",
            )
        raise
    if inference_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=PHASE_EMOTION_INFERENCE,
            started_at=inference_started_at,
            profile="fast",
        )
    return result


def _recv_worker_message(
    connection: Connection,
    *,
    stage: str,
) -> WorkerMessage:
    """Receives one worker message and validates tuple envelope shape."""
    try:
        raw_message = connection.recv()
    except EOFError as err:
        raise FastInferenceExecutionError(
            f"Fast inference worker exited before sending {stage} payload."
        ) from err
    if not isinstance(raw_message, tuple) or not raw_message:
        raise FastInferenceExecutionError(
            "Fast inference worker returned malformed payload."
        )
    return cast(WorkerMessage, raw_message)


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    if message[0] != "phase":
        return False
    if len(message) != 2 or message[1] != "setup_complete":
        raise FastInferenceExecutionError(
            "Fast inference worker returned malformed phase payload."
        )
    return True


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    kind = worker_message[0]
    if kind == "ok":
        if len(worker_message) != 2:
            raise FastInferenceExecutionError(
                "Fast inference worker returned malformed success payload."
            )
        result = worker_message[1]
        if not isinstance(result, InferenceResult):
            raise FastInferenceExecutionError(
                "Fast inference worker returned unexpected result type."
            )
        return result
    if kind == "phase":
        raise FastInferenceExecutionError(
            "Fast inference worker returned setup phase without completion payload."
        )
    if kind == "err":
        if len(worker_message) != 3:
            raise FastInferenceExecutionError(
                "Fast inference worker returned malformed error payload."
            )
        _raise_worker_error(worker_message[1], worker_message[2])
    raise FastInferenceExecutionError(
        "Fast inference worker returned unknown payload status."
    )


def _worker_entry(payload: FastProcessPayload, connection: Connection) -> None:
    """Executes one fast inference operation inside child process."""
    try:
        prepared_operation = _prepare_process_operation(payload)
        connection.send(("phase", "setup_complete"))
        result = _run_process_operation(prepared_operation)
        connection.send(("ok", result))
    except BaseException as err:
        connection.send(("err", type(err).__name__, str(err)))
    finally:
        connection.close()


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
    process.terminate()
    process.join(timeout=_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_KILL_GRACE_SECONDS)


def _raise_worker_error(error_type: str, message: str) -> None:
    """Rehydrates child-process errors into runtime-domain exceptions."""
    if error_type == "ValueError":
        raise ValueError(message)
    if error_type == "FastModelUnavailableError":
        raise FastModelUnavailableError(message)
    if error_type == "FastModelLoadError":
        raise FastModelLoadError(message)
    if error_type == "FastInferenceTimeoutError":
        raise FastInferenceTimeoutError(message)
    if error_type == "FastTransientBackendError":
        raise FastTransientBackendError(message)
    if error_type == "RuntimeError":
        raise RuntimeError(message)
    raise FastInferenceExecutionError(
        f"Fast inference worker failed with {error_type}: {message}"
    )


@contextmanager
def _single_flight_lock(
    *,
    profile: str,
    backend_model_id: str | None,
) -> Iterator[None]:
    """Serializes in-process inference calls for one profile/model tuple."""
    model_key = (
        backend_model_id.strip()
        if isinstance(backend_model_id, str) and backend_model_id.strip()
        else "__default__"
    )
    key = (profile, model_key)
    with _SINGLE_FLIGHT_GUARD:
        lock = _SINGLE_FLIGHT_LOCKS.setdefault(key, Lock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _retry_delay_seconds(base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay for one retry attempt."""
    if base_delay <= 0.0:
        return 0.0
    return base_delay * float(attempt)
