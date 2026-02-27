"""Accurate-profile inference runner with bounded retries and timeout guards."""

from __future__ import annotations

import multiprocessing as mp
import random
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from threading import Lock
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.models.emotion_model import (
    LoadedModel,
    load_model,
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
)
from ser.pool import mean_std_pool, temporal_pooling_windows
from ser.repr import (
    Emotion2VecBackend,
    EncodedSequence,
    FeatureBackend,
    PoolingWindow,
    WhisperBackend,
)
from ser.repr.runtime_policy import resolve_feature_runtime_policy
from ser.runtime.contracts import InferenceRequest
from ser.runtime.phase_contract import PHASE_EMOTION_INFERENCE, PHASE_EMOTION_SETUP
from ser.runtime.phase_timing import (
    log_phase_completed,
    log_phase_failed,
    log_phase_started,
)
from ser.runtime.policy import run_with_retry_policy
from ser.runtime.postprocessing import (
    SegmentPostprocessingConfig,
    postprocess_frame_predictions,
)
from ser.runtime.schema import (
    OUTPUT_SCHEMA_VERSION,
    FramePrediction,
    InferenceResult,
    SegmentPrediction,
)
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger = get_logger(__name__)

type FeatureMatrix = NDArray[np.float64]
type InferenceRunner = Callable[[], InferenceResult]
type WorkerPhaseMessage = tuple[Literal["phase"], Literal["setup_complete"]]
type WorkerSuccessMessage = tuple[Literal["ok"], InferenceResult]
type WorkerErrorMessage = tuple[Literal["err"], str, str]
type WorkerMessage = WorkerPhaseMessage | WorkerSuccessMessage | WorkerErrorMessage

_TERMINATE_GRACE_SECONDS = 0.5
_KILL_GRACE_SECONDS = 0.5
_SINGLE_FLIGHT_LOCKS: dict[tuple[str, str], Lock] = {}
_SINGLE_FLIGHT_GUARD = Lock()
_MPS_OOM_SIGNATURE = "mps backend out of memory"
_MPS_ALLOCATED_PATTERN = re.compile(
    r"mps allocated:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_OTHER_ALLOC_PATTERN = re.compile(
    r"other allocations:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_MAX_ALLOWED_PATTERN = re.compile(
    r"max allowed:\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)
_MPS_REQUIRED_PATTERN = re.compile(
    r"tried to allocate\s*([0-9]+(?:\.[0-9]+)?)\s*(kb|mb|gb|tb)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AccurateProcessPayload:
    """Serializable payload for one process-isolated accurate inference attempt."""

    request: InferenceRequest
    settings: AppConfig
    expected_backend_id: str
    expected_profile: str
    expected_backend_model_id: str | None


@dataclass(frozen=True)
class _PreparedAccurateOperation:
    """Holds setup-complete data for one accurate worker compute phase."""

    loaded_model: LoadedModel
    backend: FeatureBackend
    audio: NDArray[np.float32]
    sample_rate: int
    runtime_config: ProfileRuntimeConfig


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
    runtime_config = _runtime_config_for_profile(settings, expected_profile)
    resolved_expected_backend_model_id = expected_backend_model_id
    if (
        resolved_expected_backend_model_id is None
        and expected_backend_id == "hf_whisper"
    ):
        resolved_expected_backend_model_id = resolve_accurate_model_id(settings)
    use_process_isolation = (
        enforce_timeout
        and loaded_model is None
        and backend is None
        and settings.runtime_flags.profile_pipeline
        and runtime_config.process_isolation
    )

    process_payload: AccurateProcessPayload | None = None
    active_loaded_model: LoadedModel | None = None
    active_backend: FeatureBackend | None = None
    audio_array: NDArray[np.float32] | None = None
    sample_rate = 0
    setup_started_at: float | None = None
    if use_process_isolation:
        process_payload = AccurateProcessPayload(
            request=request,
            settings=_build_process_settings_snapshot(settings),
            expected_backend_id=expected_backend_id,
            expected_profile=expected_profile,
            expected_backend_model_id=resolved_expected_backend_model_id,
        )
    else:
        setup_started_at = log_phase_started(
            logger,
            phase_name=PHASE_EMOTION_SETUP,
            profile=expected_profile,
        )
        if loaded_model is None:
            try:
                active_loaded_model = load_model(
                    settings=settings,
                    expected_backend_id=expected_backend_id,
                    expected_profile=expected_profile,
                    expected_backend_model_id=resolved_expected_backend_model_id,
                )
            except FileNotFoundError as err:
                if setup_started_at is not None:
                    log_phase_failed(
                        logger,
                        phase_name=PHASE_EMOTION_SETUP,
                        started_at=setup_started_at,
                        profile=expected_profile,
                    )
                raise AccurateModelUnavailableError(str(err)) from err
            except ValueError as err:
                if setup_started_at is not None:
                    log_phase_failed(
                        logger,
                        phase_name=PHASE_EMOTION_SETUP,
                        started_at=setup_started_at,
                        profile=expected_profile,
                    )
                raise AccurateModelLoadError(
                    "Failed to load accurate-profile model artifact from configured paths."
                ) from err
        else:
            active_loaded_model = loaded_model
        try:
            _ensure_accurate_compatible_model(
                active_loaded_model,
                expected_backend_id=expected_backend_id,
                expected_profile=expected_profile,
                expected_backend_model_id=resolved_expected_backend_model_id,
            )
            runtime_override = settings.feature_runtime_policy.for_backend(
                expected_backend_id
            )
            _warn_on_runtime_selector_mismatch(
                loaded_model=active_loaded_model,
                backend_id=expected_backend_id,
                requested_device=settings.torch_runtime.device,
                requested_dtype=settings.torch_runtime.dtype,
                backend_override_device=(
                    runtime_override.device if runtime_override is not None else None
                ),
                backend_override_dtype=(
                    runtime_override.dtype if runtime_override is not None else None
                ),
                profile=expected_profile,
            )
            audio, sample_rate = read_audio_file(request.file_path)
            audio_array = np.asarray(audio, dtype=np.float32)
            active_backend = (
                backend
                if backend is not None
                else _build_backend_for_profile(
                    expected_backend_id=expected_backend_id,
                    expected_backend_model_id=resolved_expected_backend_model_id,
                    settings=settings,
                )
            )
        except Exception:
            if setup_started_at is not None:
                log_phase_failed(
                    logger,
                    phase_name=PHASE_EMOTION_SETUP,
                    started_at=setup_started_at,
                    profile=expected_profile,
                )
            raise
    with _single_flight_lock(
        profile=expected_profile,
        backend_model_id=resolved_expected_backend_model_id,
    ):
        cpu_fallback_applied = False

        if not use_process_isolation:
            if active_backend is None:
                raise RuntimeError(
                    "Accurate backend is missing for in-process inference."
                )
            try:
                _prepare_accurate_backend_runtime(backend=active_backend)
            except Exception:
                if setup_started_at is not None:
                    log_phase_failed(
                        logger,
                        phase_name=PHASE_EMOTION_SETUP,
                        started_at=setup_started_at,
                        profile=expected_profile,
                    )
                raise
            if setup_started_at is not None:
                log_phase_completed(
                    logger,
                    phase_name=PHASE_EMOTION_SETUP,
                    started_at=setup_started_at,
                    profile=expected_profile,
                )
                setup_started_at = None

        def operation() -> InferenceResult:
            if enforce_timeout:
                if use_process_isolation:
                    if process_payload is None:
                        raise RuntimeError(
                            "Accurate process payload is missing for isolated execution."
                        )
                    return _run_with_process_timeout(
                        process_payload,
                        timeout_seconds=runtime_config.timeout_seconds,
                    )
                inference_started_at = log_phase_started(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    profile=expected_profile,
                )

                def timeout_operation() -> InferenceResult:
                    if (
                        active_loaded_model is None
                        or active_backend is None
                        or audio_array is None
                    ):
                        raise RuntimeError(
                            "Accurate inference operation prerequisites are missing."
                        )
                    return _run_accurate_inference_once(
                        loaded_model=active_loaded_model,
                        backend=active_backend,
                        audio=audio_array,
                        sample_rate=sample_rate,
                        runtime_config=runtime_config,
                    )

                try:
                    result = _run_with_timeout(
                        timeout_operation,
                        timeout_seconds=runtime_config.timeout_seconds,
                    )
                except Exception:
                    log_phase_failed(
                        logger,
                        phase_name=PHASE_EMOTION_INFERENCE,
                        started_at=inference_started_at,
                        profile=expected_profile,
                    )
                    raise
                log_phase_completed(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    started_at=inference_started_at,
                    profile=expected_profile,
                )
                return result
            if (
                active_loaded_model is None
                or active_backend is None
                or audio_array is None
            ):
                raise RuntimeError(
                    "Accurate inference operation prerequisites are missing."
                )
            inference_started_at = log_phase_started(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                profile=expected_profile,
            )
            try:
                result = _run_accurate_inference_once(
                    loaded_model=active_loaded_model,
                    backend=active_backend,
                    audio=audio_array,
                    sample_rate=sample_rate,
                    runtime_config=runtime_config,
                )
            except Exception:
                log_phase_failed(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    started_at=inference_started_at,
                    profile=expected_profile,
                )
                raise
            log_phase_completed(
                logger,
                phase_name=PHASE_EMOTION_INFERENCE,
                started_at=inference_started_at,
                profile=expected_profile,
            )
            return result

        def on_transient_failure(
            err: Exception,
            _attempt: int,
            _transient_failures: int,
        ) -> None:
            nonlocal active_backend, process_payload, cpu_fallback_applied
            if cpu_fallback_applied:
                return
            if expected_backend_id != "hf_whisper":
                return
            if not _is_mps_out_of_memory_error(err):
                return
            if not use_process_isolation and backend is not None:
                logger.info(
                    "Accurate inference detected MPS OOM but cannot switch "
                    "to CPU when a custom backend instance is injected."
                )
                return

            current_device = (
                process_payload.settings.torch_runtime.device
                if use_process_isolation and process_payload is not None
                else settings.torch_runtime.device
            )
            normalized_device = current_device.strip().lower()
            if normalized_device not in {"auto", "mps"}:
                return
            memory_summary = _summarize_mps_oom_memory(str(err))
            logger.info(
                "Accurate inference will retry on CPU after MPS OOM (%s).",
                memory_summary,
            )
            cpu_fallback_applied = True
            try:
                if use_process_isolation:
                    if process_payload is None:
                        raise RuntimeError(
                            "Accurate process payload is missing for CPU fallback."
                        )
                    process_payload = replace(
                        process_payload,
                        settings=_settings_with_torch_device(
                            process_payload.settings,
                            device="cpu",
                        ),
                    )
                    return

                cpu_settings = _settings_with_torch_device(settings, device="cpu")
                active_backend = _build_backend_for_profile(
                    expected_backend_id=expected_backend_id,
                    expected_backend_model_id=resolved_expected_backend_model_id,
                    settings=cpu_settings,
                )
                _prepare_accurate_backend_runtime(backend=active_backend)
            except Exception as swap_err:
                cpu_fallback_applied = False
                logger.warning(
                    "Accurate inference CPU fallback setup failed; "
                    "continuing with existing retry policy: %s",
                    swap_err,
                )

        try:
            return run_with_retry_policy(
                operation=operation,
                runtime_config=runtime_config,
                allow_retries=allow_retries,
                profile_label="Accurate",
                timeout_error_type=AccurateInferenceTimeoutError,
                transient_error_type=AccurateTransientBackendError,
                transient_exhausted_error=lambda err: AccurateInferenceExecutionError(
                    "Accurate inference exhausted retry budget after backend failures."
                ),
                retry_delay_seconds=_retry_delay_seconds,
                logger=logger,
                on_transient_failure=on_transient_failure,
            )
        except AccurateRuntimeDependencyError:
            raise
        except ValueError:
            raise
        except AccurateInferenceExecutionError:
            raise
        except RuntimeError as err:
            raise AccurateInferenceExecutionError(
                "Accurate inference failed with a non-retryable runtime error."
            ) from err


def _run_accurate_inference_once(
    *,
    loaded_model: LoadedModel,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    runtime_config: ProfileRuntimeConfig,
) -> InferenceResult:
    """Runs one accurate inference attempt without retry control."""
    encoded = _encode_accurate_sequence(
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
            "Feature vector size mismatch for accurate-profile model. "
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
        segments=_segment_predictions(
            frame_predictions,
            runtime_config=runtime_config,
        ),
        frames=frame_predictions,
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
        raise AccurateInferenceTimeoutError(
            f"Accurate inference exceeded timeout budget ({timeout_seconds:.2f}s)."
        ) from err
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _run_with_process_timeout(
    payload: AccurateProcessPayload,
    *,
    timeout_seconds: float,
) -> InferenceResult:
    """Runs one process-isolated attempt with timeout applied only to compute."""
    setup_started_at = log_phase_started(
        logger,
        phase_name=PHASE_EMOTION_SETUP,
        profile=payload.expected_profile,
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
                    profile=payload.expected_profile,
                )
                inference_started_at = log_phase_started(
                    logger,
                    phase_name=PHASE_EMOTION_INFERENCE,
                    profile=payload.expected_profile,
                )
                if timeout_seconds <= 0.0:
                    worker_message = _recv_worker_message(parent_conn, stage="compute")
                else:
                    if not parent_conn.poll(timeout_seconds):
                        if process.is_alive():
                            _terminate_worker_process(process)
                            raise AccurateInferenceTimeoutError(
                                "Accurate inference exceeded timeout budget "
                                f"({timeout_seconds:.2f}s)."
                            )
                        raise AccurateInferenceExecutionError(
                            "Accurate inference worker exited before sending compute result."
                        )
                    worker_message = _recv_worker_message(parent_conn, stage="compute")
            else:
                worker_message = setup_message
            if process.is_alive():
                process.join(timeout=_TERMINATE_GRACE_SECONDS)
            if process.exitcode not in (0, None) and not parent_conn.poll():
                raise AccurateInferenceExecutionError(
                    "Accurate inference worker exited without a result payload."
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
                profile=payload.expected_profile,
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_SETUP,
                started_at=setup_started_at,
                profile=payload.expected_profile,
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
                profile=payload.expected_profile,
            )
        elif not setup_completed:
            log_phase_failed(
                logger,
                phase_name=PHASE_EMOTION_SETUP,
                started_at=setup_started_at,
                profile=payload.expected_profile,
            )
        raise
    if inference_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=PHASE_EMOTION_INFERENCE,
            started_at=inference_started_at,
            profile=payload.expected_profile,
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
        raise AccurateInferenceExecutionError(
            f"Accurate inference worker exited before sending {stage} payload."
        ) from err
    if not isinstance(raw_message, tuple) or not raw_message:
        raise AccurateInferenceExecutionError(
            "Accurate inference worker returned malformed payload."
        )
    return cast(WorkerMessage, raw_message)


def _is_setup_complete_message(message: WorkerMessage) -> bool:
    """Returns whether one worker message marks setup completion."""
    if message[0] != "phase":
        return False
    if len(message) != 2 or message[1] != "setup_complete":
        raise AccurateInferenceExecutionError(
            "Accurate inference worker returned malformed phase payload."
        )
    return True


def _parse_worker_completion_message(worker_message: WorkerMessage) -> InferenceResult:
    """Parses one worker completion message and returns inference result."""
    kind = worker_message[0]
    if kind == "ok":
        if len(worker_message) != 2:
            raise AccurateInferenceExecutionError(
                "Accurate inference worker returned malformed success payload."
            )
        result = worker_message[1]
        if not isinstance(result, InferenceResult):
            raise AccurateInferenceExecutionError(
                "Accurate inference worker returned unexpected result type."
            )
        return result
    if kind == "phase":
        raise AccurateInferenceExecutionError(
            "Accurate inference worker returned setup phase without completion payload."
        )
    if kind == "err":
        if len(worker_message) != 3:
            raise AccurateInferenceExecutionError(
                "Accurate inference worker returned malformed error payload."
            )
        _raise_worker_error(worker_message[1], worker_message[2])
    raise AccurateInferenceExecutionError(
        "Accurate inference worker returned unknown payload status."
    )


def _worker_entry(
    payload: AccurateProcessPayload,
    connection: Connection,
) -> None:
    """Executes one inference operation inside child process."""
    try:
        prepared_operation = _prepare_process_operation(payload)
        connection.send(("phase", "setup_complete"))
        result = _run_process_operation(prepared_operation)
        connection.send(("ok", result))
    except BaseException as err:
        connection.send(("err", type(err).__name__, str(err)))
    finally:
        connection.close()


def _prepare_process_operation(
    payload: AccurateProcessPayload,
) -> _PreparedAccurateOperation:
    """Performs untimed setup for one process-isolated accurate operation."""
    settings = payload.settings
    runtime_config = _runtime_config_for_profile(settings, payload.expected_profile)
    try:
        loaded_model = load_model(
            settings=settings,
            expected_backend_id=payload.expected_backend_id,
            expected_profile=payload.expected_profile,
            expected_backend_model_id=payload.expected_backend_model_id,
        )
    except FileNotFoundError as err:
        raise AccurateModelUnavailableError(str(err)) from err
    except ValueError as err:
        raise AccurateModelLoadError(
            "Failed to load accurate-profile model artifact from configured paths."
        ) from err
    _ensure_accurate_compatible_model(
        loaded_model,
        expected_backend_id=payload.expected_backend_id,
        expected_profile=payload.expected_profile,
        expected_backend_model_id=payload.expected_backend_model_id,
    )
    runtime_override = settings.feature_runtime_policy.for_backend(
        payload.expected_backend_id
    )
    _warn_on_runtime_selector_mismatch(
        loaded_model=loaded_model,
        backend_id=payload.expected_backend_id,
        requested_device=settings.torch_runtime.device,
        requested_dtype=settings.torch_runtime.dtype,
        backend_override_device=(
            runtime_override.device if runtime_override is not None else None
        ),
        backend_override_dtype=(
            runtime_override.dtype if runtime_override is not None else None
        ),
        profile=payload.expected_profile,
    )
    audio, sample_rate = read_audio_file(payload.request.file_path)
    audio_array = np.asarray(audio, dtype=np.float32)
    backend = _build_backend_for_profile(
        expected_backend_id=payload.expected_backend_id,
        expected_backend_model_id=payload.expected_backend_model_id,
        settings=settings,
    )
    _prepare_accurate_backend_runtime(backend=backend)
    return _PreparedAccurateOperation(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio_array,
        sample_rate=sample_rate,
        runtime_config=runtime_config,
    )


def _run_process_operation(prepared: _PreparedAccurateOperation) -> InferenceResult:
    """Runs one accurate compute phase inside isolated worker process."""
    return _run_accurate_inference_once(
        loaded_model=prepared.loaded_model,
        backend=prepared.backend,
        audio=prepared.audio,
        sample_rate=prepared.sample_rate,
        runtime_config=prepared.runtime_config,
    )


def _build_process_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Builds a process-safe settings snapshot for spawn-based workers."""
    return _settings_with_torch_device(
        settings,
        device=settings.torch_runtime.device,
    )


def _settings_with_torch_device(settings: AppConfig, *, device: str) -> AppConfig:
    """Returns process-safe settings with one normalized torch device selector."""
    return replace(
        settings,
        emotions=dict(settings.emotions),
        torch_runtime=replace(settings.torch_runtime, device=device),
    )


def _runtime_config_for_profile(
    settings: AppConfig,
    expected_profile: str,
) -> ProfileRuntimeConfig:
    """Returns runtime config for one accurate-profile variant."""
    if expected_profile == "accurate":
        return settings.accurate_runtime
    if expected_profile == "accurate-research":
        return settings.accurate_research_runtime
    raise AccurateModelUnavailableError(
        f"Unsupported accurate runtime profile {expected_profile!r}."
    )


def _build_backend_for_profile(
    *,
    expected_backend_id: str,
    expected_backend_model_id: str | None,
    settings: AppConfig,
) -> FeatureBackend:
    """Builds a feature backend aligned with profile/backend runtime expectations."""
    backend_override = settings.feature_runtime_policy.for_backend(expected_backend_id)
    if expected_backend_id == "hf_whisper":
        model_id = (
            expected_backend_model_id
            if expected_backend_model_id is not None
            else resolve_accurate_model_id(settings)
        )
        runtime_policy = resolve_feature_runtime_policy(
            backend_id=expected_backend_id,
            requested_device=settings.torch_runtime.device,
            requested_dtype=settings.torch_runtime.dtype,
            backend_override_device=(
                backend_override.device if backend_override is not None else None
            ),
            backend_override_dtype=(
                backend_override.dtype if backend_override is not None else None
            ),
        )
        return WhisperBackend(
            model_id=model_id,
            cache_dir=settings.models.huggingface_cache_root,
            device=runtime_policy.device,
            dtype=runtime_policy.dtype,
        )
    if expected_backend_id == "emotion2vec":
        model_id = (
            expected_backend_model_id
            if expected_backend_model_id is not None
            else resolve_accurate_research_model_id(settings)
        )
        runtime_policy = resolve_feature_runtime_policy(
            backend_id=expected_backend_id,
            requested_device=settings.torch_runtime.device,
            requested_dtype=settings.torch_runtime.dtype,
            backend_override_device=(
                backend_override.device if backend_override is not None else None
            ),
            backend_override_dtype=(
                backend_override.dtype if backend_override is not None else None
            ),
        )
        return Emotion2VecBackend(
            model_id=model_id,
            device=runtime_policy.device,
            modelscope_cache_root=settings.models.modelscope_cache_root,
            huggingface_cache_root=settings.models.huggingface_cache_root,
        )
    raise AccurateModelUnavailableError(
        f"Unsupported accurate runtime backend id {expected_backend_id!r}."
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
    if error_type == "AccurateRuntimeDependencyError":
        raise AccurateRuntimeDependencyError(message)
    if error_type == "AccurateTransientBackendError":
        raise AccurateTransientBackendError(message)
    if error_type == "AccurateModelUnavailableError":
        raise AccurateModelUnavailableError(message)
    if error_type == "AccurateModelLoadError":
        raise AccurateModelLoadError(message)
    if error_type == "AccurateInferenceTimeoutError":
        raise AccurateInferenceTimeoutError(message)
    if error_type == "RuntimeError":
        raise RuntimeError(message)
    raise AccurateInferenceExecutionError(
        f"Accurate inference worker failed with {error_type}: {message}"
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


def _retry_delay_seconds(*, base_delay: float, attempt: int) -> float:
    """Returns bounded retry delay with small jitter."""
    if base_delay <= 0.0:
        return 0.0
    jitter = random.uniform(0.0, base_delay * 0.1)
    return (base_delay * float(attempt)) + jitter


def _is_mps_out_of_memory_error(err: Exception) -> bool:
    """Returns whether one error message indicates MPS out-of-memory pressure."""
    return _MPS_OOM_SIGNATURE in str(err).strip().lower()


def _summarize_mps_oom_memory(message: str) -> str:
    """Builds a compact required/available memory summary for MPS OOM logs."""
    required_mib = _extract_memory_mib(message, _MPS_REQUIRED_PATTERN)
    allocated_mib = _extract_memory_mib(message, _MPS_ALLOCATED_PATTERN)
    other_allocated_mib = _extract_memory_mib(message, _MPS_OTHER_ALLOC_PATTERN) or 0.0
    max_allowed_mib = _extract_memory_mib(message, _MPS_MAX_ALLOWED_PATTERN)
    if (
        required_mib is not None
        and allocated_mib is not None
        and max_allowed_mib is not None
    ):
        available_mib = max(max_allowed_mib - allocated_mib - other_allocated_mib, 0.0)
        return (
            f"required={_format_memory_mib(required_mib)} "
            f"available={_format_memory_mib(available_mib)}"
        )
    if required_mib is not None:
        return f"required={_format_memory_mib(required_mib)}"
    return "memory headroom exhausted"


def _extract_memory_mib(message: str, pattern: re.Pattern[str]) -> float | None:
    """Extracts one memory quantity from message text and converts it to MiB."""
    match = pattern.search(message)
    if match is None:
        return None
    amount_text, unit_text = match.groups()
    try:
        amount = float(amount_text)
    except ValueError:
        return None
    unit = unit_text.lower()
    if unit == "kb":
        return amount / 1024.0
    if unit == "mb":
        return amount
    if unit == "gb":
        return amount * 1024.0
    if unit == "tb":
        return amount * 1024.0 * 1024.0
    return None


def _format_memory_mib(memory_mib: float) -> str:
    """Formats one MiB value into a compact MB/GB string for one-line logs."""
    if memory_mib >= 1024.0:
        return f"{memory_mib / 1024.0:.2f}GB"
    return f"{memory_mib:.1f}MB"


def _is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    message = str(err).lower()
    return "requires optional dependencies" in message or "not installed" in message


def _encode_accurate_sequence(
    *,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
) -> EncodedSequence:
    """Encodes accurate audio sequence and maps dependency/transient failures."""
    try:
        return backend.encode_sequence(audio, sample_rate)
    except RuntimeError as err:
        if _is_dependency_error(err):
            raise AccurateRuntimeDependencyError(str(err)) from err
        raise AccurateTransientBackendError(str(err)) from err


def _prepare_accurate_backend_runtime(*, backend: FeatureBackend) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    prepare_runtime = getattr(backend, "prepare_runtime", None)
    if not callable(prepare_runtime):
        return
    try:
        prepare_runtime()
    except RuntimeError as err:
        if _is_dependency_error(err):
            raise AccurateRuntimeDependencyError(str(err)) from err
        raise AccurateTransientBackendError(str(err)) from err


def _ensure_accurate_compatible_model(
    loaded_model: LoadedModel,
    *,
    expected_backend_id: str,
    expected_profile: str,
    expected_backend_model_id: str | None,
) -> None:
    """Validates that loaded artifact metadata is compatible with accurate runtime."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise AccurateModelUnavailableError(
            "Accurate profile requires a v2 model artifact metadata envelope. "
            "Train an accurate-profile model before inference."
        )

    backend_id = metadata.get("backend_id")
    if backend_id != expected_backend_id:
        raise AccurateModelUnavailableError(
            "No accurate-profile model artifact is available. "
            f"Found backend_id={backend_id!r}; expected {expected_backend_id!r}."
        )

    profile = metadata.get("profile")
    if profile != expected_profile:
        raise AccurateModelUnavailableError(
            "No accurate-profile model artifact is available. "
            f"Found profile={profile!r}; expected {expected_profile!r}."
        )
    if expected_backend_model_id is not None:
        backend_model_id = metadata.get("backend_model_id")
        if (
            not isinstance(backend_model_id, str)
            or backend_model_id.strip() != expected_backend_model_id
        ):
            raise AccurateModelUnavailableError(
                "No accurate-profile model artifact is available. "
                f"Found backend_model_id={backend_model_id!r}; "
                f"expected {expected_backend_model_id!r}."
            )


def _warn_on_runtime_selector_mismatch(
    *,
    loaded_model: LoadedModel,
    backend_id: str,
    requested_device: str,
    requested_dtype: str,
    backend_override_device: str | None,
    backend_override_dtype: str | None,
    profile: str,
) -> None:
    """Warns when artifact runtime selectors differ from current runtime settings."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        return

    artifact_torch_device = metadata.get("torch_device")
    artifact_torch_dtype = metadata.get("torch_dtype")
    if not isinstance(artifact_torch_device, str) or not isinstance(
        artifact_torch_dtype, str
    ):
        return

    normalized_artifact_device = artifact_torch_device.strip().lower()
    normalized_artifact_dtype = artifact_torch_dtype.strip().lower()
    if not normalized_artifact_device or not normalized_artifact_dtype:
        return

    runtime_policy = resolve_feature_runtime_policy(
        backend_id=backend_id,
        requested_device=requested_device,
        requested_dtype=requested_dtype,
        backend_override_device=backend_override_device,
        backend_override_dtype=backend_override_dtype,
    )
    runtime_device = runtime_policy.device.strip().lower()
    runtime_dtype = runtime_policy.dtype.strip().lower()
    mismatch_components: list[str] = []
    if normalized_artifact_device != runtime_device:
        mismatch_components.append(
            f"device artifact={normalized_artifact_device!r} runtime={runtime_device!r}"
        )
    if normalized_artifact_dtype != runtime_dtype:
        mismatch_components.append(
            f"dtype artifact={normalized_artifact_dtype!r} runtime={runtime_dtype!r}"
        )
    if mismatch_components:
        logger.warning(
            "Artifact torch runtime selectors differ from current settings for %s "
            "profile (%s); embedding distribution may shift.",
            profile,
            ", ".join(mismatch_components),
        )


def _pooling_windows_from_encoded_frames(
    encoded: EncodedSequence,
    *,
    runtime_config: ProfileRuntimeConfig,
) -> list[PoolingWindow]:
    """Creates accurate temporal pooling windows from configured window policy."""
    return temporal_pooling_windows(
        encoded,
        window_size_seconds=runtime_config.pool_window_size_seconds,
        window_stride_seconds=runtime_config.pool_window_stride_seconds,
    )


def _predict_labels(model: object, features: FeatureMatrix) -> list[str]:
    """Runs model prediction and validates row-aligned label output."""
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise RuntimeError(
            "Loaded accurate model does not expose a callable predict()."
        )

    labels = np.asarray(predict(features), dtype=object)
    if labels.ndim != 1:
        raise RuntimeError(
            "Accurate model predict() returned invalid rank; expected 1D labels."
        )
    if int(labels.shape[0]) != int(features.shape[0]):
        raise RuntimeError(
            "Accurate model prediction row count mismatch. "
            f"Expected {int(features.shape[0])}, got {int(labels.shape[0])}."
        )
    return [str(item) for item in labels.tolist()]


def _confidence_and_probabilities(
    model: object,
    features: FeatureMatrix,
    *,
    expected_rows: int,
) -> tuple[list[float], list[dict[str, float] | None]]:
    """Returns per-frame confidence and optional class-probability mappings."""
    fallback_confidence = [1.0] * expected_rows
    fallback_probabilities: list[dict[str, float] | None] = [None] * expected_rows

    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        logger.warning(
            "Accurate model does not expose predict_proba; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    classes_attr = getattr(model, "classes_", None)
    class_values: list[object] | None = None
    if isinstance(classes_attr, np.ndarray):
        class_values = list(classes_attr.tolist())
    elif isinstance(classes_attr, list | tuple):
        class_values = list(classes_attr)
    if class_values is None:
        logger.warning(
            "Accurate model classes_ metadata is unavailable; using confidence fallback."
        )
        return fallback_confidence, fallback_probabilities

    try:
        raw_probabilities = np.asarray(predict_proba(features), dtype=np.float64)
    except Exception as err:
        logger.warning(
            "Accurate model predict_proba failed; using fallback. Error: %s", err
        )
        return fallback_confidence, fallback_probabilities

    if raw_probabilities.ndim != 2:
        logger.warning(
            "Accurate model predict_proba returned invalid rank %s; using fallback.",
            raw_probabilities.shape,
        )
        return fallback_confidence, fallback_probabilities
    if int(raw_probabilities.shape[0]) != expected_rows:
        logger.warning(
            "Accurate model predict_proba row mismatch (expected=%s, got=%s); using fallback.",
            expected_rows,
            int(raw_probabilities.shape[0]),
        )
        return fallback_confidence, fallback_probabilities

    class_labels = [str(item) for item in class_values]
    if int(raw_probabilities.shape[1]) != len(class_labels):
        logger.warning(
            "Accurate model predict_proba class mismatch (classes=%s, probs=%s); using fallback.",
            len(class_labels),
            int(raw_probabilities.shape[1]),
        )
        return fallback_confidence, fallback_probabilities

    confidences = [float(np.max(row)) for row in raw_probabilities]
    probabilities: list[dict[str, float] | None] = [
        {class_labels[idx]: float(row[idx]) for idx in range(len(class_labels))}
        for row in raw_probabilities
    ]
    return confidences, probabilities


def _segment_predictions(
    frame_predictions: list[FramePrediction],
    *,
    runtime_config: ProfileRuntimeConfig,
) -> list[SegmentPrediction]:
    """Builds stable segment predictions with smoothing/hysteresis/min-duration cleanup."""
    return postprocess_frame_predictions(
        frame_predictions,
        config=SegmentPostprocessingConfig(
            smoothing_window_frames=runtime_config.post_smoothing_window_frames,
            hysteresis_enter_confidence=runtime_config.post_hysteresis_enter_confidence,
            hysteresis_exit_confidence=runtime_config.post_hysteresis_exit_confidence,
            min_segment_duration_seconds=runtime_config.post_min_segment_duration_seconds,
        ),
    )
