"""Operation setup helpers for medium-profile inference paths."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, MediumRuntimeConfig

_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


class _RequestLike(Protocol):
    """Minimal request contract required by medium worker preparation."""

    @property
    def file_path(self) -> str: ...


class _PayloadLike(Protocol):
    """Serializable payload contract required by medium worker preparation."""

    @property
    def request(self) -> _RequestLike: ...

    @property
    def settings(self) -> AppConfig: ...

    @property
    def expected_backend_model_id(self) -> str: ...


_PayloadT = TypeVar("_PayloadT", bound=_PayloadLike)


class _RuntimePolicyLike(Protocol):
    """Runtime policy selectors used by medium worker preparation."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...


class _LoadedModelMetadataLike(Protocol):
    """Loaded model contract exposing artifact metadata for compatibility checks."""

    @property
    def artifact_metadata(self) -> dict[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class PreparedMediumOperation(Generic[_LoadedModelT, _BackendT]):
    """Setup-complete payload required to run one worker compute phase."""

    loaded_model: _LoadedModelT
    backend: _BackendT
    audio: NDArray[np.float32]
    sample_rate: int
    runtime_config: MediumRuntimeConfig


@dataclass(slots=True)
class MediumRetryOperationState(Generic[_PayloadT, _LoadedModelT, _BackendT]):
    """Mutable retry state for one medium inference request."""

    process_payload: _PayloadT | None = None
    prepared_operation: PreparedMediumOperation[_LoadedModelT, _BackendT] | None = None
    cpu_fallback_applied: bool = False


def prepare_retry_state(
    *,
    use_process_isolation: bool,
    request: _RequestLike,
    settings: AppConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    expected_backend_model_id: str,
    policy_device: str,
    policy_dtype: str,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    log_phase_started: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    build_process_payload: Callable[[], _PayloadT],
    prepare_in_process_operation: Callable[
        ...,
        PreparedMediumOperation[_LoadedModelT, _BackendT],
    ],
) -> tuple[
    MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT], float | None
]:
    """Builds initial retry state for medium runtime and performs untimed setup."""
    retry_state = MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT]()
    setup_started_at: float | None = None
    if use_process_isolation:
        retry_state.process_payload = build_process_payload()
        return retry_state, setup_started_at

    setup_started_at = log_phase_started(
        logger,
        phase_name=setup_phase_name,
        profile=profile,
    )
    try:
        retry_state.prepared_operation = prepare_in_process_operation(
            request=request,
            settings=settings,
            loaded_model=loaded_model,
            backend=backend,
            expected_backend_model_id=expected_backend_model_id,
            runtime_device=policy_device,
            runtime_dtype=policy_dtype,
        )
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    return retry_state, setup_started_at


def finalize_in_process_setup(
    *,
    use_process_isolation: bool,
    state: MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT],
    setup_started_at: float | None,
    logger: logging.Logger,
    profile: str,
    setup_phase_name: str,
    prepare_medium_backend_runtime: Callable[[_BackendT], None],
    log_phase_completed: Callable[..., float | None],
    log_phase_failed: Callable[..., float | None],
    runtime_error_factory: Callable[[str], Exception],
) -> None:
    """Prepares in-process backend runtime and emits setup phase diagnostics."""
    if use_process_isolation:
        return
    if state.prepared_operation is None:
        raise runtime_error_factory(
            "Medium inference operation prerequisites are missing."
        )
    try:
        prepare_medium_backend_runtime(state.prepared_operation.backend)
    except Exception:
        if setup_started_at is not None:
            log_phase_failed(
                logger,
                phase_name=setup_phase_name,
                started_at=setup_started_at,
                profile=profile,
            )
        raise
    if setup_started_at is not None:
        log_phase_completed(
            logger,
            phase_name=setup_phase_name,
            started_at=setup_started_at,
            profile=profile,
        )


def ensure_medium_compatible_model(
    loaded_model: _LoadedModelMetadataLike,
    *,
    expected_backend_model_id: str,
    unavailable_error_factory: Callable[[str], Exception],
) -> None:
    """Validates that loaded artifact metadata is compatible with medium runtime."""
    metadata = loaded_model.artifact_metadata
    if not isinstance(metadata, dict):
        raise unavailable_error_factory(
            "Medium profile requires a v2 model artifact metadata envelope. "
            "Train a medium-profile model before inference."
        )

    backend_id = metadata.get("backend_id")
    if backend_id != "hf_xlsr":
        raise unavailable_error_factory(
            "No medium-profile model artifact is available. "
            f"Found backend_id={backend_id!r}; expected 'hf_xlsr'."
        )

    profile = metadata.get("profile")
    if profile != "medium":
        raise unavailable_error_factory(
            "No medium-profile model artifact is available. "
            f"Found profile={profile!r}; expected 'medium'."
        )
    backend_model_id = metadata.get("backend_model_id")
    if (
        not isinstance(backend_model_id, str)
        or backend_model_id.strip() != expected_backend_model_id
    ):
        raise unavailable_error_factory(
            "No medium-profile model artifact is available. "
            f"Found backend_model_id={backend_model_id!r}; "
            f"expected {expected_backend_model_id!r}."
        )


def prepare_process_operation(
    payload: _PayloadLike,
    *,
    load_medium_model: Callable[[AppConfig, str], _LoadedModelT],
    ensure_medium_compatible_model: Callable[[_LoadedModelT, str], None],
    resolve_runtime_policy: Callable[[AppConfig], _RuntimePolicyLike],
    warn_on_runtime_selector_mismatch: Callable[[_LoadedModelT, str, str], None],
    read_audio_file: Callable[[str], tuple[NDArray[np.float32], int]],
    build_medium_backend: Callable[[AppConfig, str, str, str], _BackendT],
    prepare_medium_backend_runtime: Callable[[_BackendT], None],
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedMediumOperation[_LoadedModelT, _BackendT]:
    """Performs untimed setup for one process-isolated medium operation."""
    settings = payload.settings
    try:
        loaded_model = load_medium_model(settings, payload.expected_backend_model_id)
    except FileNotFoundError as err:
        raise model_unavailable_error_factory(str(err)) from err
    except ValueError as err:
        raise model_load_error_factory(
            "Failed to load medium-profile model artifact from configured paths."
        ) from err
    ensure_medium_compatible_model(loaded_model, payload.expected_backend_model_id)
    runtime_policy = resolve_runtime_policy(settings)
    warn_on_runtime_selector_mismatch(
        loaded_model,
        runtime_policy.device,
        runtime_policy.dtype,
    )
    audio, sample_rate = read_audio_file(payload.request.file_path)
    audio_array = np.asarray(audio, dtype=np.float32)
    backend = build_medium_backend(
        settings,
        payload.expected_backend_model_id,
        runtime_policy.device,
        runtime_policy.dtype,
    )
    prepare_medium_backend_runtime(backend)
    return PreparedMediumOperation(
        loaded_model=loaded_model,
        backend=backend,
        audio=audio_array,
        sample_rate=sample_rate,
        runtime_config=settings.medium_runtime,
    )


def prepare_in_process_operation(
    *,
    request: _RequestLike,
    settings: AppConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    load_medium_model: Callable[[AppConfig, str], _LoadedModelT],
    ensure_medium_compatible_model: Callable[[_LoadedModelT, str], None],
    warn_on_runtime_selector_mismatch: Callable[[_LoadedModelT, str, str], None],
    read_audio_file: Callable[[str], tuple[NDArray[np.float32], int]],
    build_medium_backend: Callable[[AppConfig, str, str, str], _BackendT],
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedMediumOperation[_LoadedModelT, _BackendT]:
    """Performs untimed setup for one in-process medium operation."""
    active_loaded_model: _LoadedModelT
    if loaded_model is None:
        try:
            active_loaded_model = load_medium_model(
                settings,
                expected_backend_model_id,
            )
        except FileNotFoundError as err:
            raise model_unavailable_error_factory(str(err)) from err
        except ValueError as err:
            raise model_load_error_factory(
                "Failed to load medium-profile model artifact from configured paths."
            ) from err
    else:
        active_loaded_model = loaded_model

    ensure_medium_compatible_model(active_loaded_model, expected_backend_model_id)
    warn_on_runtime_selector_mismatch(
        active_loaded_model,
        runtime_device,
        runtime_dtype,
    )
    audio, sample_rate = read_audio_file(request.file_path)
    audio_array = np.asarray(audio, dtype=np.float32)
    active_backend = (
        backend
        if backend is not None
        else build_medium_backend(
            settings,
            expected_backend_model_id,
            runtime_device,
            runtime_dtype,
        )
    )
    return PreparedMediumOperation(
        loaded_model=active_loaded_model,
        backend=active_backend,
        audio=audio_array,
        sample_rate=sample_rate,
        runtime_config=settings.medium_runtime,
    )


def run_inference_operation(
    *,
    enforce_timeout: bool,
    use_process_isolation: bool,
    process_payload: _PayloadT | None,
    prepared_operation: PreparedMediumOperation[_LoadedModelT, _BackendT] | None,
    timeout_seconds: float,
    logger: logging.Logger,
    profile: str,
    inference_phase_name: str,
    log_phase_started: Callable[..., float],
    log_phase_completed: Callable[..., float],
    log_phase_failed: Callable[..., float],
    run_with_process_timeout: Callable[[_PayloadT, float], _ResultT],
    run_process_operation: Callable[
        [PreparedMediumOperation[_LoadedModelT, _BackendT]],
        _ResultT,
    ],
    run_with_timeout: Callable[[Callable[[], _ResultT], float], _ResultT],
    runtime_error_factory: Callable[[str], Exception],
) -> _ResultT:
    """Runs one medium inference attempt with phase/timing orchestration."""
    if enforce_timeout:
        if use_process_isolation:
            if process_payload is None:
                raise runtime_error_factory(
                    "Medium process payload is missing for isolated execution."
                )
            return run_with_process_timeout(process_payload, timeout_seconds)
        inference_started_at = log_phase_started(
            logger,
            phase_name=inference_phase_name,
            profile=profile,
        )

        def timeout_operation() -> _ResultT:
            if prepared_operation is None:
                raise runtime_error_factory(
                    "Medium inference operation prerequisites are missing."
                )
            return run_process_operation(prepared_operation)

        try:
            result = run_with_timeout(timeout_operation, timeout_seconds)
        except Exception:
            log_phase_failed(
                logger,
                phase_name=inference_phase_name,
                started_at=inference_started_at,
                profile=profile,
            )
            raise
        log_phase_completed(
            logger,
            phase_name=inference_phase_name,
            started_at=inference_started_at,
            profile=profile,
        )
        return result
    if prepared_operation is None:
        raise runtime_error_factory(
            "Medium inference operation prerequisites are missing."
        )
    inference_started_at = log_phase_started(
        logger,
        phase_name=inference_phase_name,
        profile=profile,
    )
    try:
        result = run_process_operation(prepared_operation)
    except Exception:
        log_phase_failed(
            logger,
            phase_name=inference_phase_name,
            started_at=inference_started_at,
            profile=profile,
        )
        raise
    log_phase_completed(
        logger,
        phase_name=inference_phase_name,
        started_at=inference_started_at,
        profile=profile,
    )
    return result


def run_process_operation(
    prepared: PreparedMediumOperation[_LoadedModelT, _BackendT],
    *,
    run_medium_inference_once: Callable[
        [_LoadedModelT, _BackendT, NDArray[np.float32], int, MediumRuntimeConfig],
        _ResultT,
    ],
) -> _ResultT:
    """Runs one medium compute phase inside isolated worker process."""
    return run_medium_inference_once(
        prepared.loaded_model,
        prepared.backend,
        prepared.audio,
        prepared.sample_rate,
        prepared.runtime_config,
    )


def build_transient_failure_handler(
    *,
    state: MediumRetryOperationState[_PayloadT, _LoadedModelT, _BackendT],
    use_process_isolation: bool,
    injected_backend: _BackendT | None,
    policy_device: str,
    logger: logging.Logger,
    should_retry_on_cpu_after_transient_failure: Callable[[Exception], bool],
    summarize_transient_failure: Callable[[Exception], str],
    process_payload_cpu_fallback: Callable[[_PayloadT], _PayloadT],
    in_process_cpu_backend_builder: Callable[[], _BackendT],
    prepare_medium_backend_runtime: Callable[[_BackendT], None],
    replace_prepared_backend: Callable[
        [PreparedMediumOperation[_LoadedModelT, _BackendT], _BackendT],
        PreparedMediumOperation[_LoadedModelT, _BackendT],
    ],
    runtime_error_factory: Callable[[str], Exception],
) -> Callable[[Exception, int, int], None]:
    """Builds retry callback that applies one-time CPU fallback for transient MPS errors."""

    def on_transient_failure(
        err: Exception,
        _attempt: int,
        _transient_failures: int,
    ) -> None:
        if state.cpu_fallback_applied:
            return
        if not should_retry_on_cpu_after_transient_failure(err):
            return
        if not use_process_isolation and injected_backend is not None:
            logger.info(
                "Medium inference detected MPS transient failure but cannot "
                "switch to CPU when a custom backend instance is injected."
            )
            return

        current_device = (
            state.process_payload.settings.torch_runtime.device
            if use_process_isolation and state.process_payload is not None
            else policy_device
        )
        if current_device.strip().lower() not in {"auto", "mps"}:
            return
        state.cpu_fallback_applied = True
        logger.info(
            "Medium inference will retry on CPU after transient MPS failure: %s",
            summarize_transient_failure(err),
        )
        try:
            if use_process_isolation:
                if state.process_payload is None:
                    raise runtime_error_factory(
                        "Medium process payload is missing for CPU fallback."
                    )
                state.process_payload = process_payload_cpu_fallback(
                    state.process_payload
                )
                return
            active_backend = in_process_cpu_backend_builder()
            prepare_medium_backend_runtime(active_backend)
            if state.prepared_operation is None:
                raise runtime_error_factory(
                    "Medium inference operation prerequisites are missing."
                )
            state.prepared_operation = replace_prepared_backend(
                state.prepared_operation,
                active_backend,
            )
        except Exception as swap_err:
            state.cpu_fallback_applied = False
            logger.warning(
                "Medium inference CPU fallback setup failed; continuing with "
                "existing retry policy: %s",
                swap_err,
            )

    return on_transient_failure


__all__ = [
    "MediumRetryOperationState",
    "PreparedMediumOperation",
    "build_transient_failure_handler",
    "ensure_medium_compatible_model",
    "finalize_in_process_setup",
    "prepare_in_process_operation",
    "prepare_process_operation",
    "prepare_retry_state",
    "run_inference_operation",
    "run_process_operation",
]
