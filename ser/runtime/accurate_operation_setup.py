"""Setup and compute adapters for accurate-profile inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from numpy.typing import NDArray

from ser.config import AppConfig, ProfileRuntimeConfig
from ser.runtime import accurate_worker_operation as accurate_worker_operation_helpers
from ser.runtime.accurate_worker_operation import PreparedAccurateOperation

_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


class _RequestLike(Protocol):
    """Minimal request contract required by accurate setup adapters."""

    @property
    def file_path(self) -> str: ...


class _PayloadLike(Protocol):
    """Serializable payload contract required by accurate setup adapters."""

    @property
    def request(self) -> _RequestLike: ...

    @property
    def settings(self) -> AppConfig: ...

    @property
    def expected_backend_id(self) -> str: ...

    @property
    def expected_profile(self) -> str: ...

    @property
    def expected_backend_model_id(self) -> str | None: ...


def prepare_in_process_operation(
    *,
    request: _RequestLike,
    settings: AppConfig,
    runtime_config: ProfileRuntimeConfig,
    loaded_model: _LoadedModelT | None,
    backend: _BackendT | None,
    load_accurate_model: Callable[[AppConfig], _LoadedModelT],
    validate_loaded_model: Callable[[_LoadedModelT], None],
    read_audio_file: Callable[[str], tuple[NDArray, int]],
    build_backend_for_profile: Callable[[AppConfig], _BackendT],
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedAccurateOperation[_LoadedModelT, _BackendT]:
    """Performs untimed setup for one in-process accurate operation."""
    return accurate_worker_operation_helpers.prepare_in_process_operation(
        request=request,
        settings=settings,
        runtime_config=runtime_config,
        loaded_model=loaded_model,
        backend=backend,
        load_accurate_model=load_accurate_model,
        validate_loaded_model=validate_loaded_model,
        read_audio_file=read_audio_file,
        build_backend_for_profile=build_backend_for_profile,
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def prepare_process_operation(
    payload: _PayloadLike,
    *,
    resolve_runtime_config: Callable[[AppConfig, str], ProfileRuntimeConfig],
    load_accurate_model: Callable[[_PayloadLike], _LoadedModelT],
    validate_loaded_model: Callable[[_LoadedModelT, _PayloadLike], None],
    read_audio_file: Callable[[str], tuple[NDArray, int]],
    build_backend_for_payload: Callable[[_PayloadLike], _BackendT],
    prepare_accurate_backend_runtime: Callable[[_BackendT], None],
    model_unavailable_error_factory: Callable[[str], Exception],
    model_load_error_factory: Callable[[str], Exception],
) -> PreparedAccurateOperation[_LoadedModelT, _BackendT]:
    """Performs untimed setup for one process-isolated accurate operation."""
    return accurate_worker_operation_helpers.prepare_process_operation(
        payload=payload,
        resolve_runtime_config=resolve_runtime_config,
        load_accurate_model=load_accurate_model,
        validate_loaded_model=validate_loaded_model,
        read_audio_file=read_audio_file,
        build_backend_for_payload=build_backend_for_payload,
        prepare_accurate_backend_runtime=prepare_accurate_backend_runtime,
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def run_process_operation(
    prepared: PreparedAccurateOperation[_LoadedModelT, _BackendT],
    *,
    run_accurate_inference_once: Callable[
        [_LoadedModelT, _BackendT, NDArray, int, ProfileRuntimeConfig],
        _ResultT,
    ],
) -> _ResultT:
    """Runs one isolated accurate compute phase from prepared worker state."""
    return accurate_worker_operation_helpers.run_process_operation(
        prepared,
        run_accurate_inference_once=run_accurate_inference_once,
    )


__all__ = [
    "prepare_in_process_operation",
    "prepare_process_operation",
    "run_process_operation",
]
