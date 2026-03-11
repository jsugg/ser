"""Process-isolated setup and compute adapters for medium-profile inference."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig, MediumRuntimeConfig
from ser.runtime import medium_worker_operation as medium_worker_operation_helpers
from ser.runtime.medium_worker_operation import PreparedMediumOperation

_LoadedModelT = TypeVar("_LoadedModelT")
_BackendT = TypeVar("_BackendT")
_ResultT = TypeVar("_ResultT")


class _RequestLike(Protocol):
    """Minimal request contract required by isolated medium operations."""

    @property
    def file_path(self) -> str: ...


class _PayloadLike(Protocol):
    """Serializable payload contract required by isolated medium operations."""

    @property
    def request(self) -> _RequestLike: ...

    @property
    def settings(self) -> AppConfig: ...

    @property
    def expected_backend_model_id(self) -> str: ...


class _RuntimePolicyLike(Protocol):
    """Runtime policy selectors returned for the medium backend."""

    @property
    def device(self) -> str: ...

    @property
    def dtype(self) -> str: ...


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
    return medium_worker_operation_helpers.prepare_process_operation(
        payload=payload,
        load_medium_model=load_medium_model,
        ensure_medium_compatible_model=ensure_medium_compatible_model,
        resolve_runtime_policy=resolve_runtime_policy,
        warn_on_runtime_selector_mismatch=warn_on_runtime_selector_mismatch,
        read_audio_file=read_audio_file,
        build_medium_backend=build_medium_backend,
        prepare_medium_backend_runtime=prepare_medium_backend_runtime,
        model_unavailable_error_factory=model_unavailable_error_factory,
        model_load_error_factory=model_load_error_factory,
    )


def run_process_operation(
    prepared: PreparedMediumOperation[_LoadedModelT, _BackendT],
    *,
    run_medium_inference_once: Callable[
        [_LoadedModelT, _BackendT, NDArray[np.float32], int, MediumRuntimeConfig],
        _ResultT,
    ],
) -> _ResultT:
    """Runs one isolated medium compute phase from prepared worker state."""
    return medium_worker_operation_helpers.run_process_operation(
        prepared,
        run_medium_inference_once=run_medium_inference_once,
    )


__all__ = [
    "prepare_process_operation",
    "run_process_operation",
]
