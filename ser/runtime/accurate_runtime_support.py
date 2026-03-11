"""Accurate-profile runtime support adapters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.repr import EncodedSequence, FeatureBackend
from ser.runtime.accurate_backend_runtime import (
    prepare_accurate_backend_runtime as _prepare_accurate_backend_runtime_impl,
)
from ser.runtime.accurate_backend_runtime import (
    settings_with_torch_device as _settings_with_torch_device_impl,
)
from ser.runtime.retry_primitives import (
    is_optional_dependency_runtime_error as _is_optional_dependency_runtime_error_impl,
)


def build_process_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Builds a process-safe settings snapshot for spawn-based workers."""
    return _settings_snapshot_for_device(
        settings,
        device=settings.torch_runtime.device,
    )


def build_cpu_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Builds one settings snapshot pinned to CPU torch selectors."""
    return _settings_snapshot_for_device(
        settings,
        device="cpu",
    )


def encode_accurate_sequence(
    *,
    backend: FeatureBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> EncodedSequence:
    """Encodes accurate audio sequence and maps dependency/transient failures."""
    try:
        return backend.encode_sequence(audio, sample_rate)
    except RuntimeError as err:
        if _is_dependency_error(err):
            raise dependency_error_factory(str(err)) from err
        raise transient_error_factory(str(err)) from err


def prepare_accurate_backend_runtime(
    backend: FeatureBackend,
    *,
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> None:
    """Warms backend runtime components so setup work is outside timeout budgets."""
    _prepare_accurate_backend_runtime_impl(
        backend=backend,
        is_dependency_error=_is_dependency_error,
        dependency_error_factory=dependency_error_factory,
        transient_error_factory=transient_error_factory,
    )


def _is_dependency_error(err: RuntimeError) -> bool:
    """Returns whether runtime error indicates missing optional modules."""
    return _is_optional_dependency_runtime_error_impl(err)


def _settings_snapshot_for_device(
    settings: AppConfig,
    *,
    device: str,
) -> AppConfig:
    """Clones settings while pinning torch runtime to one device selector."""
    return _settings_with_torch_device_impl(
        settings,
        device=device,
    )


__all__ = [
    "build_cpu_settings_snapshot",
    "build_process_settings_snapshot",
    "encode_accurate_sequence",
    "prepare_accurate_backend_runtime",
]
