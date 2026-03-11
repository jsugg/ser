"""Medium-profile runtime support adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.repr import EncodedSequence, XLSRBackend
from ser.runtime import medium_worker_operation as medium_worker_operation_helpers
from ser.runtime.medium_backend_runtime import build_medium_backend as _build_medium_backend_impl
from ser.runtime.medium_backend_runtime import (
    settings_with_torch_runtime as _settings_with_torch_runtime_impl,
)
from ser.runtime.medium_backend_runtime import (
    warn_on_runtime_selector_mismatch as _warn_on_runtime_selector_mismatch_impl,
)


class LoadedModelLike(Protocol):
    """Structural contract for loaded model metadata access."""

    @property
    def artifact_metadata(self) -> dict[str, object] | None: ...


class SequenceEncoderBackend(Protocol):
    """Structural contract for backends that can encode one audio sequence."""

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence: ...


def build_medium_backend_for_settings(
    *,
    settings: AppConfig,
    expected_backend_model_id: str,
    runtime_device: str,
    runtime_dtype: str,
    backend_factory: Callable[..., XLSRBackend] = XLSRBackend,
) -> XLSRBackend:
    """Builds one XLS-R backend with explicit runtime selectors."""
    return _build_medium_backend_impl(
        settings=settings,
        expected_backend_model_id=expected_backend_model_id,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        backend_factory=backend_factory,
    )


def build_runtime_settings_snapshot(
    settings: AppConfig,
    *,
    runtime_device: str,
    runtime_dtype: str,
) -> AppConfig:
    """Build one process-safe settings snapshot with explicit runtime selectors."""
    return _settings_with_torch_runtime_impl(
        settings,
        device=runtime_device,
        dtype=runtime_dtype,
    )


def build_cpu_settings_snapshot(settings: AppConfig) -> AppConfig:
    """Build one settings snapshot pinned to CPU/float32 runtime selectors."""
    return build_runtime_settings_snapshot(
        settings,
        runtime_device="cpu",
        runtime_dtype="float32",
    )


def build_cpu_medium_backend_for_settings(
    *,
    settings: AppConfig,
    expected_backend_model_id: str,
    backend_factory: Callable[..., XLSRBackend] = XLSRBackend,
) -> XLSRBackend:
    """Build one CPU fallback XLS-R backend with normalized runtime selectors."""
    return build_medium_backend_for_settings(
        settings=build_cpu_settings_snapshot(settings),
        expected_backend_model_id=expected_backend_model_id,
        runtime_device="cpu",
        runtime_dtype="float32",
        backend_factory=backend_factory,
    )


def ensure_medium_loaded_model_compatibility(
    loaded_model: LoadedModelLike,
    *,
    expected_backend_model_id: str,
    unavailable_error_factory: Callable[[str], Exception],
) -> None:
    """Validates that medium artifact metadata matches the current backend id."""
    medium_worker_operation_helpers.ensure_medium_compatible_model(
        loaded_model,
        expected_backend_model_id=expected_backend_model_id,
        unavailable_error_factory=unavailable_error_factory,
    )


def warn_on_medium_runtime_selector_mismatch(
    *,
    loaded_model: LoadedModelLike,
    profile: str,
    runtime_device: str,
    runtime_dtype: str,
    logger: logging.Logger,
) -> None:
    """Warns when artifact runtime selectors differ from the active runtime."""
    _warn_on_runtime_selector_mismatch_impl(
        loaded_model=loaded_model,
        profile=profile,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        logger=logger,
    )


def validate_medium_loaded_model_runtime_contract(
    loaded_model: LoadedModelLike,
    *,
    expected_backend_model_id: str,
    profile: str,
    runtime_device: str,
    runtime_dtype: str,
    unavailable_error_factory: Callable[[str], Exception],
    logger: logging.Logger,
) -> None:
    """Validates medium artifact compatibility and runtime selector alignment."""
    ensure_medium_loaded_model_compatibility(
        loaded_model,
        expected_backend_model_id=expected_backend_model_id,
        unavailable_error_factory=unavailable_error_factory,
    )
    warn_on_medium_runtime_selector_mismatch(
        loaded_model=loaded_model,
        profile=profile,
        runtime_device=runtime_device,
        runtime_dtype=runtime_dtype,
        logger=logger,
    )


def encode_medium_sequence(
    *,
    backend: SequenceEncoderBackend,
    audio: NDArray[np.float32],
    sample_rate: int,
    is_dependency_error: Callable[[RuntimeError], bool],
    dependency_error_factory: Callable[[str], Exception],
    transient_error_factory: Callable[[str], Exception],
) -> EncodedSequence:
    """Encodes one medium audio sequence with runtime-domain error mapping."""
    try:
        return backend.encode_sequence(audio, sample_rate)
    except RuntimeError as err:
        if is_dependency_error(err):
            raise dependency_error_factory(str(err)) from err
        raise transient_error_factory(str(err)) from err


__all__ = [
    "build_cpu_medium_backend_for_settings",
    "build_cpu_settings_snapshot",
    "build_medium_backend_for_settings",
    "build_runtime_settings_snapshot",
    "encode_medium_sequence",
    "ensure_medium_loaded_model_compatibility",
    "validate_medium_loaded_model_runtime_contract",
    "warn_on_medium_runtime_selector_mismatch",
]
