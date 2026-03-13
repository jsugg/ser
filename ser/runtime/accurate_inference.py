"""Accurate-profile public inference boundary."""

from __future__ import annotations

from ser._internal.runtime import accurate_public_boundary as _boundary_support
from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel
from ser.repr import FeatureBackend
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import InferenceResult
from ser.utils.logger import get_logger

logger = get_logger(__name__)


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
    """Runs accurate-profile inference with bounded retries and timeout budget."""
    return _boundary_support.run_accurate_inference_from_public_boundary(
        request,
        settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        allow_retries=allow_retries,
        expected_backend_id=expected_backend_id,
        expected_profile=expected_profile,
        expected_backend_model_id=expected_backend_model_id,
        logger=logger,
        model_unavailable_error_type=AccurateModelUnavailableError,
        runtime_dependency_error_type=AccurateRuntimeDependencyError,
        model_load_error_type=AccurateModelLoadError,
        timeout_error_type=AccurateInferenceTimeoutError,
        inference_execution_error_type=AccurateInferenceExecutionError,
        transient_backend_error_type=AccurateTransientBackendError,
    )


__all__ = [
    "AccurateInferenceExecutionError",
    "AccurateInferenceTimeoutError",
    "AccurateModelLoadError",
    "AccurateModelUnavailableError",
    "AccurateRuntimeDependencyError",
    "AccurateTransientBackendError",
    "run_accurate_inference",
]
