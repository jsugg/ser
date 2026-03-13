"""Medium-profile public inference boundary."""

from __future__ import annotations

from ser._internal.runtime import medium_public_boundary as _boundary_support
from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel
from ser.repr import XLSRBackend
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import InferenceResult
from ser.utils.logger import get_logger

logger = get_logger(__name__)


class MediumModelUnavailableError(FileNotFoundError):
    """Raised when a compatible medium-profile model artifact is unavailable."""


class MediumRuntimeDependencyError(RuntimeError):
    """Raised when medium optional runtime dependencies are missing."""


class MediumModelLoadError(RuntimeError):
    """Raised when medium model artifact loading fails unexpectedly."""


class MediumInferenceTimeoutError(TimeoutError):
    """Raised when medium inference exceeds configured timeout budget."""


class MediumInferenceExecutionError(RuntimeError):
    """Raised when medium inference exhausts retries without recovery."""


class MediumTransientBackendError(RuntimeError):
    """Raised for retryable medium backend encoding failures."""


def run_medium_inference(
    request: InferenceRequest,
    settings: AppConfig,
    *,
    loaded_model: LoadedModel | None = None,
    backend: XLSRBackend | None = None,
    enforce_timeout: bool = True,
    allow_retries: bool = True,
) -> InferenceResult:
    """Runs medium-profile inference with bounded retries and timeout budget."""
    return _boundary_support.run_medium_inference_from_public_boundary(
        request,
        settings,
        loaded_model=loaded_model,
        backend=backend,
        enforce_timeout=enforce_timeout,
        allow_retries=allow_retries,
        logger=logger,
        model_unavailable_error_type=MediumModelUnavailableError,
        runtime_dependency_error_type=MediumRuntimeDependencyError,
        model_load_error_type=MediumModelLoadError,
        timeout_error_type=MediumInferenceTimeoutError,
        execution_error_type=MediumInferenceExecutionError,
        transient_error_type=MediumTransientBackendError,
    )


__all__ = [
    "MediumInferenceExecutionError",
    "MediumInferenceTimeoutError",
    "MediumModelLoadError",
    "MediumModelUnavailableError",
    "MediumRuntimeDependencyError",
    "MediumTransientBackendError",
    "run_medium_inference",
]
