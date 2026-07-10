"""Fast-profile public inference boundary."""

from __future__ import annotations

from ser._internal.runtime import fast_public_boundary as _boundary_support
from ser.config import AppConfig
from ser.models.emotion_model import LoadedModel
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import InferenceResult
from ser.utils.logger import get_logger

logger = get_logger(__name__)


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
    return _boundary_support.run_fast_inference_from_public_boundary(
        request,
        settings,
        loaded_model=loaded_model,
        enforce_timeout=enforce_timeout,
        allow_retries=allow_retries,
        logger=logger,
        model_unavailable_error_type=FastModelUnavailableError,
        model_load_error_type=FastModelLoadError,
        timeout_error_type=FastInferenceTimeoutError,
        execution_error_type=FastInferenceExecutionError,
        transient_error_type=FastTransientBackendError,
    )


__all__ = [
    "FastInferenceExecutionError",
    "FastInferenceTimeoutError",
    "FastModelLoadError",
    "FastModelUnavailableError",
    "FastTransientBackendError",
    "run_fast_inference",
]
