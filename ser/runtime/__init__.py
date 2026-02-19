"""Runtime helpers for profiling, orchestration, and benchmarks."""

from .backend_hooks import build_backend_hooks
from .contracts import InferenceExecution, InferenceRequest
from .pipeline import RuntimePipeline, create_runtime_pipeline
from .registry import (
    RuntimeCapability,
    UnsupportedProfileError,
    ensure_profile_supported,
    resolve_runtime_capability,
)

__all__ = [
    "InferenceExecution",
    "InferenceRequest",
    "RuntimeCapability",
    "RuntimePipeline",
    "UnsupportedProfileError",
    "build_backend_hooks",
    "create_runtime_pipeline",
    "ensure_profile_supported",
    "resolve_runtime_capability",
]
