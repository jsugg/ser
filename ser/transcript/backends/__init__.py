"""Transcription backend adapters."""

from .base import (
    BackendRuntimeRequest,
    CompatibilityIssue,
    CompatibilityReport,
    TranscriptionBackendAdapter,
)
from .factory import resolve_transcription_backend_adapter

__all__ = [
    "BackendRuntimeRequest",
    "CompatibilityIssue",
    "CompatibilityReport",
    "TranscriptionBackendAdapter",
    "resolve_transcription_backend_adapter",
]
