"""Adapter contracts for transcription backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ser.domain import TranscriptWord
from ser.profiles import TranscriptionBackendId

if TYPE_CHECKING:
    from ser.config import AppConfig


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """One compatibility issue discovered for a backend adapter."""

    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Compatibility findings for one backend/runtime request."""

    backend_id: TranscriptionBackendId
    functional_issues: tuple[CompatibilityIssue, ...] = ()
    operational_issues: tuple[CompatibilityIssue, ...] = ()
    noise_issues: tuple[CompatibilityIssue, ...] = ()
    policy_ids: tuple[str, ...] = ()

    @property
    def has_blocking_issues(self) -> bool:
        """Returns whether functional compatibility issues block execution."""
        return bool(self.functional_issues)


@dataclass(frozen=True, slots=True)
class BackendRuntimeRequest:
    """Transcription runtime request scoped to backend behavior."""

    model_name: str
    use_demucs: bool
    use_vad: bool


class TranscriptionBackendAdapter(Protocol):
    """Stable adapter boundary for transcription backend integrations."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns canonical backend identifier."""
        ...

    def check_compatibility(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> CompatibilityReport:
        """Returns compatibility report for one runtime request."""
        ...

    def setup_required(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> bool:
        """Returns whether setup/download must run before model load."""
        ...

    def prepare_assets(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> None:
        """Ensures backend assets exist before model loading."""
        ...

    def load_model(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> object:
        """Loads and returns one model handle."""
        ...

    def transcribe(
        self,
        *,
        model: object,
        runtime_request: BackendRuntimeRequest,
        file_path: str,
        language: str,
        settings: AppConfig,
    ) -> list[TranscriptWord]:
        """Runs transcription and returns word-level transcript rows."""
        ...
