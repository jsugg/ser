"""Compatibility orchestration helpers for transcription backends."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

from ser.config import AppConfig
from ser.profiles import TranscriptionBackendId
from ser.transcript.backends import (
    BackendRuntimeRequest,
    CompatibilityIssueImpact,
    CompatibilityReport,
)

type _CompatibilityIssueKind = Literal["noise", "operational"]
type _EmittedIssueKeySet = set[tuple[str, str, str]]
_EMITTED_COMPATIBILITY_ISSUE_KEYS: _EmittedIssueKeySet = set()


class _CompatibilityAdapter(Protocol):
    """Minimal compatibility adapter contract used during orchestration."""

    def check_compatibility(
        self,
        *,
        runtime_request: BackendRuntimeRequest,
        settings: AppConfig,
    ) -> CompatibilityReport:
        """Returns compatibility report for one runtime request."""
        ...


class _BackendProfile(Protocol):
    """Minimal profile contract for compatibility orchestration."""

    @property
    def backend_id(self) -> TranscriptionBackendId:
        """Returns backend identifier for compatibility resolution."""
        ...


_TProfile = TypeVar("_TProfile", bound=_BackendProfile)
type _AdapterResolver = Callable[[TranscriptionBackendId], _CompatibilityAdapter]
type _ErrorFactory = Callable[[str], Exception]


def _resolve_emitted_issue_keys(
    emitted_issue_keys: _EmittedIssueKeySet | None,
) -> _EmittedIssueKeySet:
    """Returns the shared compatibility registry when no explicit registry is provided."""
    return _EMITTED_COMPATIBILITY_ISSUE_KEYS if emitted_issue_keys is None else emitted_issue_keys


def _summarize_operational_issue_message(issue_message: str, *, max_chars: int) -> str:
    """Returns one concise operational issue message for CLI log hygiene."""
    normalized = " ".join(issue_message.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _log_compatibility_issue_once(
    *,
    backend_id: TranscriptionBackendId,
    issue_kind: _CompatibilityIssueKind,
    issue_code: str,
    issue_message: str,
    issue_impact: CompatibilityIssueImpact,
    emitted_issue_keys: _EmittedIssueKeySet,
    logger: logging.Logger,
) -> None:
    """Logs one non-blocking compatibility issue once per backend/kind/code tuple."""
    issue_key = (backend_id, issue_kind, issue_code)
    if issue_key in emitted_issue_keys:
        return
    emitted_issue_keys.add(issue_key)
    if issue_kind == "noise":
        logger.debug(
            "Transcription backend '%s' noise issue [%s]: %s",
            backend_id,
            issue_code,
            issue_message,
        )
        return
    log_level = logging.INFO if issue_impact == "informational" else logging.WARNING
    logger.log(
        log_level,
        "Transcription backend '%s' non-blocking operational issue [%s]: %s "
        "Run `ser doctor` for full remediation details.",
        backend_id,
        issue_code,
        _summarize_operational_issue_message(issue_message, max_chars=180),
    )


def mark_compatibility_issues_as_emitted(
    *,
    backend_id: TranscriptionBackendId,
    issue_kind: _CompatibilityIssueKind,
    issue_codes: tuple[str, ...],
    emitted_issue_keys: _EmittedIssueKeySet | None = None,
) -> None:
    """Marks compatibility issues as already emitted to prevent duplicate logs."""
    active_emitted_issue_keys = _resolve_emitted_issue_keys(emitted_issue_keys)
    for issue_code in issue_codes:
        if not issue_code:
            continue
        active_emitted_issue_keys.add((backend_id, issue_kind, issue_code))


def check_adapter_compatibility(
    *,
    active_profile: _TProfile,
    settings: AppConfig,
    runtime_request: BackendRuntimeRequest | None,
    runtime_request_resolver: Callable[[_TProfile, AppConfig], BackendRuntimeRequest],
    adapter_resolver: _AdapterResolver,
    error_factory: _ErrorFactory,
    emitted_issue_keys: _EmittedIssueKeySet | None = None,
    logger: logging.Logger,
) -> CompatibilityReport:
    """Validates backend compatibility and logs non-blocking issues once."""
    active_emitted_issue_keys = _resolve_emitted_issue_keys(emitted_issue_keys)
    backend_id = active_profile.backend_id
    adapter = adapter_resolver(backend_id)
    resolved_runtime_request = (
        runtime_request_resolver(active_profile, settings)
        if runtime_request is None
        else runtime_request
    )
    report = adapter.check_compatibility(
        runtime_request=resolved_runtime_request,
        settings=settings,
    )
    for issue in report.noise_issues:
        _log_compatibility_issue_once(
            backend_id=backend_id,
            issue_kind="noise",
            issue_code=issue.code,
            issue_message=issue.message,
            issue_impact=issue.impact,
            emitted_issue_keys=active_emitted_issue_keys,
            logger=logger,
        )
    for issue in report.operational_issues:
        _log_compatibility_issue_once(
            backend_id=backend_id,
            issue_kind="operational",
            issue_code=issue.code,
            issue_message=issue.message,
            issue_impact=issue.impact,
            emitted_issue_keys=active_emitted_issue_keys,
            logger=logger,
        )
    if report.has_blocking_issues:
        details = (
            "; ".join(issue.message for issue in report.functional_issues)
            or "backend compatibility validation failed"
        )
        raise error_factory(details)
    return report


__all__ = [
    "check_adapter_compatibility",
    "mark_compatibility_issues_as_emitted",
]
