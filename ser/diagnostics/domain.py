"""Domain models for SER environment diagnostics and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type DiagnosticSeverity = Literal["info", "warning", "error"]
type PreflightMode = Literal["off", "warn", "strict"]


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """Represents one actionable diagnostic finding."""

    code: str
    severity: DiagnosticSeverity
    message: str
    remediation: tuple[str, ...] = ()
    blocking: bool = False


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Aggregates findings produced by one diagnostics execution."""

    findings: tuple[DiagnosticFinding, ...]

    @property
    def has_blocking_findings(self) -> bool:
        """Returns whether any finding requires failing execution."""
        return any(finding.blocking for finding in self.findings)

    @property
    def has_warning_or_higher(self) -> bool:
        """Returns whether any warning or error finding exists."""
        return any(
            finding.severity in {"warning", "error"} for finding in self.findings
        )

    @property
    def has_error(self) -> bool:
        """Returns whether any error finding exists."""
        return any(finding.severity == "error" for finding in self.findings)

    def counts_by_severity(self) -> dict[DiagnosticSeverity, int]:
        """Returns one severity-count index for report summarization."""
        counts: dict[DiagnosticSeverity, int] = {"info": 0, "warning": 0, "error": 0}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def to_dict(self) -> dict[str, object]:
        """Returns one JSON-serializable report payload."""
        return {
            "summary": {
                "counts": self.counts_by_severity(),
                "has_blocking_findings": self.has_blocking_findings,
                "has_warning_or_higher": self.has_warning_or_higher,
                "has_error": self.has_error,
            },
            "findings": [
                {
                    "code": finding.code,
                    "severity": finding.severity,
                    "message": finding.message,
                    "blocking": finding.blocking,
                    "remediation": list(finding.remediation),
                }
                for finding in self.findings
            ],
        }
