"""Report serialization and persistence helpers for profile quality gate."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Protocol, cast


class _ComparisonLike(Protocol):
    """Minimal comparison shape required for gate pass/fail enforcement."""

    @property
    def passes_quality_gate(self) -> bool:
        """Whether the profile comparison satisfies quality thresholds."""
        ...

    @property
    def failure_reasons(self) -> Sequence[str]:
        """Human-readable threshold failure reasons."""
        ...


class _ReportLike(Protocol):
    """Minimal report shape required for gate pass/fail enforcement."""

    @property
    def comparison(self) -> _ComparisonLike:
        """Cross-profile quality comparison payload."""
        ...


def build_report_payload(report: object) -> dict[str, object]:
    """Converts a dataclass report object into a JSON-safe dictionary."""
    if not is_dataclass(report) or isinstance(report, type):
        raise TypeError("report must be a dataclass instance")
    payload = asdict(report)
    if not isinstance(payload, dict):
        raise TypeError("dataclass conversion did not return a dictionary payload")
    return cast(dict[str, object], payload)


def serialize_report_payload(payload: Mapping[str, object]) -> str:
    """Serializes report payload with deterministic key order and indentation."""
    return json.dumps(payload, indent=2, sort_keys=True)


def resolve_report_output_path(
    *,
    output_path: str | None,
    default_directory: Path,
    default_file_name: str = "profile_quality_gate_report.json",
) -> Path:
    """Resolves report output location from explicit argument or default folder."""
    return (
        Path(output_path)
        if output_path is not None
        else default_directory / default_file_name
    )


def write_serialized_report(*, serialized: str, output_path: Path) -> None:
    """Persists serialized report with parent-directory creation and newline suffix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")


def enforce_quality_gate(
    report: _ReportLike,
    *,
    require_pass: bool,
) -> None:
    """Raises terminal error when pass enforcement is enabled and gate fails."""
    if not require_pass:
        return
    if report.comparison.passes_quality_gate:
        return
    reasons = "; ".join(str(reason) for reason in report.comparison.failure_reasons)
    raise SystemExit(f"Quality gate failed: {reasons}")
