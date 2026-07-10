# pyright: reportUnsupportedDunderAll=false
"""Public diagnostics facade."""

from __future__ import annotations

import importlib
from typing import Any

from ser.diagnostics.domain import (
    DiagnosticFinding,
    DiagnosticReport,
    DiagnosticSeverity,
    PreflightMode,
)

_DYNAMIC_EXPORTS: dict[str, str] = {
    "apply_profile_override_for_diagnostics": "ser._internal.diagnostics.service",
    "format_report_brief": "ser._internal.diagnostics.service",
    "format_report_json": "ser._internal.diagnostics.service",
    "format_report_text": "ser._internal.diagnostics.service",
    "format_startup_preflight_one_liner": "ser._internal.diagnostics.service",
    "parse_preflight_mode": "ser._internal.diagnostics.service",
    "run_doctor_command": "ser._internal.diagnostics.command",
    "run_doctor_diagnostics": "ser._internal.diagnostics.service",
    "should_fail_preflight": "ser._internal.diagnostics.service",
}

__all__ = [
    "DiagnosticFinding",
    "DiagnosticReport",
    "DiagnosticSeverity",
    "PreflightMode",
    *_DYNAMIC_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Lazily resolves one diagnostics facade export."""
    module_name = _DYNAMIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ser.diagnostics' has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
