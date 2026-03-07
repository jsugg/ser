"""Import-boundary contract tests for the public API facade."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest


def test_no_first_party_module_imports_internal_api_directly() -> None:
    """All first-party imports should consume API callables through `ser.api`."""
    package_root = Path("ser")
    blocked_patterns = (
        re.compile(r"^\s*from\s+ser\._internal\.api(?:\.|\s+)"),
        re.compile(r"^\s*import\s+ser\._internal\.api(?:\.|\s+|$)"),
    )
    allowed_files = {
        (package_root / "api.py").resolve(),
    }
    for source_path in package_root.rglob("*.py"):
        resolved_path = source_path.resolve()
        if resolved_path in allowed_files:
            continue
        if "_internal" in source_path.parts:
            continue
        source_text = source_path.read_text(encoding="utf-8")
        if any(pattern.search(source_text) for pattern in blocked_patterns):
            raise AssertionError(
                f"Internal API import boundary violation in {source_path}. "
                "Use `ser.api` for public imports."
            )


def test_cli_main_uses_api_command_wrappers_for_runtime_policy() -> None:
    """CLI runtime policy should flow through stable API command wrappers only."""
    cli_main_source = Path("ser/__main__.py").read_text(encoding="utf-8")

    required_wrappers = (
        "run_restricted_backend_cli_gate(",
        "run_startup_preflight_cli_gate(",
        "run_transcription_runtime_calibration_command(",
        "run_training_command(",
        "run_inference_command(",
    )
    for wrapper_call in required_wrappers:
        assert wrapper_call in cli_main_source

    blocked_policy_calls = (
        "prepare_restricted_backend_opt_in_state(",
        "enforce_restricted_backends_for_cli(",
        "run_training_workflow(",
        "run_inference_workflow(",
        "classify_training_exception(",
        "classify_inference_exception(",
    )
    for blocked_call in blocked_policy_calls:
        assert blocked_call not in cli_main_source


@pytest.mark.parametrize(
    "module_name",
    ("ser.api_data", "ser.api_runtime", "ser.api_diagnostics"),
)
def test_legacy_api_implementation_modules_are_not_publicly_importable(
    module_name: str,
) -> None:
    """Legacy top-level API implementation modules should stay removed."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
