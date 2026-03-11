"""Import-boundary contract tests for the public API facade."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWLIST_FILE = _REPO_ROOT / "tests" / "fixtures" / "public_internal_allowlist.txt"


def _explicit_allowlist_paths() -> set[Path]:
    """Loads the authoritative public-to-internal allowlist for contract enforcement."""
    return {
        (_REPO_ROOT / relative_path).resolve()
        for relative_path in _ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines()
        if relative_path and not relative_path.startswith("#")
    }


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


def test_public_to_internal_imports_match_explicit_allowlist() -> None:
    """Public modules importing `_internal` should match the authoritative allowlist."""
    package_root = Path("ser")
    import_pattern = re.compile(r"^\s*(?:from\s+ser\._internal|import\s+ser\._internal)", re.M)
    discovered_files = {
        source_path.resolve()
        for source_path in package_root.rglob("*.py")
        if "_internal" not in source_path.parts
        and import_pattern.search(source_path.read_text(encoding="utf-8"))
    }
    assert discovered_files == _explicit_allowlist_paths()


def test_cli_main_uses_internal_cli_support_modules_for_runtime_policy() -> None:
    """CLI runtime policy should flow through internal CLI support modules only."""
    cli_main_source = Path("ser/__main__.py").read_text(encoding="utf-8")

    required_imports = (
        "from ser._internal.cli.data import run_configure_command, run_data_command",
        "from ser._internal.cli.diagnostics import (",
        "from ser._internal.cli.runtime import (",
        "run_restricted_backend_cli_gate(",
        "run_startup_preflight_cli_gate(",
        "run_transcription_runtime_calibration_command(",
        "run_training_command(",
        "run_inference_command(",
    )
    for required_import in required_imports:
        assert required_import in cli_main_source

    blocked_policy_calls = (
        "from ser.api import (",
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
