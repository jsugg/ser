"""Import-boundary contract tests for the public API facade."""

from __future__ import annotations

import importlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract


@dataclass(frozen=True)
class _BoundaryPolicyEntry:
    """One allowed public-to-internal dependency entry from boundary policy."""

    relative_path: str
    reason: str


def _load_boundary_policy_entries(repo_root: Path) -> tuple[_BoundaryPolicyEntry, ...]:
    """Loads the authoritative public-to-internal boundary policy entries."""
    policy_data = tomllib.loads((repo_root / "boundary_policy.toml").read_text(encoding="utf-8"))
    raw_entries = policy_data.get("public_internal_import")
    if not isinstance(raw_entries, list):
        raise AssertionError("boundary_policy.toml must define [[public_internal_import]] entries.")

    entries: list[_BoundaryPolicyEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must be a table, got {type(raw_entry)!r}."
            )
        raw_path = raw_entry.get("path")
        raw_reason = raw_entry.get("reason")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must define a non-empty path."
            )
        if not isinstance(raw_reason, str) or not raw_reason.strip():
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must define a non-empty reason."
            )
        entries.append(
            _BoundaryPolicyEntry(
                relative_path=raw_path.strip(),
                reason=raw_reason.strip(),
            )
        )
    return tuple(entries)


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    """Returns one repository-relative path resolved from the root fixture."""
    return (repo_root / relative_path).resolve()


def _explicit_allowlist_paths(repo_root: Path) -> set[Path]:
    """Loads the authoritative public-to-internal allowlist for contract enforcement."""
    return {
        _resolve_repo_path(repo_root, entry.relative_path)
        for entry in _load_boundary_policy_entries(repo_root)
    }


def test_boundary_policy_file_is_well_formed(repo_root: Path) -> None:
    """Boundary policy entries should be sorted, unique, and point to public modules."""
    entries = _load_boundary_policy_entries(repo_root)
    assert entries, "boundary_policy.toml must declare at least one allowed public module."

    relative_paths = [entry.relative_path for entry in entries]
    assert relative_paths == sorted(
        relative_paths
    ), "boundary_policy.toml entries must stay sorted by path for reviewability."
    assert len(set(relative_paths)) == len(
        relative_paths
    ), "boundary_policy.toml must not contain duplicate paths."
    for entry in entries:
        assert entry.relative_path.startswith("ser/")
        assert entry.relative_path.endswith(".py")
        assert "_internal" not in Path(entry.relative_path).parts
        assert _resolve_repo_path(
            repo_root, entry.relative_path
        ).exists(), f"Boundary policy path {entry.relative_path!r} does not exist."
        assert entry.reason


def test_no_first_party_module_imports_internal_api_directly(repo_root: Path) -> None:
    """All first-party imports should consume API callables through `ser.api`."""
    package_root = repo_root / "ser"
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


def test_public_to_internal_imports_match_explicit_allowlist(repo_root: Path) -> None:
    """Public modules importing `_internal` should match the authoritative allowlist."""
    package_root = repo_root / "ser"
    import_pattern = re.compile(r"^\s*(?:from\s+ser\._internal|import\s+ser\._internal)", re.M)
    discovered_files = {
        source_path.resolve()
        for source_path in package_root.rglob("*.py")
        if "_internal" not in source_path.parts
        and import_pattern.search(source_path.read_text(encoding="utf-8"))
    }
    assert discovered_files == _explicit_allowlist_paths(repo_root)


def test_cli_main_uses_internal_cli_support_modules_for_runtime_policy(repo_root: Path) -> None:
    """CLI runtime policy should flow through internal CLI support modules only."""
    cli_main_source = (repo_root / "ser" / "__main__.py").read_text(encoding="utf-8")

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
