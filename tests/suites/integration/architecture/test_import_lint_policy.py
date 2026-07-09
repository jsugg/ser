"""Contract tests for import-lint public boundary policy configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.topology_contract


def _load_ruff_lint_config(repo_root: Path) -> dict[str, Any]:
    """Loads Ruff lint configuration from pyproject."""
    pyproject_data = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    tool_config = cast(dict[str, Any], pyproject_data["tool"])
    ruff_config = cast(dict[str, Any], tool_config["ruff"])
    return cast(dict[str, Any], ruff_config["lint"])


def _load_boundary_policy_paths(repo_root: Path) -> set[str]:
    """Loads public facade paths from the authoritative boundary policy."""
    policy_data = tomllib.loads((repo_root / "boundary_policy.toml").read_text(encoding="utf-8"))
    entries = policy_data["public_internal_import"]
    if not isinstance(entries, list):
        raise AssertionError("boundary_policy.toml must define public_internal_import entries.")
    paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise AssertionError("boundary_policy.toml entries must be tables.")
        path = entry.get("path")
        if not isinstance(path, str):
            raise AssertionError("boundary_policy.toml entries must define string paths.")
        paths.add(path)
    return paths


def test_import_lint_policy_covers_all_internal_modules(repo_root: Path) -> None:
    """Ruff banned-api policy should cover every public import of `ser._internal`."""
    lint_config = _load_ruff_lint_config(repo_root)
    tidy_imports_config = cast(dict[str, Any], lint_config["flake8-tidy-imports"])
    banned_api = cast(dict[str, dict[str, str]], tidy_imports_config["banned-api"])

    assert set(banned_api) == {"ser._internal"}
    assert "boundary_policy.toml" in banned_api["ser._internal"]["msg"]


def test_import_lint_policy_limits_tid251_exceptions_to_boundary_files(repo_root: Path) -> None:
    """Only boundary-policy facade files should bypass TID251."""
    lint_config = _load_ruff_lint_config(repo_root)
    per_file_ignores = cast(dict[str, list[str]], lint_config["per-file-ignores"])

    tid251_ignored_files = {
        file_path
        for file_path, ignored_rules in per_file_ignores.items()
        if "TID251" in ignored_rules
    }
    assert tid251_ignored_files == _load_boundary_policy_paths(repo_root)


def test_import_lint_lane_runs_boundary_contract_tests(repo_root: Path) -> None:
    """The import-lint lane should enforce both lint rules and boundary contracts."""
    script = (repo_root / "scripts" / "run_import_lint.sh").read_text(encoding="utf-8")

    assert "find ser -path 'ser/_internal' -prune -o -name '*.py' -print | sort" in script
    assert "ruff check --select TID251" in script
    assert "tests/suites/integration/architecture/test_api_import_boundary.py" in script
    assert "tests/suites/integration/architecture/test_import_lint_policy.py" in script
