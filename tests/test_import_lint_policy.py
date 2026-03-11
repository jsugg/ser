"""Contract tests for import-lint public boundary policy configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.topology_contract


def _load_ruff_lint_config() -> dict[str, Any]:
    """Loads Ruff lint configuration from pyproject."""
    pyproject_data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    tool_config = cast(dict[str, Any], pyproject_data["tool"])
    ruff_config = cast(dict[str, Any], tool_config["ruff"])
    return cast(dict[str, Any], ruff_config["lint"])


def test_import_lint_policy_covers_internal_api_modules() -> None:
    """Ruff banned-api policy should cover all internal API implementation modules."""
    lint_config = _load_ruff_lint_config()
    tidy_imports_config = cast(dict[str, Any], lint_config["flake8-tidy-imports"])
    banned_api = cast(dict[str, dict[str, str]], tidy_imports_config["banned-api"])

    required_modules = {
        "ser._internal.api",
        "ser._internal.api.data",
        "ser._internal.api.diagnostics",
        "ser._internal.api.runtime",
    }
    assert required_modules.issubset(set(banned_api))
    for module_name in required_modules:
        assert "ser.api" in banned_api[module_name]["msg"]


def test_import_lint_policy_limits_tid251_exceptions_to_boundary_files() -> None:
    """Only facade + API contract test files should bypass TID251 for internal imports."""
    lint_config = _load_ruff_lint_config()
    per_file_ignores = cast(dict[str, list[str]], lint_config["per-file-ignores"])

    allowed_tid251_files = {
        "ser/api.py",
        "ser/_internal/cli/data.py",
        "ser/_internal/cli/diagnostics.py",
        "ser/_internal/cli/runtime.py",
        "tests/test_api.py",
    }
    for file_path, ignored_rules in per_file_ignores.items():
        if "TID251" in ignored_rules:
            assert file_path in allowed_tid251_files
    for allowed_file in allowed_tid251_files:
        assert "TID251" in per_file_ignores[allowed_file]


def test_import_lint_lane_runs_boundary_contract_tests() -> None:
    """The import-lint lane should enforce both lint rules and boundary contracts."""
    script = Path("scripts/run_import_lint.sh").read_text(encoding="utf-8")

    assert "ruff check --select TID251" in script
    assert "tests/test_api_import_boundary.py" in script
    assert "tests/test_import_lint_policy.py" in script
