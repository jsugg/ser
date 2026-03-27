"""Contracts for pytest suite bootstrap responsibilities and marker ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract

_ROOT_CONFTEST = Path("tests/conftest.py")
_ALLOWED_ROOT_MARKERS = frozenset(
    {
        "integration",
        "smoke",
        "topology_contract",
        "unit",
        "usefixtures",
    }
)
_IGNORED_MODULE_MARKERS = frozenset(
    {
        "filterwarnings",
        "integration",
        "smoke",
        "topology_contract",
        "unit",
        "usefixtures",
    }
)
_EXPECTED_SPECIAL_MARKERS = {
    "tests/suites/integration/test_process_isolation.py": {"process_isolation"},
}


def _extract_marker_names(node: ast.expr) -> set[str]:
    """Extract marker names from a supported `pytestmark` expression."""
    if isinstance(node, ast.Call):
        return _extract_marker_names(node.func)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
        if isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest":
            if node.value.attr == "mark":
                return {node.attr}
    if isinstance(node, ast.List | ast.Tuple):
        markers: set[str] = set()
        for element in node.elts:
            markers.update(_extract_marker_names(element))
        return markers
    raise AssertionError(f"Unsupported pytest marker expression: {ast.dump(node)}")


def _module_marker_names(path: Path) -> set[str]:
    """Return module-level pytest markers declared by one test module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "pytestmark":
            continue
        return _extract_marker_names(node.value)
    return set()


def _root_assigned_marker_names(path: Path) -> set[str]:
    """Return pytest marker names added dynamically by the root conftest."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    markers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_marker":
            continue
        if len(node.args) != 1:
            raise AssertionError("Root conftest marker injection must pass exactly one marker.")
        markers.update(_extract_marker_names(node.args[0]))
    return markers


def test_root_conftest_stays_lean(repo_root: Path) -> None:
    """Root suite bootstrap should stay small and structural."""
    lines = (repo_root / _ROOT_CONFTEST).read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 80, f"tests/conftest.py grew to {len(lines)} lines (max 80)."


def test_root_conftest_registers_fixture_plugin_only(repo_root: Path) -> None:
    """Root bootstrap should expose shared fixtures via pytest_plugins only."""
    tree = ast.parse((repo_root / _ROOT_CONFTEST).read_text(encoding="utf-8"))
    plugin_values: tuple[str, ...] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "pytest_plugins":
            continue
        if not isinstance(node.value, ast.Tuple):
            raise AssertionError("pytest_plugins must stay a tuple of plugin module strings.")
        plugin_values = tuple(
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
        break

    assert plugin_values == ("tests.fixtures.settings",)


def test_root_conftest_only_assigns_structural_markers(repo_root: Path) -> None:
    """Dynamic root marker injection should stay limited to suite semantics."""
    assigned = _root_assigned_marker_names(repo_root / _ROOT_CONFTEST)
    assert assigned == _ALLOWED_ROOT_MARKERS


def test_special_markers_are_explicitly_owned_by_modules(repo_root: Path) -> None:
    """Special-purpose pytest markers should stay declared by the owning module."""
    discovered: dict[str, set[str]] = {}
    for module_path in sorted((repo_root / "tests" / "suites").rglob("test_*.py")):
        marker_names = _module_marker_names(module_path) - _IGNORED_MODULE_MARKERS
        if marker_names:
            discovered[module_path.relative_to(repo_root).as_posix()] = marker_names

    assert discovered == _EXPECTED_SPECIAL_MARKERS
