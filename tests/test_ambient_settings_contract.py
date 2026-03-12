"""Architecture contract tests for source-level settings access."""

from __future__ import annotations

import ast
from pathlib import Path

type GetSettingsCallSite = tuple[str, str]

EXPECTED_GET_SETTINGS_CALL_SITES: set[GetSettingsCallSite] = set()


class _GetSettingsCallVisitor(ast.NodeVisitor):
    """Collects every `get_settings()` call site with its enclosing qualname."""

    def __init__(self, relative_path: str) -> None:
        self._relative_path = relative_path
        self._stack: list[str] = []
        self.call_sites: set[GetSettingsCallSite] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visits a function definition while tracking its qualname."""
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visits an async function definition while tracking its qualname."""
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visits a class definition while tracking nested qualnames."""
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Collects direct `get_settings()` calls."""
        if isinstance(node.func, ast.Name) and node.func.id == "get_settings":
            qualname = ".".join(self._stack) if self._stack else "<module>"
            self.call_sites.add((self._relative_path, qualname))
        self.generic_visit(node)


def _collect_get_settings_call_sites(repo_root: Path) -> set[GetSettingsCallSite]:
    """Returns every direct `get_settings()` call site under `ser/`."""
    package_root = repo_root / "ser"
    call_sites: set[GetSettingsCallSite] = set()
    for path in sorted(package_root.rglob("*.py")):
        relative_path = path.relative_to(repo_root).as_posix()
        visitor = _GetSettingsCallVisitor(relative_path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        call_sites.update(visitor.call_sites)
    return call_sites


def test_get_settings_usage_is_restricted_to_public_boundaries(repo_root: Path) -> None:
    """Source modules should avoid direct ambient settings lookups entirely."""
    assert _collect_get_settings_call_sites(repo_root) == EXPECTED_GET_SETTINGS_CALL_SITES
