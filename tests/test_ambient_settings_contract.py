"""Architecture contract tests for ambient settings usage."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "ser"

type GetSettingsCallSite = tuple[str, str]

EXPECTED_GET_SETTINGS_CALL_SITES: set[GetSettingsCallSite] = {
    ("ser/api.py", "_resolve_boundary_settings"),
    ("ser/data/data_loader.py", "_resolve_boundary_settings"),
    ("ser/diagnostics/command.py", "_resolve_boundary_settings"),
    ("ser/features/feature_extractor.py", "_resolve_boundary_settings"),
    ("ser/models/emotion_model.py", "_resolve_boundary_settings"),
    ("ser/runtime/profile_quality_gate.py", "_resolve_boundary_settings"),
    ("ser/transcript/profiling.py", "_resolve_boundary_settings"),
    ("ser/transcript/transcript_extractor.py", "_resolve_boundary_settings"),
    ("ser/utils/__init__.py", "_resolve_boundary_settings"),
}


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


def _collect_get_settings_call_sites() -> set[GetSettingsCallSite]:
    """Returns every direct `get_settings()` call site under `ser/`."""
    call_sites: set[GetSettingsCallSite] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        visitor = _GetSettingsCallVisitor(relative_path)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        call_sites.update(visitor.call_sites)
    return call_sites


def test_get_settings_usage_is_restricted_to_public_boundaries() -> None:
    """Ambient settings lookup should remain limited to approved boundary wrappers."""
    assert _collect_get_settings_call_sites() == EXPECTED_GET_SETTINGS_CALL_SITES
