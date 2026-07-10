#!/usr/bin/env python3
"""Enforce the public-to-private import boundary from `boundary_policy.toml`."""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

MAX_POLICY_ENTRIES = 10
PACKAGE_NAME = "ser"
POLICY_FILE_NAME = "boundary_policy.toml"


@dataclass(frozen=True)
class _ImportOccurrence:
    """One statically identifiable public import of a private module."""

    source_path: Path
    line_number: int
    target: str


def _mapping(value: object, description: str) -> dict[str, object]:
    """Returns a validated mapping from an untrusted TOML boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a mapping.")
    return cast(dict[str, object], value)


def _policy_paths(repo_root: Path) -> set[Path]:
    """Loads and validates the exact public files allowed to import private owners."""
    raw_policy: object = tomllib.loads((repo_root / POLICY_FILE_NAME).read_text(encoding="utf-8"))
    policy = _mapping(raw_policy, POLICY_FILE_NAME)
    raw_entries = policy.get("public_internal_import")
    if not isinstance(raw_entries, list):
        raise ValueError("boundary policy must define public_internal_import tables.")
    if not raw_entries:
        raise ValueError("boundary policy must contain at least one public facade.")
    if len(raw_entries) > MAX_POLICY_ENTRIES:
        raise ValueError(
            f"boundary policy has {len(raw_entries)} entries; maximum is {MAX_POLICY_ENTRIES}."
        )

    paths: list[Path] = []
    raw_paths: list[str] = []
    for index, raw_entry in enumerate(raw_entries, start=1):
        entry = _mapping(raw_entry, f"boundary policy entry {index}")
        raw_path = entry.get("path")
        reason = entry.get("reason")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"boundary policy entry {index} has no non-empty path.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"boundary policy entry {index} has no non-empty reason.")
        normalized_path = raw_path.strip()
        if not normalized_path.startswith(f"{PACKAGE_NAME}/") or not normalized_path.endswith(
            ".py"
        ):
            raise ValueError(
                f"boundary policy entry {index} must name a public Python source path: {normalized_path!r}."
            )
        resolved_path = (repo_root / normalized_path).resolve()
        if not resolved_path.is_relative_to(repo_root) or not resolved_path.is_file():
            raise ValueError(
                f"boundary policy entry {index} path does not exist: {normalized_path!r}."
            )
        if "_internal" in resolved_path.relative_to(repo_root).parts:
            raise ValueError(f"boundary policy entry {index} must not name a private source path.")
        raw_paths.append(normalized_path)
        paths.append(resolved_path)

    if raw_paths != sorted(raw_paths):
        raise ValueError("boundary policy entries must be sorted by path.")
    if len(set(paths)) != len(paths):
        raise ValueError("boundary policy paths must be unique.")
    return set(paths)


def _module_name(repo_root: Path, source_path: Path) -> str:
    """Returns the importable module name for one public source file."""
    relative_path = source_path.relative_to(repo_root).with_suffix("")
    parts = list(relative_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _relative_module_name(source_path: Path, repo_root: Path, node: ast.ImportFrom) -> str:
    """Resolves a relative import module name without executing source code."""
    if node.level == 0:
        return node.module or ""
    source_module = _module_name(repo_root, source_path)
    source_package = (
        source_module if source_path.name == "__init__.py" else source_module.rpartition(".")[0]
    )
    package_parts = source_package.split(".") if source_package else []
    parent_count = node.level - 1
    if parent_count >= len(package_parts):
        return node.module or ""
    base_parts = package_parts[: len(package_parts) - parent_count]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _is_private_module(module_name: str) -> bool:
    """Returns whether an import target is inside the private implementation tree."""
    return module_name == f"{PACKAGE_NAME}._internal" or module_name.startswith(
        f"{PACKAGE_NAME}._internal."
    )


def _literal_string(node: ast.expr) -> str | None:
    """Returns one statically known string import target, if present."""
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _importlib_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Finds local names that can invoke `importlib.import_module`."""
    module_aliases = {"importlib"}
    function_aliases = {"import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    function_aliases.add(alias.asname or alias.name)
    return module_aliases, function_aliases


def _is_dynamic_import_call(
    node: ast.Call,
    *,
    module_aliases: set[str],
    function_aliases: set[str],
) -> bool:
    """Returns whether one call performs a dynamic module import."""
    if isinstance(node.func, ast.Name):
        return node.func.id in function_aliases or node.func.id == "__import__"
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in module_aliases
        and node.func.attr == "import_module"
    )


def _discover_private_imports(source_path: Path, repo_root: Path) -> tuple[_ImportOccurrence, ...]:
    """Finds direct and dynamic private imports in one public module.

    Dynamic import targets must be literal strings so the boundary remains auditable.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    module_aliases, function_aliases = _importlib_aliases(tree)
    occurrences: list[_ImportOccurrence] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_private_module(alias.name):
                    occurrences.append(_ImportOccurrence(source_path, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            imported_module = _relative_module_name(source_path, repo_root, node)
            if _is_private_module(imported_module):
                occurrences.append(_ImportOccurrence(source_path, node.lineno, imported_module))
            elif imported_module == PACKAGE_NAME and any(
                alias.name == "_internal" for alias in node.names
            ):
                occurrences.append(
                    _ImportOccurrence(source_path, node.lineno, f"{PACKAGE_NAME}._internal")
                )
        elif isinstance(node, ast.Call) and _is_dynamic_import_call(
            node,
            module_aliases=module_aliases,
            function_aliases=function_aliases,
        ):
            if not node.args:
                raise ValueError(
                    f"{source_path}:{node.lineno}: dynamic import has no target argument."
                )
            target = _literal_string(node.args[0])
            if target is None:
                raise ValueError(
                    f"{source_path}:{node.lineno}: dynamic import target must be a literal string."
                )
            if target.startswith("."):
                raise ValueError(
                    f"{source_path}:{node.lineno}: relative dynamic imports are not permitted in public modules."
                )
            if _is_private_module(target):
                occurrences.append(_ImportOccurrence(source_path, node.lineno, target))
    return tuple(occurrences)


def check_public_internal_imports(repo_root: Path) -> tuple[str, ...]:
    """Returns actionable boundary-policy violations for the repository.

    Args:
        repo_root: Repository root containing `ser` and `boundary_policy.toml`.

    Returns:
        Sorted human-readable violations. An empty tuple means the boundary holds.
    """
    policy_paths = _policy_paths(repo_root)
    package_root = repo_root / PACKAGE_NAME
    violations: list[str] = []
    occurrences: list[_ImportOccurrence] = []
    for source_path in sorted(package_root.rglob("*.py")):
        if "_internal" in source_path.relative_to(package_root).parts:
            continue
        try:
            occurrences.extend(_discover_private_imports(source_path, repo_root))
        except (SyntaxError, ValueError) as error:
            violations.append(str(error))

    discovered_paths = {occurrence.source_path.resolve() for occurrence in occurrences}
    unexpected_paths = sorted(discovered_paths - policy_paths)
    stale_paths = sorted(policy_paths - discovered_paths)
    for occurrence in occurrences:
        if occurrence.source_path.resolve() in unexpected_paths:
            relative_path = occurrence.source_path.relative_to(repo_root)
            violations.append(
                f"{relative_path}:{occurrence.line_number}: imports private target {occurrence.target!r} "
                "without a boundary-policy entry."
            )
    for source_path in stale_paths:
        violations.append(
            f"{source_path.relative_to(repo_root)}: boundary-policy entry is stale; no private import found."
        )
    return tuple(sorted(set(violations)))


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root to validate",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    try:
        violations = check_public_internal_imports(repo_root)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"public-to-private import check failed: {error}", file=sys.stderr)
        return 1
    if not violations:
        return 0
    for violation in violations:
        print(violation, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
