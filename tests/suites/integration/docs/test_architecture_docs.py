"""Integration-style contract tests for contributor-facing architecture docs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_README_DOC_LINK_PATTERN = re.compile(
    r"https://github\.com/jsugg/ser/(?:blob|tree)/main/(docs/[A-Za-z0-9_./-]+)"
)
_ARCHITECTURE_RELATIVE_LINK_PATTERN = re.compile(r"\(([^)]+)\)")
_ARCHITECTURE_COUNT_PATTERN = re.compile(
    r"- (?P<label>Source modules under `ser/`|Test modules under `tests/`|"
    r"Public modules outside `_internal/`|Internal owner/helper modules under `_internal/`): "
    r"`(?P<count>\d+)`"
)
_REMOVED_TRACKER_REFERENCES = (
    "ser_refactor_implementation_journal.md",
    "ser_refactor_status.md",
)


def _repo_root() -> Path:
    """Returns the repository root for docs contract tests."""
    return Path(__file__).resolve().parents[4]


def _expected_codebase_counts(root: Path) -> dict[str, int]:
    """Builds the current architecture snapshot counts from the working tree."""
    source_files = list((root / "ser").rglob("*.py"))
    test_files = list((root / "tests").rglob("*.py"))
    return {
        "Source modules under `ser/`": len(source_files),
        "Test modules under `tests/`": len(test_files),
        "Public modules outside `_internal/`": sum(
            1 for path in source_files if "_internal" not in path.parts
        ),
        "Internal owner/helper modules under `_internal/`": sum(
            1 for path in source_files if "_internal" in path.parts
        ),
    }


def test_readme_architecture_links_resolve_to_existing_docs() -> None:
    """README architecture links should point at docs artifacts that exist in-tree."""
    root = _repo_root()
    readme_text = (root / "README.md").read_text(encoding="utf-8")
    resolved_targets = {
        root / relative_path for relative_path in _README_DOC_LINK_PATTERN.findall(readme_text)
    }

    assert resolved_targets
    assert all(target.is_file() for target in resolved_targets)


def test_architecture_index_links_resolve_to_existing_docs() -> None:
    """Architecture index should only reference local docs artifacts that exist."""
    root = _repo_root()
    architecture_text = (root / "docs" / "architecture.md").read_text(encoding="utf-8")
    resolved_targets = {
        (root / "docs" / relative_path).resolve()
        for relative_path in _ARCHITECTURE_RELATIVE_LINK_PATTERN.findall(architecture_text)
    }

    assert root / "docs" / "adr" / "README.md" in resolved_targets
    assert all(target.is_file() for target in resolved_targets)


def test_codebase_architecture_snapshot_counts_match_current_tree() -> None:
    """Architecture snapshot counts should match the current repository tree."""
    root = _repo_root()
    architecture_text = (root / "docs" / "codebase-architecture.md").read_text(encoding="utf-8")
    reported_counts = {
        match.group("label"): int(match.group("count"))
        for match in _ARCHITECTURE_COUNT_PATTERN.finditer(architecture_text)
    }

    assert reported_counts == _expected_codebase_counts(root)


def test_compatibility_matrix_does_not_reference_removed_tracker_docs() -> None:
    """Compatibility matrix should not point contributors at removed tracker files."""
    root = _repo_root()
    compatibility_text = (root / "docs" / "compatibility-matrix.md").read_text(encoding="utf-8")

    assert all(reference not in compatibility_text for reference in _REMOVED_TRACKER_REFERENCES)
