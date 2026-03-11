"""Architecture documentation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWLIST_FILE = _REPO_ROOT / "tests" / "fixtures" / "public_internal_allowlist.txt"
_DEPENDENCY_MAP = _REPO_ROOT / "docs" / "subsystem-dependency-map.md"


def _explicit_allowlist_paths() -> set[Path]:
    """Loads the architecture allowlist contract from test-owned fixture data."""
    return {
        (_REPO_ROOT / relative_path).resolve()
        for relative_path in _ALLOWLIST_FILE.read_text(encoding="utf-8").splitlines()
        if relative_path and not relative_path.startswith("#")
    }


def _documented_allowlist_paths() -> set[Path]:
    """Loads the public-to-internal allowlist documented in the dependency map."""
    document = _DEPENDENCY_MAP.read_text(encoding="utf-8")
    allowlist_section = document.split("## Explicit soft-boundary allowlist", maxsplit=1)[1].split(
        "## Dependency risk summary", maxsplit=1
    )[0]
    return {
        (_REPO_ROOT / relative_path).resolve()
        for relative_path in re.findall(r"`(ser/[^`]+\.py)`", allowlist_section)
    }


def test_dependency_map_allowlist_matches_boundary_contract() -> None:
    """Architecture docs should stay aligned with the executable allowlist contract."""
    assert _documented_allowlist_paths() == _explicit_allowlist_paths()
