"""Architecture documentation contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARCHITECTURE_DOCS: tuple[Path, ...] = (
    _REPO_ROOT / "docs" / "architecture.md",
    _REPO_ROOT / "docs" / "architecture-diagram.md",
    _REPO_ROOT / "docs" / "architecture-refactor-roadmap.md",
    _REPO_ROOT / "docs" / "codebase-architecture.md",
    _REPO_ROOT / "docs" / "subsystem-dependency-map.md",
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_PUBLIC_INTERNAL_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+ser\._internal|import\s+ser\._internal)",
    re.MULTILINE,
)
_COUNT_PATTERNS: dict[str, re.Pattern[str]] = {
    "ser_modules": re.compile(r"Source modules under `ser/`: `(\d+)`"),
    "test_modules": re.compile(r"Test modules under `tests/`: `(\d+)`"),
    "public_modules": re.compile(r"Public modules outside `_internal/`: `(\d+)`"),
    "internal_modules": re.compile(r"Internal owner/helper modules under `_internal/`: `(\d+)`"),
    "public_internal_importers": re.compile(
        r"Public modules importing `_internal` directly: `(\d+)`"
    ),
}


def _iter_relative_links(document: Path) -> list[str]:
    """Returns all non-external relative markdown links in one document."""
    relative_links: list[str] = []
    for target in _MARKDOWN_LINK_RE.findall(document.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_target = target.split("#", 1)[0]
        if clean_target:
            relative_links.append(clean_target)
    return relative_links


def _architecture_snapshot() -> dict[str, int]:
    """Computes the current codebase counts described in the architecture docs."""
    ser_modules = tuple((_REPO_ROOT / "ser").rglob("*.py"))
    test_modules = tuple((_REPO_ROOT / "tests").rglob("*.py"))
    public_modules = tuple(
        path for path in ser_modules if "_internal" not in path.relative_to(_REPO_ROOT).parts
    )
    internal_modules = tuple(
        path for path in ser_modules if "_internal" in path.relative_to(_REPO_ROOT).parts
    )
    public_internal_importers = tuple(
        path
        for path in public_modules
        if _PUBLIC_INTERNAL_IMPORT_RE.search(path.read_text(encoding="utf-8"))
    )
    return {
        "ser_modules": len(ser_modules),
        "test_modules": len(test_modules),
        "public_modules": len(public_modules),
        "internal_modules": len(internal_modules),
        "public_internal_importers": len(public_internal_importers),
    }


def _documented_architecture_snapshot() -> dict[str, int]:
    """Parses the numeric architecture snapshot from the narrative architecture doc."""
    document = (_REPO_ROOT / "docs" / "codebase-architecture.md").read_text(encoding="utf-8")
    parsed: dict[str, int] = {}
    for key, pattern in _COUNT_PATTERNS.items():
        match = pattern.search(document)
        if match is None:
            raise AssertionError(
                f"Could not find architecture snapshot field {key!r} in docs/codebase-architecture.md."
            )
        parsed[key] = int(match.group(1))
    return parsed


def _documented_allowlist_paths() -> set[str]:
    """Parses the explicit allowlist in the subsystem dependency map."""
    document = (_REPO_ROOT / "docs" / "subsystem-dependency-map.md").read_text(encoding="utf-8")
    allowlist_section = document.split("## Explicit soft-boundary allowlist", maxsplit=1)[1].split(
        "## Dependency risk summary", maxsplit=1
    )[0]
    return set(re.findall(r"`(ser/[^`]+\.py)`", allowlist_section))


def _discovered_public_internal_importers() -> set[str]:
    """Discovers public modules importing ``ser._internal`` directly."""
    package_root = _REPO_ROOT / "ser"
    discovered: set[str] = set()
    for source_path in package_root.rglob("*.py"):
        relative_path = source_path.relative_to(_REPO_ROOT)
        if "_internal" in relative_path.parts:
            continue
        if _PUBLIC_INTERNAL_IMPORT_RE.search(source_path.read_text(encoding="utf-8")):
            discovered.add(relative_path.as_posix())
    return discovered


@pytest.mark.parametrize("document", _ARCHITECTURE_DOCS, ids=lambda path: path.name)
def test_architecture_doc_relative_links_resolve(document: Path) -> None:
    """Architecture markdown links should resolve from the document location."""
    unresolved_links = [
        target
        for target in _iter_relative_links(document)
        if not (document.parent / target).resolve().exists()
    ]
    assert unresolved_links == [], (
        f"{document.relative_to(_REPO_ROOT)} contains broken relative links: " f"{unresolved_links}"
    )


def test_codebase_architecture_snapshot_matches_repository_state() -> None:
    """Narrative architecture counts should stay aligned to the current workspace."""
    assert _documented_architecture_snapshot() == _architecture_snapshot()


def test_dependency_map_allowlist_matches_discovered_public_internal_importers() -> None:
    """The dependency-map allowlist should match the discovered public importer set."""
    documented_allowlist = _documented_allowlist_paths()
    discovered_allowlist = _discovered_public_internal_importers()

    assert documented_allowlist == discovered_allowlist
