"""Repository documentation link contract tests."""

from __future__ import annotations

import re
from pathlib import Path

_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_EXTERNAL_LINK_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _markdown_files(repo_root: Path) -> list[Path]:
    """Returns repository Markdown files covered by local link checks."""
    return [
        repo_root / "README.md",
        repo_root / "CONTRIBUTING.md",
        *sorted((repo_root / "docs").rglob("*.md")),
    ]


def _local_markdown_link_targets(markdown_file: Path) -> list[str]:
    """Extracts non-external Markdown link targets from one file."""
    targets: list[str] = []
    for match in _MARKDOWN_LINK_RE.finditer(markdown_file.read_text(encoding="utf-8")):
        raw_target = match.group(1).split("#", 1)[0]
        if (
            not raw_target
            or _EXTERNAL_LINK_RE.match(raw_target)
            or raw_target.startswith("mailto:")
        ):
            continue
        targets.append(raw_target)
    return targets


def test_local_markdown_links_resolve(repo_root: Path) -> None:
    """Local documentation links should point at files present in this repository."""
    missing_links: list[str] = []
    for markdown_file in _markdown_files(repo_root):
        for target in _local_markdown_link_targets(markdown_file):
            resolved_target = (markdown_file.parent / target).resolve()
            if not resolved_target.exists():
                relative_source = markdown_file.relative_to(repo_root)
                missing_links.append(f"{relative_source}: {target}")

    assert missing_links == []
