"""Behavioral contracts for public-to-private import boundary checking."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_policy(repo_root: Path, paths: tuple[str, ...]) -> None:
    """Writes one minimal valid boundary policy for a synthetic repository."""
    entries = "\n".join(
        (
            "[[public_internal_import]]\n"
            f'path = "{path}"\n'
            'reason = "Synthetic facade for boundary-check regression coverage."\n'
        )
        for path in paths
    )
    (repo_root / "boundary_policy.toml").write_text(f"version = 1\n\n{entries}", encoding="utf-8")


def _run_checker(repo_root: Path, project_root: Path) -> subprocess.CompletedProcess[str]:
    """Runs the real checker against one synthetic repository root."""
    return subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "check_public_internal_imports.py"),
            "--repo-root",
            str(repo_root),
        ],
        capture_output=True,
        text=True,
    )


def test_checker_rejects_unlisted_dynamic_private_import(repo_root: Path, tmp_path: Path) -> None:
    """A literal `importlib.import_module` bypass must fail the boundary check."""
    package_root = tmp_path / "ser"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "allowed.py").write_text(
        "from ser._internal.owner import allowed\n",
        encoding="utf-8",
    )
    (package_root / "facade.py").write_text(
        'import importlib\nimportlib.import_module("ser._internal.owner")\n',
        encoding="utf-8",
    )
    _write_policy(tmp_path, ("ser/allowed.py",))

    completed = _run_checker(tmp_path, repo_root)

    assert completed.returncode != 0
    assert "ser/facade.py:2: imports private target" in completed.stderr


def test_checker_accepts_policy_listed_dynamic_private_import(
    repo_root: Path, tmp_path: Path
) -> None:
    """A policy-listed literal lazy facade import remains valid and auditable."""
    package_root = tmp_path / "ser"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "facade.py").write_text(
        'from importlib import import_module\nimport_module("ser._internal.owner")\n',
        encoding="utf-8",
    )
    _write_policy(tmp_path, ("ser/facade.py",))

    completed = _run_checker(tmp_path, repo_root)

    assert completed.returncode == 0, completed.stderr or completed.stdout
