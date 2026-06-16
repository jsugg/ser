"""Contracts for wheel-install smoke test script."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_wheel_smoke_script_has_valid_bash_syntax(repo_root: Path) -> None:
    """Wheel smoke script should parse before packaging CI invokes it."""
    script_path = repo_root / "scripts" / "workflows" / "smoke_test_wheel_install.sh"

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_wheel_smoke_script_forces_fresh_same_version_install(repo_root: Path) -> None:
    """Wheel smoke should install the current wheel even when package version is unchanged."""
    script_path = repo_root / "scripts" / "workflows" / "smoke_test_wheel_install.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "rm -rf .pkg-smoke" in script_text
    assert "pip install --force-reinstall --no-deps" in script_text
