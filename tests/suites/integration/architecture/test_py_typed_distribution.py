"""Distribution contracts for the PEP 561 public typing marker."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def test_wheel_contains_py_typed_marker(repo_root: Path, tmp_path: Path) -> None:
    """A fresh wheel must ship the marker used by downstream type checkers."""
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_paths = tuple(tmp_path.glob("ser-*.whl"))
    assert len(wheel_paths) == 1
    with zipfile.ZipFile(wheel_paths[0]) as wheel:
        assert "ser/py.typed" in wheel.namelist()
