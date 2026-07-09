"""Import-cost contracts for tier-1 public modules."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract


def test_tier_one_public_imports_do_not_import_torch(repo_root: Path) -> None:
    """Tier-1 public module imports should not eagerly import torch."""
    script = """
import sys

import ser
import ser.api
import ser.config
import ser.domain
import ser.profiles
import ser.utils

if "torch" in sys.modules:
    raise SystemExit("torch imported during tier-1 public imports")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
