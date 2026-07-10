"""Contract test for the reviewed tier-1 public API snapshot."""

from __future__ import annotations

import difflib
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract


def test_public_api_snapshot_matches_current_surface(repo_root: Path) -> None:
    """Tier-1 public API drift should require an explicit snapshot update."""
    snapshot_path = repo_root / "tests/suites/integration/architecture/public_api_snapshot.json"
    dump_command = [sys.executable, "scripts/dump_public_api.py"]
    completed = subprocess.run(
        dump_command,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "Public API snapshot dump failed. Fix the exported surface or run "
        "`uv run --frozen --extra dev python scripts/dump_public_api.py --write` "
        f"after an intentional API change.\n{completed.stderr}"
    )

    expected = snapshot_path.read_text(encoding="utf-8")
    actual = completed.stdout
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(),
                actual.splitlines(),
                fromfile=str(snapshot_path),
                tofile="current public API",
                lineterm="",
            )
        )
        raise AssertionError(
            "Tier-1 public API snapshot drifted. If this is intentional, run "
            "`uv run --frozen --extra dev python scripts/dump_public_api.py --write` "
            "and review the JSON diff.\n"
            f"{diff[:12000]}"
        )


def test_public_api_snapshot_records_reexported_type_contracts(repo_root: Path) -> None:
    """Re-exported API classes must include fields, not only owner-module paths."""
    completed = subprocess.run(
        [sys.executable, "scripts/dump_public_api.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    snapshot = json.loads(completed.stdout)
    inference_request = snapshot["modules"]["ser.api"]["exports"]["InferenceRequest"]
    assert inference_request["kind"] == "alias"
    contract = inference_request["contract"]
    assert contract["kind"] == "class"
    assert "file_path" in contract["members"]
