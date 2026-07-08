"""Contracts for full-dataset quality gate shell script defaults."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_full_dataset_quality_gate_script_has_valid_bash_syntax(repo_root: Path) -> None:
    """Full-gate script should parse before CI invokes expensive profile work."""
    script_path = repo_root / "scripts" / "run_full_dataset_quality_gate.sh"

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_full_dataset_quality_gate_default_models_dir_is_platform_aware(repo_root: Path) -> None:
    """Default artifact lookup should match SER platform data-dir conventions."""
    script_path = repo_root / "scripts" / "run_full_dataset_quality_gate.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "Library/Application Support/ser/models" in script_text
    assert "${XDG_DATA_HOME:-$HOME/.local/share}/ser/models" in script_text
    assert (
        'models_dir="${SER_MODELS_DIR:-$HOME/Library/Application Support/ser/models}"'
        not in script_text
    )


def test_full_dataset_quality_gate_uses_frozen_uv_commands(repo_root: Path) -> None:
    """Full-gate script should not mutate the project lockfile in CI lanes."""
    script_path = repo_root / "scripts" / "run_full_dataset_quality_gate.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "uv run --frozen ser --train" in script_text
    assert "uv run --frozen --extra medium ser --train" in script_text
    assert (
        "uv run --frozen --extra medium python -m ser.runtime.profile_quality_gate" in script_text
    )
