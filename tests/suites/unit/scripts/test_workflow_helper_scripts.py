"""Contracts for shell helpers invoked by GitHub workflow lanes."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_workflow_helper_scripts_have_valid_bash_syntax(repo_root: Path) -> None:
    """Workflow shell helpers should parse before CI invokes expensive lanes."""
    script_paths = [
        repo_root / "scripts" / "configure_validation_dataset_consents.sh",
        repo_root / "scripts" / "setup_compatible_env.sh",
        repo_root / "scripts" / "workflows" / "run_profile_smoke.sh",
        repo_root / "scripts" / "workflows" / "setup_validation_environment.sh",
        repo_root / "scripts" / "workflows" / "sync_validation_dependencies.sh",
    ]

    for script_path in script_paths:
        subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_workflow_helpers_default_to_frozen_ci_uv_paths(repo_root: Path) -> None:
    """CI-specific helpers should use lock-preserving uv invocations."""
    run_profile_smoke = (repo_root / "scripts" / "workflows" / "run_profile_smoke.sh").read_text(
        encoding="utf-8"
    )
    validation_consents = (
        repo_root / "scripts" / "configure_validation_dataset_consents.sh"
    ).read_text(encoding="utf-8")
    setup_validation = (
        repo_root / "scripts" / "workflows" / "setup_validation_environment.sh"
    ).read_text(encoding="utf-8")

    assert "uv_run_args+=(--frozen)" in run_profile_smoke
    assert 'uv run --frozen "$@" ser configure' in validation_consents
    assert 'setup_args=(--python "$python_version" --frozen --skip-git-hooks)' in setup_validation
