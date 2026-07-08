"""Contracts for pull request scope classification used by default CI."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Runs a command and returns captured text output."""
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _git(repo: Path, *args: str) -> str:
    """Runs git in a temporary fixture repository."""
    return _run(["git", *args], cwd=repo).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    """Creates a commit and returns its SHA."""
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def classifier_repo(tmp_path: Path) -> Path:
    """Creates a tiny git repository for classifier contract cases."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "ci-contracts@example.invalid")
    _git(repo, "config", "user.name", "CI Contracts")
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _commit(repo, "initial")
    return repo


def _classify(
    *,
    repo: Path,
    script_path: Path,
    event_name: str,
    base_sha: str | None = None,
    head_sha: str | None = None,
) -> dict[str, str]:
    """Runs the classifier script and returns GitHub-output key/value pairs."""
    output_path = repo / "classifier-output.txt"
    env = os.environ.copy()
    env["CI_EVENT_NAME"] = event_name
    if base_sha is not None:
        env["CI_BASE_SHA"] = base_sha
    if head_sha is not None:
        env["CI_HEAD_SHA"] = head_sha

    _run(["bash", str(script_path), str(output_path)], cwd=repo, env=env)
    return dict(
        line.split("=", maxsplit=1) for line in output_path.read_text(encoding="utf-8").splitlines()
    )


def _write(repo: Path, relative_path: str, content: str = "changed\n") -> None:
    """Writes one fixture file."""
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize("event_name", ["push", "schedule", "workflow_dispatch"])
def test_non_pull_request_events_always_run_full_ci(
    classifier_repo: Path,
    repo_root: Path,
    event_name: str,
) -> None:
    """Push, schedule, and manual events should not use docs-only shortcuts."""
    result = _classify(
        repo=classifier_repo,
        script_path=repo_root / "scripts" / "ci_classify_changes.sh",
        event_name=event_name,
    )

    assert result == {
        "run_full": "true",
        "docs_only": "false",
        "reason": "non_pull_request",
    }


def test_empty_pull_request_diff_runs_full_ci(classifier_repo: Path, repo_root: Path) -> None:
    """An empty PR diff should not be treated as a docs-only pass."""
    head_sha = _git(classifier_repo, "rev-parse", "HEAD")

    result = _classify(
        repo=classifier_repo,
        script_path=repo_root / "scripts" / "ci_classify_changes.sh",
        event_name="pull_request",
        base_sha=head_sha,
        head_sha=head_sha,
    )

    assert result == {
        "run_full": "true",
        "docs_only": "false",
        "reason": "empty_diff",
    }


def test_docs_assets_and_codeowners_pull_request_can_skip_heavy_ci(
    classifier_repo: Path,
    repo_root: Path,
) -> None:
    """Docs, assets, and CODEOWNERS-only PRs should be explicit safe skips."""
    base_sha = _git(classifier_repo, "rev-parse", "HEAD")
    _write(classifier_repo, "docs/ci-note.md")
    _write(classifier_repo, ".github/assets/banner.txt")
    _write(classifier_repo, ".github/CODEOWNERS", "* @jsugg\n")
    head_sha = _commit(classifier_repo, "docs only")

    result = _classify(
        repo=classifier_repo,
        script_path=repo_root / "scripts" / "ci_classify_changes.sh",
        event_name="pull_request",
        base_sha=base_sha,
        head_sha=head_sha,
    )

    assert result == {
        "run_full": "false",
        "docs_only": "true",
        "reason": "docs_only_pull_request",
    }


@pytest.mark.parametrize(
    "changed_path",
    [
        "pyproject.toml",
        "uv.lock",
        "Makefile",
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
        "scripts/ci_classify_changes.sh",
        "ser/api.py",
        "tests/suites/unit/test_example.py",
        "unknown/generated.bin",
    ],
)
def test_code_tooling_and_unknown_pull_request_paths_run_full_ci(
    classifier_repo: Path,
    repo_root: Path,
    changed_path: str,
) -> None:
    """Executable, package, test, workflow, and unknown paths should run full CI."""
    base_sha = _git(classifier_repo, "rev-parse", "HEAD")
    _write(classifier_repo, changed_path)
    head_sha = _commit(classifier_repo, f"change {changed_path}")

    result = _classify(
        repo=classifier_repo,
        script_path=repo_root / "scripts" / "ci_classify_changes.sh",
        event_name="pull_request",
        base_sha=base_sha,
        head_sha=head_sha,
    )

    assert result == {
        "run_full": "true",
        "docs_only": "false",
        "reason": "full_ci_required",
    }
