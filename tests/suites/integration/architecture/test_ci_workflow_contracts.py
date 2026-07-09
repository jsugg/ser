"""Contracts for GitHub Actions workflow policy and release safety."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

pytestmark = pytest.mark.topology_contract


class _GitHubActionsLoader(yaml.SafeLoader):
    """YAML loader that keeps GitHub's `on` key as a string."""


_GitHubActionsLoader.yaml_implicit_resolvers = {
    key: [resolver for resolver in resolvers if resolver[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _as_mapping(value: object, *, context: str) -> dict[str, object]:
    """Returns a typed mapping or fails with workflow context."""
    if not isinstance(value, dict):
        raise AssertionError(f"{context} should be a mapping.")
    if not all(isinstance(key, str) for key in value):
        raise AssertionError(f"{context} should have string keys.")
    return cast(dict[str, object], value)


def _as_sequence(value: object, *, context: str) -> list[object]:
    """Returns a typed sequence or fails with workflow context."""
    if not isinstance(value, list):
        raise AssertionError(f"{context} should be a sequence.")
    return value


def _workflow(repo_root: Path, workflow_name: str) -> dict[str, object]:
    """Loads one GitHub Actions workflow."""
    workflow_path = repo_root / ".github" / "workflows" / workflow_name
    loaded = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=_GitHubActionsLoader)
    return _as_mapping(loaded, context=workflow_name)


def _jobs(workflow: dict[str, object]) -> dict[str, object]:
    """Returns a workflow's jobs mapping."""
    return _as_mapping(workflow["jobs"], context="jobs")


def _job(workflow: dict[str, object], job_id: str) -> dict[str, object]:
    """Returns one workflow job by id."""
    return _as_mapping(_jobs(workflow)[job_id], context=f"job {job_id}")


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    """Returns typed steps for a normal job."""
    return [
        _as_mapping(step, context="step") for step in _as_sequence(job["steps"], context="steps")
    ]


def _needs(job: dict[str, object]) -> set[str]:
    """Normalizes a job's needs value into a set of job ids."""
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    return {str(need) for need in _as_sequence(needs, context="needs")}


def _run_commands(job: dict[str, object]) -> str:
    """Concatenates shell snippets from one job."""
    return "\n".join(str(step["run"]) for step in _steps(job) if "run" in step)


def _step_by_name(job: dict[str, object], step_name: str) -> dict[str, object]:
    """Returns one workflow step by display name."""
    for step in _steps(job):
        if step.get("name") == step_name:
            return step
    raise AssertionError(f"Missing workflow step: {step_name}")


def _event_names(workflow: dict[str, object]) -> set[str]:
    """Returns event names from one workflow trigger block."""
    trigger = workflow["on"]
    if isinstance(trigger, str):
        return {trigger}
    if isinstance(trigger, list):
        return {str(event_name) for event_name in trigger}
    return set(_as_mapping(trigger, context="on"))


def _workflow_paths(repo_root: Path) -> list[Path]:
    """Returns all workflow YAML files."""
    workflows_dir = repo_root / ".github" / "workflows"
    return sorted([*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")])


def test_no_workflow_uses_pull_request_target(repo_root: Path) -> None:
    """Workflows should not run untrusted pull requests with privileged semantics."""
    offenders: list[str] = []
    for workflow_path in _workflow_paths(repo_root):
        workflow = _workflow(repo_root, workflow_path.name)
        if "pull_request_target" in _event_names(workflow):
            offenders.append(workflow_path.name)

    assert offenders == []


def test_workflow_permissions_are_least_privilege(repo_root: Path) -> None:
    """Workflow permissions should stay explicit and narrow."""
    expected_permissions = {
        "ci.yml": {"contents": "read"},
        "codeql.yml": {"actions": "read", "contents": "read", "security-events": "write"},
        "darwin-x86_64-validation.yml": {"contents": "read"},
        "dependency-review.yml": {"contents": "read", "pull-requests": "read"},
        "full-dataset-quality-gate-regression.yml": {"contents": "read"},
        "linux-python-3_13-cli-validation.yml": {"contents": "read"},
        "linux-selfhosted-gpu-validation.yml": {"contents": "read"},
        "macos15-mps-validation.yml": {"contents": "read"},
        "python-publish-testpypi.yml": {"contents": "read"},
        "python-publish.yml": {"contents": "read"},
        "scorecard.yml": {"contents": "read"},
    }

    for workflow_name, expected in expected_permissions.items():
        workflow = _workflow(repo_root, workflow_name)
        assert (
            _as_mapping(workflow["permissions"], context=f"{workflow_name} permissions") == expected
        )

        for job_id, raw_job in _jobs(workflow).items():
            job = _as_mapping(raw_job, context=f"{workflow_name}:{job_id}")
            raw_permissions = job.get("permissions")
            if raw_permissions is None:
                continue
            permissions = _as_mapping(
                raw_permissions, context=f"{workflow_name}:{job_id} permissions"
            )
            if permissions.get("id-token") == "write":
                assert workflow_name in {"python-publish.yml", "python-publish-testpypi.yml"}
                assert job_id.startswith("publish-to-")


def test_action_refs_do_not_use_default_branch_refs(repo_root: Path) -> None:
    """Action refs should avoid mutable default branches while SHA pinning is deferred."""
    offenders: list[str] = []
    for workflow_path in _workflow_paths(repo_root):
        workflow = _workflow(repo_root, workflow_path.name)
        for job_id, raw_job in _jobs(workflow).items():
            job = _as_mapping(raw_job, context=f"{workflow_path.name}:{job_id}")
            for step in _steps(job) if "steps" in job else []:
                uses = step.get("uses")
                if not isinstance(uses, str) or uses.startswith("./"):
                    continue
                if "@" not in uses:
                    offenders.append(f"{workflow_path.name}:{job_id}:{uses}")
                    continue
                ref = uses.rsplit("@", maxsplit=1)[1]
                if ref in {"main", "master", "HEAD"}:
                    offenders.append(f"{workflow_path.name}:{job_id}:{uses}")

    assert offenders == []


def test_ci_required_aggregate_covers_all_critical_gates(repo_root: Path) -> None:
    """Default CI should expose one stable always-reporting required check."""
    workflow = _workflow(repo_root, "ci.yml")
    jobs = _jobs(workflow)

    assert {
        "changes",
        "code-quality",
        "resolve",
        "tests",
        "coverage",
        "contract-gates",
        "build",
        "required-ci",
    }.issubset(set(jobs))

    required_ci = _job(workflow, "required-ci")
    assert _needs(required_ci) == {
        "changes",
        "code-quality",
        "resolve",
        "tests",
        "coverage",
        "contract-gates",
        "build",
    }
    assert "always()" in str(required_ci["if"])

    gate_script = _run_commands(required_ci)
    for result_name in (
        "CODE_QUALITY_RESULT",
        "RESOLVE_RESULT",
        "TESTS_RESULT",
        "COVERAGE_RESULT",
        "CONTRACT_GATES_RESULT",
        "BUILD_RESULT",
    ):
        assert result_name in gate_script
    assert "docs_only_pull_request" in gate_script
    assert "GITHUB_STEP_SUMMARY" in gate_script


def test_ci_visibility_and_lock_controls_are_wired(repo_root: Path) -> None:
    """Default CI should run lock, workflow, report, and artifact visibility controls."""
    workflow = _workflow(repo_root, "ci.yml")
    assert "workflow_dispatch" in _event_names(workflow)

    code_quality_commands = _run_commands(_job(workflow, "code-quality"))
    assert "make lock-check" in code_quality_commands
    assert "make type-completeness" in code_quality_commands
    assert "make ci-contracts" in code_quality_commands
    assert "make workflow-lint" in code_quality_commands

    test_commands = _run_commands(_job(workflow, "tests"))
    assert "--junitxml=reports/pytest/pytest-${{ matrix.python-version }}.xml" in test_commands
    assert (
        repo_root / "tests/suites/integration/architecture/test_public_api_snapshot.py"
    ).exists()

    coverage_job = _job(workflow, "coverage")
    assert "make test-cov" in _run_commands(coverage_job)
    assert any(step.get("uses") == "actions/upload-artifact@v7" for step in _steps(coverage_job))

    build_job = _job(workflow, "build")
    build_commands = _run_commands(build_job)
    assert "uv build" in build_commands
    assert "uvx --from twine twine check --strict dist/*" in build_commands
    assert any(step.get("uses") == "actions/upload-artifact@v7" for step in _steps(build_job))


def test_accurate_research_smokes_pass_github_token_to_torch_hub(repo_root: Path) -> None:
    """Accurate-research VAD should avoid unauthenticated `torch.hub` API rate limits."""
    expectations = (
        (
            "linux-python-3_13-cli-validation.yml",
            "linux-py313-profile-smoke",
            "Accurate-research profile train and predict (compatibility smoke)",
        ),
        (
            "linux-selfhosted-gpu-validation.yml",
            "cuda-selfhosted-profile-smoke",
            "Accurate-research profile train and predict (optional)",
        ),
        (
            "linux-selfhosted-gpu-validation.yml",
            "xpu-selfhosted-profile-smoke",
            "Accurate-research profile train and predict (optional)",
        ),
        (
            "macos15-mps-validation.yml",
            "macos15-mps-profile-smoke",
            "Accurate-research profile train and predict (optional)",
        ),
    )

    for workflow_name, job_id, step_name in expectations:
        workflow = _workflow(repo_root, workflow_name)
        step = _step_by_name(_job(workflow, job_id), step_name)
        env = _as_mapping(step["env"], context=f"{workflow_name}:{job_id}:{step_name} env")

        assert env["GITHUB_TOKEN"] == "${{ github.token }}"


def test_dependency_review_and_update_automation_are_advisory(repo_root: Path) -> None:
    """Dependency review starts advisory and Dependabot covers uv plus actions."""
    workflow = _workflow(repo_root, "dependency-review.yml")
    review_job = _job(workflow, "dependency-review")
    review_steps = _steps(review_job)
    review_step = next(step for step in review_steps if step.get("id") == "dependency-review")

    assert review_step["uses"] == "actions/dependency-review-action@v5"
    assert review_step["continue-on-error"] == "true"
    assert "critical" in str(review_step["with"])

    dependabot = yaml.safe_load(
        (repo_root / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    )
    updates = _as_sequence(
        _as_mapping(dependabot, context="dependabot")["updates"], context="updates"
    )
    ecosystems = {
        _as_mapping(update, context="dependabot update")["package-ecosystem"] for update in updates
    }
    assert {"uv", "github-actions"}.issubset(ecosystems)


def test_advisory_code_scanning_and_scorecard_are_configured(repo_root: Path) -> None:
    """CodeQL and Scorecard should be visible without becoming required checks."""
    codeql = _workflow(repo_root, "codeql.yml")
    assert {"pull_request", "push", "schedule", "workflow_dispatch"}.issubset(_event_names(codeql))
    codeql_commands = "\n".join(
        str(step.get("uses", "")) for step in _steps(_job(codeql, "analyze"))
    )
    assert "github/codeql-action/init@v4" in codeql_commands
    assert "github/codeql-action/analyze@v4" in codeql_commands

    scorecard = _workflow(repo_root, "scorecard.yml")
    assert {"schedule", "workflow_dispatch"} == _event_names(scorecard)
    scorecard_steps = _steps(_job(scorecard, "scorecard"))
    assert any(step.get("uses") == "ossf/scorecard-action@v2.4.3" for step in scorecard_steps)
