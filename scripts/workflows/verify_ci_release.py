"""Verify that CI succeeded for a release commit."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def _required_env(name: str) -> str:
    """Return a required environment variable or exit with a clear error."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> int:
    """Check whether ci.yml has a successful run for the requested commit."""
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    repository = _required_env("GITHUB_REPOSITORY")
    token = _required_env("GITHUB_TOKEN")
    head_sha = _required_env("CI_HEAD_SHA")

    encoded_query = urllib.parse.urlencode({"head_sha": head_sha, "per_page": 20})
    url = f"{api_url}/repos/{repository}/actions/workflows/ci.yml/runs?{encoded_query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Failed to query GitHub Actions API: {exc}") from exc

    workflow_runs = payload.get("workflow_runs", [])
    successful_run = next(
        (run for run in workflow_runs if run.get("conclusion") == "success"),
        None,
    )
    if successful_run is None:
        raise SystemExit(f"No successful CI workflow run found for commit {head_sha}.")

    print(
        "CI verified for commit "
        f"{head_sha} via run #{successful_run.get('run_number', 'unknown')}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
