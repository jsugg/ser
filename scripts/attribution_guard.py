#!/usr/bin/env python3
"""Reject AI-tool attribution in refs, commit metadata, and pull request metadata.

Modes:
  commit-msg FILE                    git commit-msg hook (pre-commit `commit-msg` stage).
  pre-push [REMOTE] [--stdin]        git pre-push hook. Reads git's ref lines from stdin with
                                     `--stdin`, else pre-commit's `PRE_COMMIT_*` variables, else
                                     checks the current branch and its unpushed commits.
  push --before SHA --after SHA      Post-merge audit of the commits pushed to a branch.
  pull-request --repo O/R --number N Refetches pull request metadata from the GitHub API.

The pattern block below is a byte-identical copy of the shared attribution source that the
repository rulesets and local agent hooks also use. Re-vendor it; never edit it in place.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

# --- Vendored attribution patterns (byte-identical; do not edit) ---
AGENT_NAMES = r"(?:claude|anthropic|codex|openai|chatgpt|copilot)"
TOOL_NAMES = (
    r"(?:claude|anthropic|codex|openai|chatgpt|copilot|gemini|cursor|windsurf|devin|aider|cline)"
)

# Tool name used as a namespace (claude/..., ai-codex/...), not as the subject of the work.
REF_NAME = rf"(?i)^(?:refs/(?:heads|tags)/)?(?:(?:ai|agents?|assistants?|bots?|llm)[/._-])?{AGENT_NAMES}(?:/|\n?$)"

COMMIT_MESSAGE = (
    r"(?im)"
    rf"(?:^[ \t]*(?:co-authored-by|assisted-by|generated-by|ai-assisted-by)[ \t]*:[^\n]*{TOOL_NAMES})"
    rf"|(?:^[^a-z0-9\n]{{0,8}}(?:generated|created|written|authored|produced|assisted)[ \t]+(?:by|with|using|via)[ \t]+(?:an?[ \t]+|the[ \t]+)?\[?{TOOL_NAMES}(?:[^a-z0-9\n-]|\n?$))"
    r"|(?:claude\.ai/(?:code|share)|claude\.com/claude-code|chatgpt\.com/(?:codex|share))"
    r"|(?:noreply@(?:anthropic|openai)\.com)"
    rf"|(?:^merge pull request #[0-9]+ from [^ \n]+/{AGENT_NAMES}/)"
)

EMAIL = r"(?i)@(?:anthropic|openai)\.com\n?$"
LABEL = r"(?i)(?:claude|anthropic|codex|openai)"
# --- End vendored attribution patterns ---

REF_RE = re.compile(REF_NAME)
MSG_RE = re.compile(COMMIT_MESSAGE)
EMAIL_RE = re.compile(EMAIL)
LABEL_RE = re.compile(LABEL)

SCISSORS_MARKER = "------------------------ >8 ------------------------"
COMMENT_PREFIX = "#"
API_PAGE_SIZE = 100
API_TIMEOUT_SECONDS = 30
EXIT_CLEAN = 0
EXIT_VIOLATION = 1
EXIT_ERROR = 2


class GuardError(RuntimeError):
    """Raised when the guard cannot gather the data it must verify."""


@dataclass(frozen=True)
class Violation:
    """One metadata field that matched an attribution pattern."""

    field: str
    match: str

    def render(self) -> str:
        """Returns a single-line description naming the offending field."""
        return f"{self.field}: matched {self.match!r}"


@dataclass(frozen=True)
class Commit:
    """Commit metadata subject to the attribution policy."""

    sha: str
    message: str
    author_email: str
    committer_email: str


@dataclass(frozen=True)
class RefUpdate:
    """One ref update as reported by git's pre-push protocol."""

    local_sha: str
    remote_ref: str
    remote_sha: str


@dataclass(frozen=True)
class PullRequestSnapshot:
    """Pull request metadata refetched from the API at check time."""

    title: str
    body: str
    head_ref: str
    labels: tuple[str, ...]
    commits: tuple[Commit, ...]


def _search(pattern: re.Pattern[str], value: str, field: str) -> list[Violation]:
    match = pattern.search(value)
    return [] if match is None else [Violation(field=field, match=match.group(0).strip())]


def check_ref_name(name: str, field: str) -> list[Violation]:
    """Checks one branch or tag name."""
    return _search(REF_RE, name, field)


def check_message(text: str, field: str) -> list[Violation]:
    """Checks one full message text (commit message, PR title, or PR body)."""
    return _search(MSG_RE, text, field)


def check_email(email: str, field: str) -> list[Violation]:
    """Checks one author or committer email."""
    return _search(EMAIL_RE, email, field)


def check_label(name: str) -> list[Violation]:
    """Checks one pull request label name."""
    return _search(LABEL_RE, name, f"label {name!r}")


def check_commit(commit: Commit) -> list[Violation]:
    """Checks a commit's message plus its author and committer emails."""
    prefix = f"commit {commit.sha[:12]}"
    return [
        *check_message(commit.message, f"{prefix} message"),
        *check_email(commit.author_email, f"{prefix} author email"),
        *check_email(commit.committer_email, f"{prefix} committer email"),
    ]


def check_pull_request(snapshot: PullRequestSnapshot) -> list[Violation]:
    """Checks every attribution-bearing field of one pull request."""
    violations = [
        *check_message(snapshot.title, "pull request title"),
        *check_message(snapshot.body, "pull request body"),
        *check_ref_name(snapshot.head_ref, f"head ref {snapshot.head_ref!r}"),
    ]
    for label in snapshot.labels:
        violations.extend(check_label(label))
    for commit in snapshot.commits:
        violations.extend(check_commit(commit))
    return violations


def strip_commit_message_comments(text: str) -> str:
    """Drops what git strips before committing: comment lines and the verbose-diff tail."""
    kept: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith(COMMENT_PREFIX) and SCISSORS_MARKER in line:
            break
        if line.startswith(COMMENT_PREFIX):
            continue
        kept.append(line)
    return "".join(kept)


def _is_null_sha(sha: str) -> bool:
    return bool(sha) and set(sha) == {"0"}


def _git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _commit_exists(sha: str) -> bool:
    return _git(["cat-file", "-e", f"{sha}^{{commit}}"]).returncode == 0


def read_commits(revisions: Sequence[str]) -> list[Commit]:
    """Reads commit metadata for a `git log` revision selection."""
    completed = _git(["log", "-z", "--format=%H%n%ae%n%ce%n%B", *revisions, "--"])
    if completed.returncode != 0:
        raise GuardError(f"git log {' '.join(revisions)} failed: {completed.stderr.strip()}")
    commits: list[Commit] = []
    for record in completed.stdout.split("\0"):
        record = record.lstrip("\n")
        if not record:
            continue
        sha, author_email, committer_email, message = (record.split("\n", 3) + ["", "", ""])[:4]
        commits.append(Commit(sha, message, author_email, committer_email))
    return commits


def _remote_exclusion(remote: str | None) -> str:
    configured = set(_git(["remote"]).stdout.split())
    return f"--remotes={remote}" if remote in configured else "--remotes"


def _introduced_revisions(update: RefUpdate, remote: str | None) -> list[str]:
    if not _is_null_sha(update.remote_sha) and _commit_exists(update.remote_sha):
        return [f"{update.remote_sha}..{update.local_sha}"]
    return [update.local_sha, "--not", _remote_exclusion(remote)]


def parse_pre_push_lines(lines: Sequence[str]) -> list[RefUpdate]:
    """Parses `<local-ref> <local-sha> <remote-ref> <remote-sha>` lines from git."""
    updates: list[RefUpdate] = []
    for raw_line in lines:
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise GuardError(f"Malformed pre-push line: {raw_line.strip()!r}")
        _, local_sha, remote_ref, remote_sha = parts
        updates.append(RefUpdate(local_sha=local_sha, remote_ref=remote_ref, remote_sha=remote_sha))
    return updates


def check_ref_updates(updates: Sequence[RefUpdate], remote: str | None) -> list[Violation]:
    """Checks pushed ref names and every commit each update introduces."""
    violations: list[Violation] = []
    for update in updates:
        if _is_null_sha(update.local_sha):
            continue
        violations.extend(check_ref_name(update.remote_ref, f"pushed ref {update.remote_ref!r}"))
        for commit in read_commits(_introduced_revisions(update, remote)):
            violations.extend(check_commit(commit))
    return violations


def _pre_commit_ref_update() -> RefUpdate | None:
    to_ref = os.environ.get("PRE_COMMIT_TO_REF", "")
    if not to_ref:
        return None
    return RefUpdate(
        local_sha=to_ref,
        remote_ref=os.environ.get("PRE_COMMIT_REMOTE_BRANCH", ""),
        remote_sha=os.environ.get("PRE_COMMIT_FROM_REF", ""),
    )


def check_unpushed_head() -> list[Violation]:
    """Fallback for `pre-commit run`: the current branch and commits on no remote."""
    if _git(["rev-parse", "--verify", "--quiet", "HEAD"]).returncode != 0:
        return []
    violations: list[Violation] = []
    branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"])
    if branch.returncode == 0:
        name = branch.stdout.strip()
        violations.extend(check_ref_name(name, f"current branch {name!r}"))
    for commit in read_commits(["HEAD", "--not", "--remotes"]):
        violations.extend(check_commit(commit))
    return violations


def _api_get(url: str, token: str) -> object:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "attribution-guard",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            return cast(object, json.load(response))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise GuardError(f"GitHub API request failed for {url}: {error}") from error


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GuardError(f"Unexpected API payload: {context} is not an object.")
    return cast(dict[str, object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise GuardError(f"Unexpected API payload: {context} is not a string.")
    return value


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise GuardError(f"Unexpected API payload: {context} is not a list.")
    return cast(list[object], value)


def _api_commit(item: object) -> Commit:
    entry = _mapping(item, "commit entry")
    sha = _string(entry.get("sha"), "commit sha")
    details = _mapping(entry.get("commit"), f"commit {sha}")
    return Commit(
        sha=sha,
        message=_string(details.get("message"), f"commit {sha} message"),
        author_email=_string(_mapping(details.get("author"), "author").get("email"), "email"),
        committer_email=_string(
            _mapping(details.get("committer"), "committer").get("email"), "email"
        ),
    )


def fetch_pull_request(api_url: str, repo: str, number: int, token: str) -> PullRequestSnapshot:
    """Refetches title, body, head ref, labels, and every commit of one pull request."""
    base_url = f"{api_url.rstrip('/')}/repos/{repo}/pulls/{number}"
    pull = _mapping(_api_get(base_url, token), "pull request")
    raw_body = pull.get("body")
    labels = tuple(
        _string(_mapping(label, "label").get("name"), "label name")
        for label in _sequence(pull.get("labels"), "labels")
    )
    expected_commits = pull.get("commits")
    if not isinstance(expected_commits, int):
        raise GuardError("Unexpected API payload: pull request commit count is missing.")

    commits: list[Commit] = []
    page = 1
    while True:
        batch = _sequence(
            _api_get(f"{base_url}/commits?per_page={API_PAGE_SIZE}&page={page}", token),
            "commits page",
        )
        commits.extend(_api_commit(item) for item in batch)
        if len(batch) < API_PAGE_SIZE:
            break
        page += 1
    if len(commits) != expected_commits:
        raise GuardError(
            f"Fetched {len(commits)} of {expected_commits} pull request commits; "
            "refusing to pass an incomplete check."
        )
    return PullRequestSnapshot(
        title=_string(pull.get("title"), "title"),
        body="" if raw_body is None else _string(raw_body, "body"),
        head_ref=_string(_mapping(pull.get("head"), "head").get("ref"), "head ref"),
        labels=labels,
        commits=tuple(commits),
    )


def _report(violations: Sequence[Violation], scope: str) -> int:
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    for violation in violations:
        line = f"attribution guard: {violation.render()}"
        print(f"::error title=Attribution Guard::{line}" if annotate else line, file=sys.stderr)
    if violations:
        print(f"attribution guard: {len(violations)} violation(s) in {scope}.", file=sys.stderr)
        return EXIT_VIOLATION
    print(f"attribution guard: {scope}: clean.")
    return EXIT_CLEAN


def _run_commit_msg(args: argparse.Namespace) -> int:
    text = Path(args.message_file).read_text(encoding="utf-8", errors="replace")
    return _report(check_message(strip_commit_message_comments(text), "commit message"), "commit")


def _run_pre_push(args: argparse.Namespace) -> int:
    remote = cast(str | None, args.remote) or os.environ.get("PRE_COMMIT_REMOTE_NAME")
    if args.stdin:
        updates = parse_pre_push_lines(sys.stdin.read().splitlines())
        return _report(check_ref_updates(updates, remote), "push")
    pre_commit_update = _pre_commit_ref_update()
    if pre_commit_update is not None:
        return _report(check_ref_updates([pre_commit_update], remote), "push")
    return _report(check_unpushed_head(), "unpushed commits")


def _run_push(args: argparse.Namespace) -> int:
    before = cast(str, args.before)
    after = cast(str, args.after)
    if _is_null_sha(before) or not _commit_exists(before):
        print(f"attribution guard: base {before!r} unavailable; checking {after} only.")
        revisions = ["-1", after]
    else:
        revisions = [f"{before}..{after}"]
    violations: list[Violation] = []
    for commit in read_commits(revisions):
        violations.extend(check_commit(commit))
    return _report(violations, f"push {before[:12]}..{after[:12]}")


def _run_pull_request(args: argparse.Namespace) -> int:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise GuardError("GITHUB_TOKEN or GH_TOKEN is required for pull-request mode.")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    snapshot = fetch_pull_request(api_url, cast(str, args.repo), cast(int, args.number), token)
    scope = f"pull request #{args.number} ({len(snapshot.commits)} commits)"
    return _report(check_pull_request(snapshot), scope)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    modes = parser.add_subparsers(dest="mode", required=True)

    commit_msg = modes.add_parser("commit-msg", help="Check a commit message file.")
    commit_msg.add_argument("message_file")
    commit_msg.set_defaults(handler=_run_commit_msg)

    pre_push = modes.add_parser("pre-push", help="Check refs and commits about to be pushed.")
    pre_push.add_argument("remote", nargs="?", default=None)
    pre_push.add_argument("url", nargs="?", default=None)
    pre_push.add_argument("--stdin", action="store_true", help="Read git pre-push ref lines.")
    pre_push.set_defaults(handler=_run_pre_push)

    push = modes.add_parser("push", help="Audit commits pushed to a branch.")
    push.add_argument("--before", required=True)
    push.add_argument("--after", required=True)
    push.set_defaults(handler=_run_push)

    pull_request = modes.add_parser("pull-request", help="Check pull request metadata.")
    pull_request.add_argument("--repo", required=True)
    pull_request.add_argument("--number", required=True, type=int)
    pull_request.set_defaults(handler=_run_pull_request)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs one guard mode and returns its process exit status."""
    args = _build_parser().parse_args(argv)
    try:
        return cast(int, args.handler(args))
    except GuardError as error:
        print(f"attribution guard: error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
