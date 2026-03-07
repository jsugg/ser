#!/usr/bin/env bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  printf 'Missing required command: git\n' >&2
  exit 2
fi

if ! command -v make >/dev/null 2>&1; then
  printf 'Missing required command: make\n' >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  printf 'Not inside a git repository.\n' >&2
  exit 2
}

cd "$repo_root"

snapshot_repo_state() {
  local destination="$1"
  {
    git diff --binary --no-ext-diff -- .
    printf '\n--INDEX--\n'
    git diff --cached --binary --no-ext-diff -- .
    printf '\n--UNTRACKED--\n'
    git ls-files --others --exclude-standard
  } > "$destination"
}

before_state="$(mktemp)"
after_state="$(mktemp)"
cleanup() {
  rm -f "$before_state" "$after_state"
}
trap cleanup EXIT

snapshot_repo_state "$before_state"

if ! make prepush; then
  printf '\n[pre-push] Auto-fix completed, but non-fixable issues remain. Aborting push.\n' >&2
  exit 1
fi

snapshot_repo_state "$after_state"

if ! cmp -s "$before_state" "$after_state"; then
  printf '\n[pre-push] Auto-fixes changed the working tree. Review, stage, and commit those changes before pushing.\n' >&2
  git status --short >&2
  exit 1
fi
