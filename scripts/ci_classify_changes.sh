#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/ci_classify_changes.sh <github-output-path>

Environment:
  CI_EVENT_NAME   GitHub event name (for example: pull_request, push).
  CI_BASE_SHA     Base commit SHA for pull_request comparisons.
  CI_HEAD_SHA     Head commit SHA for pull_request comparisons.
EOF
}

write_output() {
  local destination="$1"
  local run_full="$2"
  local docs_only="$3"
  local reason="$4"

  {
    printf 'run_full=%s\n' "$run_full"
    printf 'docs_only=%s\n' "$docs_only"
    printf 'reason=%s\n' "$reason"
  } >> "$destination"
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

readonly output_path="$1"
readonly event_name="${CI_EVENT_NAME:-}"

if [[ -z "$event_name" ]]; then
  printf 'CI_EVENT_NAME is required.\n' >&2
  exit 2
fi

if [[ "$event_name" != "pull_request" ]]; then
  write_output "$output_path" "true" "false" "non_pull_request"
  exit 0
fi

readonly base_sha="${CI_BASE_SHA:-}"
readonly head_sha="${CI_HEAD_SHA:-}"

if [[ -z "$base_sha" || -z "$head_sha" ]]; then
  printf 'CI_BASE_SHA and CI_HEAD_SHA are required for pull_request events.\n' >&2
  exit 2
fi

changed_files=()
while IFS= read -r path; do
  changed_files+=("$path")
done < <(git diff --name-only "$base_sha" "$head_sha" --)

if [[ ${#changed_files[@]} -eq 0 ]]; then
  write_output "$output_path" "true" "false" "empty_diff"
  exit 0
fi

docs_only="true"
for path in "${changed_files[@]}"; do
  case "$path" in
    README.md|LICENSE|pyproject.toml|uv.lock|Makefile|.pre-commit-config.yaml|.github/workflows/*|scripts/*|ser/*|tests/*)
      docs_only="false"
      break
      ;;
    .github/assets/*|.github/CODEOWNERS|docs/*|*.md)
      ;;
    *)
      docs_only="false"
      break
      ;;
  esac
done

if [[ "$docs_only" == "true" ]]; then
  write_output "$output_path" "false" "true" "docs_only_pull_request"
  exit 0
fi

write_output "$output_path" "true" "false" "full_ci_required"
