#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_compatible_env.sh [options]

Options:
  --python <version>       Override Python version (default: platform-aware).
  --extra <group>          Include an optional dependency group; repeatable.
  --no-dev                 Skip development dependencies.
  --skip-git-hooks         Do not install git hooks even when `.git/` exists.
  --skip-ffmpeg-check      Do not fail if ffmpeg is missing on PATH.
  --dry-run                Print planned commands without executing them.
  -h, --help               Show this help text.

Environment overrides:
  SER_SETUP_PYTHON         Same as --python.
  SER_SETUP_EXTRAS         Comma-separated optional dependency groups.
  SER_SETUP_INCLUDE_DEV    true/false (default: true).
  SER_SETUP_INSTALL_GIT_HOOKS true/false (default: true).
  SER_SETUP_CHECK_FFMPEG   true/false (default: true).
  SER_SETUP_DRY_RUN        true/false (default: false).
EOF
}

normalize_bool() {
  local raw="$1"
  local name="$2"
  local lowered
  lowered="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    true|1|yes|y)
      printf 'true'
      ;;
    false|0|no|n)
      printf 'false'
      ;;
    *)
      printf 'Invalid boolean value for %s: %s\n' "$name" "$raw" >&2
      exit 2
      ;;
  esac
}

run_cmd() {
  if [[ "$dry_run" == "true" ]]; then
    printf '[dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

array_contains() {
  local needle="$1"
  shift

  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done

  return 1
}

for required in uname uv; do
  if ! command -v "$required" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$required" >&2
    exit 2
  fi
done

os_name="$(uname -s)"
arch_name="$(uname -m)"
default_python="3.13"

python_version="${SER_SETUP_PYTHON:-$default_python}"
include_dev="$(normalize_bool "${SER_SETUP_INCLUDE_DEV:-true}" "SER_SETUP_INCLUDE_DEV")"
install_git_hooks="$(normalize_bool "${SER_SETUP_INSTALL_GIT_HOOKS:-true}" "SER_SETUP_INSTALL_GIT_HOOKS")"
check_ffmpeg="$(normalize_bool "${SER_SETUP_CHECK_FFMPEG:-true}" "SER_SETUP_CHECK_FFMPEG")"
dry_run="$(normalize_bool "${SER_SETUP_DRY_RUN:-false}" "SER_SETUP_DRY_RUN")"
extras_csv="${SER_SETUP_EXTRAS:-}"
extras=()

if [[ -n "$extras_csv" ]]; then
  IFS=',' read -r -a extras <<< "$extras_csv"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --python\n' >&2
        exit 2
      fi
      python_version="$2"
      shift 2
      ;;
    --extra)
      if [[ $# -lt 2 ]]; then
        printf 'Missing value for --extra\n' >&2
        exit 2
      fi
      extras+=("$2")
      shift 2
      ;;
    --no-dev)
      include_dev="false"
      shift
      ;;
    --skip-git-hooks)
      install_git_hooks="false"
      shift
      ;;
    --skip-ffmpeg-check)
      check_ffmpeg="false"
      shift
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#extras[@]} -eq 0 ]]; then
  extras=(full)
fi

sync_args=(--python "$python_version")
for extra in "${extras[@]}"; do
  sync_args+=(--extra "$extra")
done

if [[ "$include_dev" == "true" ]] && ! array_contains "dev" "${extras[@]}"; then
  sync_args+=(--extra dev)
fi

printf '[setup] platform: %s/%s\n' "$os_name" "$arch_name"
printf '[setup] python: %s\n' "$python_version"
printf '[setup] extras: %s\n' "${extras[*]}"
printf '[setup] include dev tools: %s\n' "$include_dev"
printf '[setup] install git hooks: %s\n' "$install_git_hooks"
printf '[setup] ffmpeg check: %s\n' "$check_ffmpeg"

if [[ "$os_name" == "Darwin" && "$arch_name" == "x86_64" ]]; then
  if [[ "$python_version" == "3.13" || "$python_version" == 3.13.* ]]; then
    printf '[setup] note: Darwin x86_64 + Python 3.13 is currently partial (fast-profile oriented) and not an officially supported full runtime lane.\n'
    printf '[setup] note: medium/accurate/accurate-research runtime requires torch/transformers wheels not currently published for this platform tag.\n'
  fi
fi

run_cmd uv python install "$python_version"
run_cmd uv sync "${sync_args[@]}"

if [[ "$install_git_hooks" == "true" && "$include_dev" == "true" && -d .git ]]; then
  run_cmd ./scripts/install_git_hooks.sh
fi

if [[ "$check_ffmpeg" == "true" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'ffmpeg was not found on PATH. Install ffmpeg before running inference.\n' >&2
    exit 2
  fi
fi

printf '[setup] completed successfully\n'
