#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/workflows/setup_validation_environment.sh --python <version> [options]

Options:
  --python <version>        Python version passed to setup_compatible_env.sh.
  --accurate-research       Include the full extra instead of medium.
  --no-dev                  Skip development dependencies.
  -h, --help                Show this help text.
EOF
}

python_version=""
run_accurate_research="false"
include_dev="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_version="$2"
      shift 2
      ;;
    --accurate-research)
      run_accurate_research="true"
      shift
      ;;
    --no-dev)
      include_dev="false"
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

if [[ -z "$python_version" ]]; then
  printf 'Missing required --python option.\n' >&2
  usage >&2
  exit 2
fi

setup_args=(--python "$python_version" --skip-git-hooks)
if [[ "$run_accurate_research" == "true" ]]; then
  setup_args+=(--extra full)
else
  setup_args+=(--extra medium)
fi
if [[ "$include_dev" == "false" ]]; then
  setup_args+=(--no-dev)
fi

./scripts/setup_compatible_env.sh "${setup_args[@]}"

