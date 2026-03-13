#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/workflows/sync_validation_dependencies.sh --python <version> [options]

Options:
  --python <version>        Python version passed to uv sync.
  --accurate-research       Include the full extra instead of medium.
  --with-dev                Include the dev extra.
  -h, --help                Show this help text.
EOF
}

python_version=""
run_accurate_research="false"
include_dev="false"

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
    --with-dev)
      include_dev="true"
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

sync_args=(--frozen --python "$python_version")
if [[ "$include_dev" == "true" ]]; then
  sync_args+=(--extra dev)
fi
if [[ "$run_accurate_research" == "true" ]]; then
  sync_args+=(--extra full)
else
  sync_args+=(--extra medium)
fi

uv sync "${sync_args[@]}"

