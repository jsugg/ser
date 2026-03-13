#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/workflows/run_profile_smoke.sh --profile <name> [options]

Options:
  --profile <name>          Profile passed to `ser --train` and `ser --file`.
  --python <version>        Optional Python version passed to uv run.
  --uv-extra <group>        Optional uv extra; repeatable.
  --sample-file <path>      Sample file used for prediction (default: sample.wav).
  -h, --help                Show this help text.
EOF
}

profile=""
python_version=""
sample_file="sample.wav"
uv_extras=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile="$2"
      shift 2
      ;;
    --python)
      python_version="$2"
      shift 2
      ;;
    --uv-extra)
      uv_extras+=("$2")
      shift 2
      ;;
    --sample-file)
      sample_file="$2"
      shift 2
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

if [[ -z "$profile" ]]; then
  printf 'Missing required --profile option.\n' >&2
  usage >&2
  exit 2
fi

uv_run_args=()
if [[ -n "$python_version" ]]; then
  uv_run_args+=(--python "$python_version")
fi
if [[ ${#uv_extras[@]} -gt 0 ]]; then
  for extra in "${uv_extras[@]}"; do
    uv_run_args+=(--extra "$extra")
  done
fi

run_uv() {
  if [[ ${#uv_run_args[@]} -gt 0 ]]; then
    uv run "${uv_run_args[@]}" "$@"
    return 0
  fi
  uv run "$@"
}

run_uv ser --train --profile "$profile"
run_uv ser --file "$sample_file" --profile "$profile"
