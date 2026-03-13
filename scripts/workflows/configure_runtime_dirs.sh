#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/workflows/configure_runtime_dirs.sh [options]

Options:
  --max-workers <count>      Value exported as SER_MAX_WORKERS (default: 1).
  --models-dir <path>        Directory exported as SER_MODELS_DIR.
  --data-dir <path>          Directory exported as SER_DATA_DIR.
  --cache-dir <path>         Directory exported as SER_CACHE_DIR.
  --transcripts-dir <path>   Directory exported as SER_TRANSCRIPTS_DIR.
  -h, --help                 Show this help text.
EOF
}

if [[ -z "${GITHUB_ENV:-}" ]]; then
  printf 'GITHUB_ENV must be set when configuring workflow runtime directories.\n' >&2
  exit 2
fi

runner_temp="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
max_workers="1"
models_dir="$runner_temp/ser-models"
data_dir="$runner_temp/ser-data"
cache_dir="$runner_temp/ser-cache"
transcripts_dir="$runner_temp/ser-transcripts"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-workers)
      max_workers="$2"
      shift 2
      ;;
    --models-dir)
      models_dir="$2"
      shift 2
      ;;
    --data-dir)
      data_dir="$2"
      shift 2
      ;;
    --cache-dir)
      cache_dir="$2"
      shift 2
      ;;
    --transcripts-dir)
      transcripts_dir="$2"
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

mkdir -p "$models_dir" "$data_dir" "$cache_dir" "$transcripts_dir"

{
  printf 'SER_MAX_WORKERS=%s\n' "$max_workers"
  printf 'SER_MODELS_DIR=%s\n' "$models_dir"
  printf 'SER_DATA_DIR=%s\n' "$data_dir"
  printf 'SER_CACHE_DIR=%s\n' "$cache_dir"
  printf 'SER_TRANSCRIPTS_DIR=%s\n' "$transcripts_dir"
} >> "$GITHUB_ENV"

