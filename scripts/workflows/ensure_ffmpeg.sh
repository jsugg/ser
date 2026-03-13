#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/workflows/ensure_ffmpeg.sh --mode <apt-install|brew-install|required>
EOF
}

mode=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="$2"
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

if [[ -z "$mode" ]]; then
  printf 'Missing required --mode option.\n' >&2
  usage >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  case "$mode" in
    apt-install)
      sudo apt-get update
      sudo apt-get install -y ffmpeg
      ;;
    brew-install)
      brew install ffmpeg
      ;;
    required)
      printf 'ffmpeg is required on this runner.\n' >&2
      exit 1
      ;;
    *)
      printf 'Unsupported ffmpeg mode: %s\n' "$mode" >&2
      exit 2
      ;;
  esac
fi

ffmpeg -version

