#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
pins=constraints/consumer-py312.txt
resolved=$(uv pip compile --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
  --constraint "$pins" pyproject.toml "$pins")
if grep -Eq '^torch==' <<< "$resolved"; then
  echo "The core resolution unexpectedly includes PyTorch." >&2
  exit 1
fi
printf '%s\n' "$resolved"
