#!/usr/bin/env bash
set -euo pipefail

# Validation workflows train against the synthetic RAVDESS dataset, so they must
# persist the required dataset acknowledgements before invoking `ser --train`.
uv run "$@" ser configure \
  --accept-dataset-policy noncommercial \
  --accept-dataset-license cc-by-nc-sa-4.0 \
  --persist
