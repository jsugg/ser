#!/usr/bin/env bash
set -euo pipefail

# Import-layer contract lane for public API boundary enforcement.
readonly IMPORT_LINT_PATHS=(
  ser
  tests
)

uv run --frozen --extra dev ruff check --select TID251 "${IMPORT_LINT_PATHS[@]}"
uv run --frozen --extra dev pytest -q \
  tests/test_api_import_boundary.py \
  tests/test_import_lint_policy.py
