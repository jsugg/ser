#!/usr/bin/env bash
set -euo pipefail

# Import-layer contract lane for public API boundary enforcement.
mapfile -t IMPORT_LINT_PATHS < <(
  find ser -path 'ser/_internal' -prune -o -name '*.py' -print | sort
)

uv run --frozen --extra dev ruff check --select TID251 "${IMPORT_LINT_PATHS[@]}"
uv run --frozen --extra dev pytest -q \
  tests/suites/integration/architecture/test_api_import_boundary.py \
  tests/suites/integration/architecture/test_import_lint_policy.py
