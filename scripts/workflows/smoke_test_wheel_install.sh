#!/usr/bin/env bash
set -euo pipefail

wheel_glob="${1:-dist/*.whl}"
shopt -s nullglob
wheels=($wheel_glob)
shopt -u nullglob

if [[ ${#wheels[@]} -eq 0 ]]; then
  printf 'No wheels matched %s\n' "$wheel_glob" >&2
  exit 2
fi

python -m venv .pkg-smoke
. .pkg-smoke/bin/activate
python -m pip install --upgrade pip
pip install --no-deps "${wheels[@]}"

tmp_dir="$(mktemp -d)"
cd "$tmp_dir"

python - <<'PY'
import importlib.metadata as md
import ser

print(f"Installed ser version: {md.version('ser')}")
print(f"Imported from: {ser.__file__}")
print(f"Exports: {', '.join(ser.__all__)}")
PY
