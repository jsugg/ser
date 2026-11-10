#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d .git ]]; then
  printf 'Not inside a git repository root (missing .git directory).\n' >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  printf 'Missing required command: uv\n' >&2
  exit 2
fi

mkdir -p .git/hooks
cat > .git/hooks/pre-push <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
# git passes the pushed refs on stdin; only the attribution guard consumes them.
uv run --frozen --extra dev python ./scripts/attribution_guard.py pre-push --stdin "$@"
exec bash ./scripts/run_prepush_gate.sh
EOF
chmod +x .git/hooks/pre-push

uv run --frozen --extra dev pre-commit install --hook-type commit-msg

printf '[hooks] installed .git/hooks/pre-push -> attribution guard + bash ./scripts/run_prepush_gate.sh\n'
printf '[hooks] installed .git/hooks/commit-msg -> pre-commit commit-msg stage (attribution guard)\n'
