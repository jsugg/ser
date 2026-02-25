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
exec uv run pre-commit run --all-files --hook-stage pre-push
EOF
chmod +x .git/hooks/pre-push

printf '[hooks] installed .git/hooks/pre-push -> uv run pre-commit run --all-files --hook-stage pre-push\n'
