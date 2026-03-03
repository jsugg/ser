# Contributing

Thanks for contributing to `ser`.

## Development Setup

```bash
git clone https://github.com/jsugg/ser/
cd ser
./scripts/setup_compatible_env.sh
```

Platform/setup notes and runtime compatibility details remain in `README.md`.

## Local Quality Gates

Before opening a PR, run:

```bash
uv run --extra dev pre-commit run --all-files --hook-stage pre-push
uv run --extra dev pytest -q
```

## CI and Validation

- Default CI is defined in `.github/workflows/ci.yml`.
- Slow and hardware-specific lanes are manual (`workflow_dispatch`).
- Hardware validation instructions are documented in
  [`docs/ci/hardware-validation.md`](docs/ci/hardware-validation.md).

## Pull Requests

- Keep diffs small and focused.
- Add/adjust tests for behavior changes.
- Avoid mixing unrelated refactors with functional fixes.
