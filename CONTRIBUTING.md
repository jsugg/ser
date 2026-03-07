# Contributing

Thanks for contributing to `ser`.

## Development Setup
```bash
git clone https://github.com/jsugg/ser/
cd ser
./scripts/setup_compatible_env.sh
```

Setup installs runtime + dev tooling and installs the git pre-push hook when `.git/` is present.

## Local Quality Workflow
Recommended flow before opening a PR:

1. Run the local pre-push gate (auto-fix + verify).
```bash
make prepush
```
2. Run tests.
```bash
make test
```

`make prepush` may modify files (formatter auto-fixes). Re-stage and commit any changes before pushing.

Equivalent explicit gates:
```bash
make fmt       # pyupgrade --py312-plus + ruff --fix + isort + black
make lint      # ruff + black --check + isort --check-only
make type      # mypy + pyright (pythonversion 3.12)
make test      # pytest -q
make prepush-check      # canonical pre-push hook command
make topology-contracts # PR-901..PR-903 structural ownership contract lane
make import-lint        # API boundary import lint lane (TID251 banned-api policy)
```

Git pre-push hook and CI code-quality lane command:
```bash
uv run --frozen --extra dev pre-commit run --all-files --hook-stage pre-push
```

## Pre-push Hooks
Install or refresh git hooks manually:
```bash
./scripts/install_git_hooks.sh
```

The generated `.git/hooks/pre-push` runs the same canonical pre-push command above.

## Boundary Commands (When Required)
Run the boundary lane whenever your change touches `ser/api.py`, `ser/_internal/api/*`, `ser/__main__.py`, or boundary contract tests:

```bash
make import-lint
uv run pytest -q tests/test_import_lint_policy.py tests/test_api_import_boundary.py tests/test_api.py tests/test_cli.py
```

## CI Topology
Default CI is defined in `.github/workflows/ci.yml`.

Quality and validation lanes:
- `changes`: classifies pull requests so docs-only PRs can skip heavy jobs while `push` to `main` still runs the full pipeline.
- `code-quality`: runs pre-push stage hooks (ruff/black/isort/mypy/pyright), with formatter hooks in check-only mode.
- `resolve`: validates lock/extras resolution for Python 3.12 and 3.13.
- `tests`: runs pytest matrix on Python 3.12 and 3.13.
- `contract-gates`: deterministic contract lane on Python 3.12 (structural ownership gates + API boundary import-lint gate + transcription benchmark contract test).
- `build`: package build + metadata/wheel smoke checks.

## Hardware Validation
Hardware-specific workflows are manual (`workflow_dispatch`):
- [docs/ci/hardware-validation.md](docs/ci/hardware-validation.md)

Policy note:
- Full support target: `darwin-x86_64-macos13-python3.12`.
- Partial support target (fast profile only): `darwin-x86_64-macos13-python3.13`.
- GitHub Actions uses `macos-15` hosted runners because `macos-13` hosted runners are unavailable.
- macOS13 support evidence is collected through local validation.

## Pull Request Expectations
- Keep diffs focused; avoid mixing unrelated refactors with behavior changes.
- Add or update tests for every behavior change.
- Update docs when changing developer workflow, CI expectations, runtime policy, or profile contracts.
- If changing hardware/runtime policy, include evidence from the relevant workflow or local validation lane.

## Architecture Guardrails
- Keep CLI adapters thin (`ser/__main__.py`, `ser/data/cli.py`): parse flags, delegate to orchestration.
- Keep runtime command policy out of `ser/__main__.py`; route through `ser.api` command wrappers (`run_restricted_backend_cli_gate`, `run_startup_preflight_cli_gate`, `run_transcription_runtime_calibration_command`, `run_training_command`, `run_inference_command`).
- Keep public integration boundary in `ser/api.py`; avoid exposing internals directly.
- Keep dataset-specific acquisition/transforms in `ser/data/*` and orchestrate through `ser/data/dataset_prepare.py`.
- Keep tests layered by touched surface: helper unit tests, orchestration tests, and CLI/API boundary tests.
