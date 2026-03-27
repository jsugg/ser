# Compatibility Matrix (Live Snapshot)

Date initialized: 2026-02-19
Last updated: 2026-03-27
Purpose: Track compatibility coverage across Python versions, dependency extras, and runtime profiles.
Evidence source:
1. `.github/workflows/darwin-x86_64-validation.yml`
2. `.github/workflows/linux-python-3_13-cli-validation.yml`
3. `README.md`

## Matrix

Platform policy note:
1. Darwin Intel (`x86_64`) + Python `3.13` is currently **partial compatibility** (fast-profile oriented) and is not an officially supported full runtime lane.
2. Darwin Intel (`x86_64`) non-fast runtime (`medium`, `accurate`, `accurate-research`) on Python `3.13` is constrained by upstream wheel availability for `torch`/`transformers`; non-fast smoke evidence is captured on Darwin `3.12` and Linux `3.13`.
3. The supported local setup path is `./scripts/setup_compatible_env.sh`.
4. GitHub Darwin validation remains a Python `3.12` smoke lane (`.github/workflows/darwin-x86_64-validation.yml`) for hosted-runner parity and artifact capture.
5. Darwin Intel CI lane now has branch evidence from `.github/workflows/darwin-x86_64-validation.yml`:
   - run `22403895369` (`completed/success`, `synth_test_data_creation`, head SHA `aee2c3cd1b2373c6a8079af79fe02e938b868dda`).
   - `main` evidence:
     - run `22414637382` (`completed/success`, `main`, head SHA `27618a56eb9514d5df5553f7f1885c7d4f640689`).
6. Non-Darwin Python `3.13` CLI train/predict evidence lane is `.github/workflows/linux-python-3_13-cli-validation.yml`.
7. Non-Darwin Python `3.13` lane now has branch evidence:
   - run `22403895371` (`completed/success`, `synth_test_data_creation`, head SHA `aee2c3cd1b2373c6a8079af79fe02e938b868dda`).
   - `main` evidence with accurate-research enabled:
     - run `22414258948` (`completed/success`, `main`, head SHA `27618a56eb9514d5df5553f7f1885c7d4f640689`).
8. Runtime default policy is ratified:
   - `fast` is default.
   - `medium`, `accurate`, `accurate-research` remain opt-in lanes.

| Python | Install Mode | Profile | CLI Train | CLI Predict | Tests | Type/Lint | Notes |
|---|---|---|---|---|---|---|---|
| 3.12 | base | fast | pass | pass | pass | pass | Closeout refresh in Iteration 42: `uv run ruff check ser tests` pass, `uv run mypy ser tests` pass, `uv run --extra dev pyright ser tests` `0 errors/0 warnings`, `uv run --extra dev pytest -q` `204 passed`, `uv build` pass, `uvx twine check dist/*` pass. Benchmark evidence remains from Iteration 18 (`mean=1.544s`, `p95=2.963s`). |
| 3.13 | base | fast | pass | pass | pass | pass | `main` evidence captured in run `22414258948` (`.github/workflows/linux-python-3_13-cli-validation.yml`, head SHA `27618a56eb9514d5df5553f7f1885c7d4f640689`). Warning policy from Iteration 43 remains unchanged (optional-import warnings non-blocking when `errors=0`). |
| 3.12 | medium extras | medium | pass | blocked | pass | pass | Runtime path is implemented and validated with profile-matched artifacts. CLI profile mode now resolves profile-specific artifact defaults and metadata-aware candidate selection across discovered `ser_model*.pkl|*.skops`; predict remains blocked until at least one compatible medium artifact exists. |
| 3.13 | medium extras | medium | pass | pass | pass | pass | `main` evidence captured in run `22414258948` with train+predict smoke passing for `--profile medium`; warning-policy posture unchanged. Darwin x86_64 Python 3.13 remains partial/fast-oriented and not an officially supported non-fast runtime lane. |
| 3.12 | accurate extras | accurate | pass | blocked | pass | pass | Accurate path is implemented and artifact-backed predict evidence exists. CLI profile mode now resolves profile-specific artifact defaults and metadata-aware candidate selection across discovered `ser_model*.pkl|*.skops`; predict remains blocked when no compatible accurate artifact is available or runtime budgets/timeouts are exceeded for selected backend/model. |
| 3.13 | accurate extras | accurate | pass | pass | pass | pass | `main` evidence captured in run `22414258948` with train+predict smoke passing for `--profile accurate`; 3.13 warning policy remains finalized (`pyright` warnings non-blocking when `errors=0`). Darwin x86_64 Python 3.13 remains partial/fast-oriented and not an officially supported non-fast runtime lane. |
| 3.12 | full extras | accurate-research | blocked | blocked | pass | pass | CLI/profile plumbing and `emotion2vec` backend/runtime/training paths are implemented with restricted-license gating. CLI profile mode now uses metadata-aware artifact selection, but promotion remains blocked until at least one compatible accurate-research artifact and dedicated rollout/e2e evidence are captured. |
| 3.13 | full extras | accurate-research | pass | pass | pass | pass | `main` evidence captured in run `22414258948` (`.github/workflows/linux-python-3_13-cli-validation.yml`, head SHA `27618a56eb9514d5df5553f7f1885c7d4f640689`). This lane requires `SER_ENABLE_ACCURATE_RESEARCH_PROFILE` plus restricted-backend opt-in (`SER_ENABLE_RESTRICTED_BACKENDS=true` or persisted consent). Darwin x86_64 Python 3.13 remains partial/fast-oriented; accurate-research runtime evidence is Linux 3.13-backed. |

## Required Checks Per Row

1. `uv run ruff check ser tests`
2. `uv run mypy ser tests`
3. `uv run pyright ser tests`
4. `uv run pytest -q`
5. CLI smoke:
   - `uv run ser --train` (as applicable for environment constraints)
   - `uv run ser --file sample.wav`

## Status Values

Use one of:
- `pending`
- `pass`
- `fail`
- `blocked` (include reason in Notes)
