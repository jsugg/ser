# Audit Hardening Plan

Last updated: 2026-03-27
Branch: `audit-hardening-20260327`
Base: `main` at `95c7b4d7ca94993bcb0109de57fe81cedc8d711c`

## Scope

This document turns the validated audit findings into an executable implementation plan for the current repository state.

Out of scope for this task:
- PyPI/package-name distribution changes.
- Release-gate restructuring for GPU/full-dataset publication blocking.
- Architecture Decision Record authoring beyond incidental doc refreshes.

## Current Source Facts

- The repository already uses `tests/suites/{unit,integration,smoke}` with path-derived structural markers in `tests/conftest.py`.
- `CONTRIBUTING.md` still references pre-relocation root test paths for the API boundary lane.
- `.github/workflows/ci.yml` has no scheduled trigger.
- `ser/transcript/backends/faster_whisper.py` still owns real runtime and I/O branches that need stronger unit coverage.
- `ser/utils/common_utils.py` remains a tiny but uncovered utility dependency used by timeline rendering.

## Objectives

### 1. Weekly dependency-regression CI

Add a weekly scheduled run to `.github/workflows/ci.yml` so upstream dependency or resolver regressions surface even without active pull requests.

Acceptance criteria:
- `ci.yml` includes a weekly `schedule` trigger.
- Existing `changes` classification keeps scheduled runs on the full pipeline.
- Contributor docs mention the scheduled lane accurately.

### 2. Proper pytest marker governance

Strengthen test-marker discipline without relying on blanket marker edits. The repository already derives structural markers (`unit`, `integration`, `smoke`, `topology_contract`) from suite paths; the missing piece is contract enforcement around that policy and around non-structural marker ownership.

Implementation direction:
- Add architecture/contract tests inspired by `strongclaw/tests` to keep `tests/conftest.py` lean and structural.
- Assert that special-purpose markers such as `process_isolation` remain explicitly owned by the modules that require them.
- Keep structural suite markers path-derived, not hand-copied into every module.

Acceptance criteria:
- New contract tests fail if root bootstrap starts assigning non-structural markers.
- New contract tests fail if special-purpose markers move out of explicit module ownership.
- Test layout remains under `tests/suites/...` with domain-oriented placement.

### 3. Coverage uplift on runtime hotspots

Increase confidence in the least-tested real runtime code by adding focused unit tests for `FasterWhisperAdapter` and `display_elapsed_time`.

Implementation direction:
- Add a dedicated backend-focused test module under `tests/suites/unit/transcription/backends/`.
- Cover `setup_required`, `prepare_assets`, `load_model`, `transcribe`, and `_is_module_available` branches using fake modules and fake model objects.
- Add utility tests for `display_elapsed_time` under `tests/suites/unit/utils/`.

Acceptance criteria:
- New tests exercise successful and failure/edge branches for the faster-whisper adapter.
- Utility formatting behavior is covered for both long and short output styles.
- Overall branch coverage stays above the configured threshold with additional headroom.

### 4. Documentation refresh

Update contributor-facing docs to match the current suite tree, marker model, and CI behavior.

Acceptance criteria:
- `CONTRIBUTING.md` references `tests/suites/...` paths, not removed flat-root paths.
- CI topology text reflects scheduled CI and current suite/contract responsibilities.
- Any changed workflow/test semantics are documented where contributors would look first.

## Execution Checklist

- [x] Add weekly scheduled CI trigger and any supporting doc updates.
- [x] Add pytest bootstrap/marker contract tests under `tests/suites/integration/architecture/`.
- [x] Add dedicated faster-whisper adapter tests under `tests/suites/unit/transcription/backends/`.
- [x] Add `common_utils` coverage tests under `tests/suites/unit/utils/`.
- [x] Update `CONTRIBUTING.md` and nearby CI docs affected by the implementation.
- [x] Run focused validation while iterating.
- [x] Run full repo validation and compare outcomes against this document.
- [ ] Commit, push, open PR, merge once green, monitor workflows/branch freshness, then clean up worktree and local branches.

## Validation Plan

Focused checks during implementation:
- `uv run --extra dev pytest -q tests/suites/integration/architecture`
- `uv run --extra dev pytest -q tests/suites/unit/transcription/backends tests/suites/unit/utils/test_common_utils.py`
- `uv run --extra dev pytest --cov=ser.transcript.backends.faster_whisper --cov=ser.utils.common_utils --cov-branch --cov-report term-missing tests/suites/unit/transcription/backends tests/suites/unit/utils/test_common_utils.py`

Final required checks:
- `make lint`
- `make type`
- `make import-lint`
- `make test-cov`

## Validation Results

Focused validation completed:
- `uv run --extra dev pytest -q tests/suites/integration/architecture/test_pytest_suite_bootstrap.py tests/suites/unit/transcription/backends/test_faster_whisper_adapter.py tests/suites/unit/utils/test_common_utils.py` -> `24 passed`
- `uv run --extra dev pytest --cov=ser.transcript.backends.faster_whisper --cov=ser.utils.common_utils --cov-branch --cov-report term-missing tests/suites/unit/transcription/backends/test_faster_whisper_adapter.py tests/suites/unit/utils/test_common_utils.py` -> targeted coverage `80.81%`; `ser/transcript/backends/faster_whisper.py` reached `80.00%`

Final validation completed:
- `make lint` -> pass
- `make type` -> pass (`mypy` clean, `pyright` 0 errors / 0 warnings / 0 informations)
- `make import-lint` -> pass
- `make test-cov` -> pass with `986 passed` and total branch coverage `80.08%`

Outcome versus objectives:
- Weekly scheduled CI trigger added.
- Marker governance now has contract coverage around root bootstrap responsibilities and explicit special-marker ownership.
- Faster-whisper hotspot coverage materially improved and `common_utils` now has direct tests.
- Contributor docs now match the suite tree and current CI behavior.

## Notes

- Work only from the dedicated worktree at `/Users/juanpedrosugg/dev/github/ser-audit-hardening-20260327`.
- Keep branch/commit/PR naming neutral and project-scoped.
- If upstream `origin/main` changes during the task, re-check freshness before pushing and again before merging.
