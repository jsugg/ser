# Resolve Open Risks

Last updated: 2026-03-27
Scope: `ser` repository
Status: active working document

## Objective

Close the highest-value open risks identified in the March 27, 2026 codebase assessment while preserving the current public contracts, architectural seams, and release posture.

This document is intentionally operational. It records:

1. the concrete open risks
2. why they matter
3. the implementation plan to close them
4. the acceptance criteria that define "done"
5. the validation required before merge

## Current Risk Register

### Risk 1: non-actionable `librosa` warnings leak into normal inference output

- Status: closed in this change
- Severity: medium
- User impact: production CLI output is noisy and looks unreliable even when inference succeeds
- Evidence:
  - live CLI smoke run emitted raw `librosa` `UserWarning` lines during normal fast-profile inference
  - `ser/runtime/quality_gate_cli.py` already suppresses the exact warning families, but the regular feature-extraction path does not
  - `ser/utils/dsp.py` directly calls `librosa.stft`, `librosa.feature.mfcc`, `librosa.feature.chroma_stft`, and `librosa.feature.melspectrogram`
- Root cause:
  - warning suppression policy is duplicated and only applied in the quality-gate CLI path
  - handcrafted feature extraction does not use a scoped warning policy around the known `librosa` warning families
- Target state:
  - known non-actionable `librosa` warnings are suppressed in normal handcrafted feature extraction
  - suppression remains scoped and specific so unexpected warnings still surface
  - quality-gate CLI uses the same shared policy instead of duplicating the regexes
- Implementation plan:
  1. Extract shared warning filter constants and one reusable filter application helper in `ser/utils/dsp.py`.
  2. Add a scoped context manager in `ser/utils/dsp.py` so normal inference suppresses only the targeted warning families during feature extraction.
  3. Reuse the same helper in `ser/runtime/quality_gate_cli.py` so the warning policy has one source of truth.
  4. Add unit tests under `tests/suites/unit/utils/` covering:
     - suppression of the known `n_fft` warning
     - suppression of the empty-frequency-set tuning warning
     - propagation of unrelated warnings
- Acceptance criteria:
  - normal feature extraction does not emit the known `librosa` warning families
  - unrelated warnings are still observable
  - quality-gate CLI no longer hardcodes duplicate warning filter definitions
- Resolution summary:
  - shared feature-extraction warning filters now live in `ser/utils/dsp.py`
  - normal DSP extraction applies the warning policy with scoped suppression
  - `ser/runtime/quality_gate_cli.py` now reuses the shared helper
  - live CLI smoke validation no longer emits the raw `librosa` warnings

### Risk 2: architecture and README docs have drifted from the current tree

- Status: closed in this change
- Severity: medium
- User impact: maintainers and contributors follow broken or stale references
- Evidence:
  - `README.md` links to `docs/adr`, but `docs/adr` is absent
  - `docs/architecture.md` links to `docs/adr/`, but the directory is absent
  - `docs/codebase-architecture.md` still reports March 12, 2026 counts that no longer match the current tree
- Root cause:
  - documentation was not updated after subsequent architecture and test-suite growth
  - broken references are not guarded by tests
- Target state:
  - contributor-facing architecture docs only reference files and directories that exist
  - codebase snapshot counts and dates reflect the current repository state
  - at least one test protects the repaired docs contract
- Implementation plan:
  1. Repair README and architecture index links so they point to existing, maintained docs.
  2. Refresh snapshot date and counts in `docs/codebase-architecture.md`.
  3. Add a real `docs/adr/README.md` index so architecture-decision references are no longer broken.
  4. Add an integration-style docs contract test under `tests/suites/integration/docs/` that verifies the expected documentation targets exist.
- Acceptance criteria:
  - no broken architecture/doc links remain in the touched surfaces
  - architecture snapshot counts reflect the current tree
  - docs contract test passes
- Resolution summary:
  - README and architecture index now point at a real ADR index file
  - `docs/adr/README.md` now exists as the stable architecture-decision index
  - `docs/codebase-architecture.md` snapshot date and counts are refreshed
  - docs contract tests were added under `tests/suites/integration/docs/`

### Risk 3: branch coverage passes, but margin is too narrow

- Status: materially improved in this change
- Severity: medium
- User impact: small unrelated changes can start failing CI because the coverage budget is sitting near the floor
- Evidence:
  - `make test-cov` currently passes at 78.48% against a fail-under of 78.00%
  - several low-effort modules still have weak or zero coverage:
    - `ser/runtime/benchmarks.py`
    - `ser/_internal/config/runtime_environment.py`
    - `ser/_internal/transcription/in_process_orchestration.py`
- Root cause:
  - the suite is broad but some owner/helper modules remain unexercised
  - easy deterministic paths have not yet been promoted into the organized `tests/suites` structure
- Target state:
  - coverage margin is widened by adding deterministic tests for undercovered owner/helper modules
  - new tests live in organized suite directories
- Implementation plan:
  1. Add unit tests under `tests/suites/unit/runtime/` for `ser/runtime/benchmarks.py`.
  2. Add unit tests under `tests/suites/unit/internal/config/` for `ser/_internal/config/runtime_environment.py`.
  3. Add unit tests under `tests/suites/unit/transcription/` for `ser/_internal/transcription/in_process_orchestration.py`.
  4. Keep assertions focused on real contracts: success paths, failure paths, lifecycle cleanup, and emitted phase hooks.
- Acceptance criteria:
  - new deterministic tests cover the targeted modules
  - total branch coverage clears the configured threshold with materially better headroom than before this change
- Resolution summary:
  - added organized unit coverage for:
    - `ser/runtime/benchmarks.py`
    - `ser/_internal/config/runtime_environment.py`
    - `ser/_internal/transcription/in_process_orchestration.py`
    - `ser/utils/dsp.py`
  - total coverage improved from `78.48%` to `79.03%`
  - deterministic new tests live under `tests/suites/unit/` and `tests/suites/integration/`

### Risk 4: large orchestration hotspots still concentrate maintenance risk

- Status: planned
- Severity: medium
- User impact: slower and riskier future changes in runtime/transcription/data hotspots
- Evidence:
  - large remaining files include:
    - `ser/transcript/backends/stable_whisper.py`
    - `ser/_internal/runtime/accurate_public_boundary.py`
    - `ser/_internal/runtime/medium_public_boundary.py`
    - `ser/data/dataset_prepare.py`
    - `ser/runtime/profile_quality_gate.py`
- Root cause:
  - repeated extraction work has improved architecture, but some owner modules still bundle multiple orchestration concerns
- Target state:
  - further refactors are staged, boundary-safe, and guided by tests rather than broad rewrites
- Implementation plan:
  1. Avoid expanding hotspot scope in this change.
  2. Use the warning-policy consolidation in Risk 1 as the pattern: extract reusable policy/mechanics into small units without widening public APIs.
  3. Prioritize future extractions in this order:
     - `ser/runtime/profile_quality_gate.py`: isolate CLI parsing, report serialization, and evaluation wiring
     - `ser/data/dataset_prepare.py`: isolate provider/source provenance validation from manifest orchestration
     - `ser/transcript/backends/stable_whisper.py`: isolate import/runtime noise policy, transcribe-call assembly, and retry classification seams
  4. Require each future extraction to land with direct unit coverage against the new owner/helper seam.
- Acceptance criteria:
  - this change does not worsen hotspot concentration
  - this document gives the next change packets a concrete extraction order and constraints

## Validation Results

- `make import-lint`: passed
- `uv run --extra dev ruff check ser tests`: passed
- `uv run --extra dev pyright --pythonversion 3.12 ser tests`: passed
- `make test-cov`: passed with `79.03%` total coverage and `939` passing tests
- `uv build`: passed
- `uv run --with twine twine check dist/*`: passed
- `uv run ser --file sample.wav --profile fast --no-transcript --preflight warn`: passed without the previously leaked raw `librosa` warning output

## Change Outcome

- Risks 1 and 2 are closed by this change.
- Risk 3 is improved and now has more CI headroom.
- Risk 4 remains a planned staged refactor track; this change reduced one small maintenance pain point by consolidating duplicated feature-warning policy.

## Execution Plan For This Change

### Phase 1: warning policy consolidation

- Introduce shared handcrafted feature warning filter helpers in `ser/utils/dsp.py`.
- Apply scoped suppression inside DSP feature extraction.
- Reuse the shared helper from `ser/runtime/quality_gate_cli.py`.

### Phase 2: targeted tests and coverage widening

- Add new unit tests under:
  - `tests/suites/unit/utils/`
  - `tests/suites/unit/runtime/`
  - `tests/suites/unit/internal/config/`
  - `tests/suites/unit/transcription/`
- Keep tests deterministic and dependency-light.

### Phase 3: documentation repair

- Refresh stale counts and dates.
- Remove broken doc references by either fixing them or creating the referenced artifact.
- Add one docs contract test under `tests/suites/integration/docs/`.

### Phase 4: release validation

- Run targeted unit and integration suites for touched areas.
- Run repository quality gates:
  - `make import-lint`
  - `uv run --extra dev ruff check ser tests`
  - `uv run --extra dev pyright --pythonversion 3.12 ser tests`
  - `make test-cov`
  - `uv build`
  - `uv run --with twine twine check dist/*`
- Run one end-to-end CLI smoke command and verify the known warning noise no longer appears.

## Definition Of Done

- Risk 1 is closed in code and tests.
- Risk 2 is closed in docs and tests.
- Risk 3 is materially improved with organized new test coverage.
- Risk 4 has a concrete staged plan and is not worsened by the implementation.
- CI-critical local gates pass from the worktree.
- Documentation reflects the current repository state.
