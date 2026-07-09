# Public API surface — top-1% hardening plan

| | |
|---|---|
| **Status** | ACTIVE |
| **Created** | 2026-07-08 |
| **Repo** | `~/dev/github/ser` (Python 3.12.8 pin, hatchling, unpublished — no PyPI release, no git tags) |
| **Branch convention** | one branch per phase off `main`, e.g. `refactor/public-api-hardening-p0` |
| **Tracking** | Task Ledger (§6) + Implementation Journal (§9) in this file |
| **Predecessor** | `.local/api-surface-tightening.md` (shipped as PR #70, merged 2026-07-08) |

---

## 1. Intent

PR #70 gave the repo a disciplined boundary *mechanism*: `ser/_internal/`, thin
`ser.api`/`ser.config` facades, `boundary_policy.toml` enforced by
`tests/suites/integration/architecture/test_api_import_boundary.py`. This plan finishes
the job so the public surface is **machine-verified, self-sufficient, versioned, and
governed** — the four properties that separate top-1% Python API surfaces (trio,
urllib3, attrs, structlog) from merely well-designed ones:

1. **Machine-verified**: the exact public surface is a checked-in snapshot; any change
   is an explicit, reviewable diff. Type completeness is scored and gated. Import cost
   is a contract test, not a convention.
2. **Self-sufficient**: every type appearing in a `ser.api` signature is importable
   from `ser.api`. No consumer ever needs a non-facade import path.
3. **Versioned**: `ser.__version__` exists; `py.typed` ships so downstream type
   checkers actually see the carefully designed types.
4. **Governed**: a written stability policy per tier, tested doc examples, and a
   changelog discipline that starts before first publish.

## 2. Goals

- G1 — Ship `py.typed`; pyright `--verifytypes` score for `ser` ≥ 95% (ratchet from baseline).
- G2 — `ser.api` exports every type in its own signatures; `__all__` complete.
- G3 — Checked-in public-API snapshot + contract test that fails on any surface drift.
- G4 — Import-cost contract test: `import ser` / `import ser.api` never imports `torch`.
- G5 — Public tree contains only tier-1 modules and genuine facades; implementation
  lives under `ser/_internal/`; `boundary_policy.toml` shrinks to facades only.
- G6 — Written stability policy, tested README examples, bootstrapped changelog.

## 3. Non-goals / out of scope

- NG1 — No runtime behavior, dependency (runtime), or algorithm changes. Moves only.
- NG2 — No version bump. Package is unpublished; stays `1.0.0`. No deprecation shims —
  hard moves are the established convention here (see predecessor doc).
- NG3 — No publishing, no triggering of publish workflows
  (`.github/workflows/python-publish*.yml`), no edits to them.
- NG4 — No mkdocs/documentation site (deferred until after first publish).
- NG5 — No towncrier; a plain `CHANGELOG.md` is enough pre-publish.
- NG6 — No CLI surface changes (`ser/__main__.py`, `ser/_internal/cli/*`).
- NG7 — No changes to the platform-conditional torch/torchaudio dependency matrix
  (versions, markers, or lock partitions) — see DD-13 for the full locked matrix and
  the files that encode it.

## 4. Current state (verified 2026-07-08)

- Tier-1 facades: `ser/__init__.py` (3 domain NamedTuples only), `ser/api.py`
  (14 functions + 4 re-exported types + `RuntimePipeline` Protocol), `ser/config.py`
  (~25 re-exports from `ser._internal.config`), `ser/domain.py`, `ser/profiles.py`,
  `ser/utils/` (lazy-import facade).
- `ser/_internal/`: 76 files (api/, cli/, config/, runtime plumbing, models support).
- Residual public implementation (~126 files): `ser/data` 39, `ser/models` 26,
  `ser/runtime` 24, `ser/transcript` 19, `ser/utils` 10, `ser/diagnostics` 4,
  `ser/features` 2, `ser/heads` 2, plus `ser/repr`.
- `boundary_policy.toml`: 17 public→`_internal` exceptions; several are implementation
  under public paths (`ser/runtime/*_inference.py`, `ser/models/training_entrypoints.py`,
  `ser/transcript/*`), not facades.
- **No `py.typed`** — the typed surface is invisible to downstream checkers.
- **No `ser.__version__`**.
- `ser.api` signature types `InferenceRequest`/`InferenceExecution`/`SubtitleFormat`/
  `DiagnosticReport` are TYPE_CHECKING-only there and not in `__all__`.
- `show_dataset_consents`/`configure_dataset_consents` return an anonymous
  `tuple[tuple[str, ...], tuple[str, ...]]`.
- `train()` has two escape hatches: `use_profile_pipeline: bool` + `pipeline_builder`.

## 5. Design decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| DD-01 | Hard moves, no deprecation shims, version stays 1.0.0. | Package never published; no external consumers. Matches predecessor doc convention. |
| DD-02 | Tier model: **tier-1** = `ser`, `ser.api`, `ser.config` (trimmed), `ser.domain`, `ser.profiles`, `ser.utils` (curated `__all__` only). **Facades** = the by-export `__init__.py`/named facade modules listed in `boundary_policy.toml`. Everything else internal. | Continues the tier-1 outcome PR #70 declared; removes the ambiguous "looks public" middle tier. |
| DD-03 | `ser.api` re-exports its full signature vocabulary at runtime (not TYPE_CHECKING-only). | A facade that requires side-door imports for its own return types isn't a facade. |
| DD-04 | Type distribution = `py.typed` + pyright `--verifytypes` ratchet gate (baseline first, target ≥ 95%). | "Has annotations" ≠ "ships a verified typed API". The ratchet prevents regression without blocking on legacy gaps day one. |
| DD-05 | Public-API snapshot generated with **griffe** (new dev-only dependency), stored at `tests/suites/integration/architecture/public_api_snapshot.json`, asserted by a contract test. Fallback if griffe is rejected: hand-rolled `inspect`-based dumper in `scripts/`. | Industry-standard tool; the snapshot diff becomes the review artifact for any API change. Dev-dep addition is allowed; runtime deps are not (NG1). |
| DD-06 | Import-cost test asserts **module absence** (`torch` not in `sys.modules` after `import ser, ser.api` in a subprocess), no wall-clock assertions. | Wall-clock is flaky on the slow WSL2 dev box and in CI. Module presence is deterministic. |
| DD-07 | `DatasetConsents(NamedTuple)` with fields `policy_ids`, `license_ids` in `ser.domain`; consent functions return it. | NamedTuple is a tuple subtype — existing unpacking keeps working; new code gets names. |
| DD-08 | `train()` drops `use_profile_pipeline`; `pipeline_builder=None` → profile pipeline, provided → custom. | One extension point. Hard removal is safe per DD-01. |
| DD-09 | `ser.__version__` via `importlib.metadata.version("ser")` with a `PackageNotFoundError` fallback to `"0.0.0.dev0"`; `pyproject.toml` version stays the single static source. | No hatch dynamic-version machinery needed; one source of truth. |
| DD-10 | Mirror the boundary at lint speed with ruff `flake8-tidy-imports` `TID251` banning `ser._internal` imports, allowlisted via `per-file-ignores` generated from `boundary_policy.toml` entries. | Violations fail in seconds at `make lint`, not only in the architecture test lane. The contract test remains authoritative. |
| DD-11 | Module moves use `git mv`; imports updated mechanically; the boundary contract test may only be strengthened, never weakened. | Preserves history; preserves the guarantee. |
| DD-12 | README examples are executed by a contract test (extract fenced `python` blocks, run in subprocess with a stub/sample where needed). No sybil/doctest framework added. | Smallest thing that makes advertised examples provably work. |
| DD-13 | The platform-conditional torch/torchaudio dependency matrix is **authoritative and frozen** for this effort. Where it lives (stable anchors, not line numbers): the marker-conditional `torch`/`torchaudio` specifiers in `[project] dependencies` of `pyproject.toml` (find: `rg -n 'torch' pyproject.toml`); the `resolution-markers` block at the top of `uv.lock` (environment partitions); and the `[[package]]` entries for `torch` and `torchaudio` in `uv.lock` (find: `rg -n '^name = "torch(audio)?"' uv.lock`). Locked outcomes: Darwin x86_64 + py<3.13 → torch==2.2.2 (rule `>=2.2,<=2.2.2`); Darwin x86_64 + py3.13 → no torch marker matches, no direct locked torch; Linux/Windows/Darwin arm64 + py3.12 → torch==2.10.0 (rule `>=2.2,<3.0`); same platforms + py3.13 → torch==2.10.0 (rule `>=2.6,<3.0`). torchaudio mirrors the split (2.2.2 on old Intel py3.12; 2.10.x has no full runtime wheels there). CI lanes depend on it: Linux CI 3.12/3.13 → 2.10.0; Darwin Intel validation (macos-15-intel, py3.12) → 2.2.2; macOS MPS arm64 → 2.10.0; Linux self-hosted GPU → 2.10.0 + Linux x86_64 CUDA deps from torch wheel metadata. Any `uv lock` re-resolution (e.g. P1-01's griffe addition) must leave every torch/torchaudio locked version and marker partition byte-identical; verify with `git diff uv.lock` scoped to those sections before committing. | This matrix encodes hard platform compatibility (old Intel Mac wheel cutoff, MPS, CUDA) that a routine lock refresh could silently destroy; four CI lanes and the GPU runner assume it. |

## 6. Task Ledger

Statuses: `TODO` → `IN-PROGRESS` → `DONE` (or `BLOCKED` / `DROPPED` with journal entry).
One task `IN-PROGRESS` at a time. Update this table **and** the journal on every transition.

### Phase 0 — Ship the types you already wrote (additive, no moves)

| ID | Task | Status | Depends on |
|----|------|--------|------------|
| P0-01 | `py.typed` marker shipped in the wheel | DONE | — |
| P0-02 | `ser.__version__` | DONE | — |
| P0-03 | `ser.api` self-sufficiency (re-export signature vocabulary) | DONE | — |
| P0-04 | `DatasetConsents` NamedTuple | DONE | — |
| P0-05 | `train()` single extension point | DONE | — |

**P0-01** — Create empty `ser/py.typed`. Add it to the hatchling build include in
`pyproject.toml` (next to the existing `ser/profile_defs.yaml` include).
*Acceptance*: `uv build` succeeds and `unzip -l dist/ser-1.0.0-py3-none-any.whl | rg py.typed`
shows the file; `make lint type` exit 0.

**P0-02** — In `ser/__init__.py`, expose `__version__` per DD-09; add to `__all__`.
*Acceptance*: `uv run --frozen --extra dev python -c "import ser; print(ser.__version__)"`
prints `1.0.0`; import-cost invariant still holds (see P1-03 command, runnable ad hoc).

**P0-03** — In `ser/api.py`: move `InferenceRequest`, `InferenceExecution`,
`SubtitleFormat`, `DiagnosticReport`, `DiagnosticFinding`, `DiagnosticSeverity` out of
the TYPE_CHECKING guard into runtime re-exports; add them plus `AppConfig` and
`ProfileName` to `__all__`. **Contingency**: importing `ser.runtime.contracts` executes
`ser/runtime/__init__.py` (which imports `pipeline`); if that pulls heavy deps, relocate
the contract dataclasses to a leaf module (candidate: `ser/_internal/runtime/` leaf
re-exported by `ser.runtime.contracts`) rather than accepting the import cost.
*Acceptance*: `uv run --frozen --extra dev python -c "import sys; from ser.api import InferenceRequest, InferenceExecution, SubtitleFormat, DiagnosticReport, AppConfig, ProfileName; assert 'torch' not in sys.modules"`
exits 0; `make check` exits 0.

**P0-04** — Add `DatasetConsents(NamedTuple)` (`policy_ids: tuple[str, ...]`,
`license_ids: tuple[str, ...]`) to `ser/domain.py`; return it from
`show_dataset_consents`/`configure_dataset_consents` in `ser.api` and the internal
owner; export from `ser.api` and `ser.domain`.
*Acceptance*: existing consent tests pass unmodified (tuple compatibility proven);
new assertion on field access added to the relevant test module; `make check` exits 0.

**P0-05** — Remove `use_profile_pipeline` from `ser.api.train` and
`ser._internal.api.runtime.train`; `pipeline_builder` becomes the sole override. Update
all callers (CLI, tests).
*Acceptance*: `rg -n "use_profile_pipeline" ser tests` returns nothing;
`make check test-cov` exit 0.

### Phase 1 — Machine-enforce the contract

| ID | Task | Status | Depends on |
|----|------|--------|------------|
| P1-01 | Public-API snapshot + drift contract test | DONE | P0-03, P0-04, P0-05 |
| P1-02 | pyright `--verifytypes` ratchet gate | DONE | P0-01 |
| P1-03 | Import-cost contract test | DONE | P0-03 |
| P1-04 | ruff TID251 boundary lint | DONE | — |
| P1-05 | CI wiring for the new gates | TODO | P1-01..04 |

**P1-01** — Add `griffe` to the dev extra (`uv lock` refresh; `make lock-check` must
pass). Script `scripts/dump_public_api.py` dumps names + signatures + annotations for
tier-1 modules (DD-02) to `tests/suites/integration/architecture/public_api_snapshot.json`
(sorted keys, stable formatting). Contract test regenerates in-memory and diffs against
the checked-in snapshot with a failure message saying how to intentionally update.
*Acceptance*: test passes on clean tree; demonstrably fails when a symbol is added to
`ser/api.py` `__all__` (show the failing run in the journal, then revert); `make check
lock-check` exit 0; per DD-13, `git diff uv.lock` shows **no** change to any torch or
torchaudio version, marker, or partition line — paste the (empty) scoped diff into the
journal entry.

**P1-02** — Make target `type-completeness`: `uv run --frozen --extra dev pyright
--verifytypes ser --ignoreexternal --outputjson` parsed against a threshold stored in
`pyproject.toml` or the Makefile. Record the baseline score first; set the gate at
baseline; file the ratchet-to-95% as follow-up work in the journal if baseline < 95%.
*Acceptance*: `make type-completeness` exits 0 at baseline and the baseline number is
recorded in the journal.

**P1-03** — Contract test (architecture suite): subprocess runs
`import ser, ser.api, ser.config, ser.domain, ser.profiles, ser.utils` then asserts
`torch not in sys.modules` (DD-06).
*Acceptance*: test passes; `make check` exits 0.

**P1-04** — Configure ruff `TID251` to ban `ser._internal` imports tree-wide, with
`per-file-ignores` exactly mirroring `boundary_policy.toml` paths. Add a comment in
`pyproject.toml` pointing at the policy file as the source of truth.
*Acceptance*: `make lint` exits 0 on clean tree; a synthetic violation (temporary
`from ser._internal.config.schema import AppConfig` in a non-allowlisted module) fails
`make lint` — demonstrate, journal it, revert.

**P1-05** — Add snapshot test lane (it runs with the existing architecture suite —
verify it is collected in CI), and `type-completeness` to the CI quality workflow.
Never touch publish workflows (NG3).
*Acceptance*: workflow YAML change passes `make workflow-lint ci-contracts`; the
updated workflow is dispatched once via `gh workflow run` (validation workflows only)
and observed green via `verifier-ci`.

### Phase 2 — Finish the tier consolidation (moves only)

| ID | Task | Status | Depends on |
|----|------|--------|------------|
| P2-01 | Inventory & classification appendix | TODO | P1-01 |
| P2-02 | `ser/models` → internal (keep facades) | TODO | P2-01 |
| P2-03 | `ser/runtime` → internal (keep contracts/pipeline/registry facades) | TODO | P2-01 |
| P2-04 | `ser/data` → internal (keep `application.py` + curated `__init__`) | TODO | P2-01 |
| P2-05 | `ser/transcript` → internal (keep policy-listed facades) | TODO | P2-01 |
| P2-06 | `ser/features`, `ser/heads`, `ser/repr`, `ser/diagnostics` residue | TODO | P2-01 |
| P2-07 | Shrink `boundary_policy.toml` to genuine facades; strengthen contract test | TODO | P2-02..06 |
| P2-08 | `ser/utils` trim to curated `__all__` | TODO | P2-01 |

**P2-01** — Generate the full classification: for every `.py` under public non-tier-1
paths, record `keep-as-facade` (with reason) or `move-internal` (with destination).
Append as Appendix A of this file. This is the review artifact for the phase; get it
into the journal before any move.
*Acceptance*: Appendix A exists, covers all ~126 files, no `unclassified` rows.

**P2-02..P2-06** — Per subpackage, in dependency order within each task: `git mv`
move-internal modules to `ser/_internal/<subpackage>/`, update imports (internal code
imports internal paths directly; facades re-export), keep public `__init__.py` exports
byte-compatible for symbols classified keep. One commit per subpackage.
*Acceptance per task*: `make check test-cov import-lint` exit 0; P1-01 snapshot test
passes **unchanged** (tier-1 surface must not move); coverage gate (fail_under=78) holds.

**P2-07** — Remove policy entries whose paths moved internal; the remaining entries are
genuine facades only. Strengthen the contract test if PR #70 didn't already: it must
fail on new `_internal` imports without a policy entry and on tier-1 `__all__` growth.
*Acceptance*: policy file ≤ 10 entries, each with a facade reason; contract test
demonstrably fails on a synthetic unlisted `_internal` import (journal, revert).

**P2-08** — `ser/utils/__init__.py` keeps only the curated lazy `__all__`; helper
modules not re-exported move internal.
*Acceptance*: same gate set as P2-02..06.

### Phase 3 — Governance

| ID | Task | Status | Depends on |
|----|------|--------|------------|
| P3-01 | `docs/api-stability.md` | TODO | P2-07 |
| P3-02 | README Python API section refresh | TODO | P2-07 |
| P3-03 | README examples executed by a contract test | TODO | P3-02 |
| P3-04 | `CHANGELOG.md` bootstrap | TODO | — |

**P3-01** — Document: tier-1 list (DD-02), the SemVer promise that activates at first
publish, what `_internal` means, how the snapshot test governs API change, pointer to
`boundary_policy.toml`.
*Acceptance*: file exists, linked from README; `make lint` (docs lint if any) passes.

**P3-02** — README "Python API" section shows `ser.api` as sole supported entry point,
a minimal `infer` example, `__version__`, and links the stability doc.
*Acceptance*: `rg` shows no README references to moved/removed symbols.

**P3-03** — Contract test per DD-12 executes README `python` blocks.
*Acceptance*: test passes; deliberately breaking an example fails it (journal, revert).

**P3-04** — `CHANGELOG.md` in keep-a-changelog format with an `[Unreleased]` section
summarizing this effort (no branding words — see §7).
*Acceptance*: file exists; `make lint` passes.

## 7. Operating instructions for the coding agent

1. **Read this file top to bottom before any work.** Re-read §5 and §8 after context
   compaction.
2. **Environment**: Python pinned 3.12.8 via `.python-version`. Every dev-tool
   invocation must be `uv run --frozen --extra dev ...` — bare `uv run` rebuilds the
   venv without dev tools. Make targets already do this.
3. **Gates** (the definition of "green"): `make lint`, `make type`, `make test-cov`
   (coverage fail_under=78), `make import-lint`, `make lock-check`. Full local sweeps
   are fine; `make quality-gate-full`, training, and accurate-profile inference belong
   in CI only (slow WSL2 box).
4. **Git discipline**: branch per phase off `main` (`refactor/public-api-hardening-pN`),
   conventional commits, one logical commit per task (P2 allows one per subpackage).
   **Never** use the words claude, anthropic, codex, or openai in branch names, commit
   messages, PR titles/descriptions, or labels. Open a PR per phase; do not merge
   without the user.
5. **Workflow per task**: journal a *start* entry → set ledger `IN-PROGRESS` → implement
   → run the task's acceptance commands + the gate set → journal a *done* entry citing
   the actual command results → set ledger `DONE` → commit (code + this file together).
6. **Verification honesty**: only claim what a command in this session demonstrated.
   Where a task says "demonstrate failure, then revert", the failing output must appear
   in the journal entry.
7. **Delegation**: run gate sweeps via a `verifier-lite` subagent with the exact
   commands; use `verifier-ci` for remote CI status. Never dispatch publish workflows.
8. **ffmpeg** is a runtime prerequisite for inference only, not tests; do not attempt
   to install it (needs sudo).

## 8. Stop conditions — halt, journal, and report to the user when…

- S1 — An acceptance or gate command fails for a reason **unrelated** to the current
  task (pre-existing breakage, environment issue).
- S2 — A task appears to require weakening the boundary contract test, the coverage
  gate, or any existing check.
- S3 — A change would touch publish workflows, trigger publishing, or bump the version.
- S4 — A move requires a runtime behavior change to keep tests green (violates NG1).
- S5 — Two consecutive failed attempts at the same acceptance criterion.
- S6 — The P2-01 classification finds a module that cannot be cleanly classified
  (genuinely ambiguous ownership).
- S7 — A dependency addition beyond `griffe` (dev-only) seems needed.
- S8 — The `/goal` turn budget in the companion goal file is reached with tasks open.
- S9 — Any operation would alter the torch/torchaudio locked matrix (DD-13): a lock
  refresh changes a torch/torchaudio version, marker, or partition, or a task seems to
  require touching the torch/torchaudio specifiers in `pyproject.toml` or the
  `torch`/`torchaudio` `[[package]]` entries or `resolution-markers` in `uv.lock`.

## 9. Implementation Journal

Protocol: newest entry **first**. One entry when a task starts (intent + approach),
one when it finalizes (evidence: commands run and their results, deviations from spec,
follow-ups). Entries also required for BLOCKED/DROPPED transitions and stop-condition
hits. Keep entries terse but evidence-bearing.

Template:

```
### YYYY-MM-DD HH:MM — <TASK-ID> <started|done|blocked>
- What: …
- Evidence: `<command>` → <result summary>
- Deviations / follow-ups: …
```

### 2026-07-09 20:40 — P1-04 done
- What: Broadened Ruff TID251 to ban `ser._internal` for public source files,
  changed the TID251 lane to scan public `ser` files only, and made per-file
  ignores mirror `boundary_policy.toml` exactly.
- Evidence: `bash scripts/run_import_lint.sh` → Ruff TID251 passed and boundary
  contract tests `16 passed`; synthetic `from ser._internal.config.schema import
  AppConfig` in non-allowlisted `ser/domain.py` made `make lint` fail with
  `TID251 ser._internal is banned`, then the synthetic edit was reverted;
  `uv run --frozen --extra dev black tests/suites/integration/architecture/test_import_lint_policy.py`
  → reformatted policy test; `make lint` → Ruff/black/isort all pass.
- Deviations / follow-ups: TID251 scans public source files only; tests and internal
  implementation modules remain free to import internal helpers directly.

### 2026-07-09 20:37 — P1-04 started
- What: Broaden Ruff TID251 from API-owner imports to all public imports of
  `ser._internal`, with exceptions generated from `boundary_policy.toml`.
- Evidence: Existing `make lint` passes before broadening, but TID251 only bans
  `ser._internal.api*`; synthetic whole-internal boundary coverage is not present yet.
- Deviations / follow-ups: The TID251 lane will target public `ser` files only so
  internal implementation modules and internal-focused tests can keep importing
  `_internal` directly.

### 2026-07-09 20:36 — P1-03 done
- What: Added `tests/suites/integration/architecture/test_public_import_cost.py`
  to assert tier-1 public imports leave `torch` absent from `sys.modules` in a
  subprocess.
- Evidence: `uv run --frozen --extra dev pytest -q tests/suites/integration/architecture/test_public_import_cost.py`
  → `1 passed`; `make check` → lint/type/test all pass, `1027 passed in 70.27s`.
- Deviations / follow-ups: none.

### 2026-07-09 20:32 — P1-03 started
- What: Add an architecture subprocess test proving tier-1 public imports do not
  import `torch`.
- Evidence: P0 import-cost one-liner already passed before this task; adding a
  collected contract test makes that invariant permanent.
- Deviations / follow-ups: none.

### 2026-07-09 20:32 — P1-02 done
- What: Added `make type-completeness`, backed by
  `scripts/check_type_completeness.py`, with the baseline stored at
  `[tool.ser.type_completeness].threshold`.
- Evidence: `make type-completeness` → `pyright verifytypes completeness:
  0.9788235294 (threshold 0.9788235294)`; `uv run --frozen --extra dev black
  --check scripts/check_type_completeness.py` → unchanged.
- Deviations / follow-ups: Baseline already exceeds 95%, so no ratchet-to-95%
  follow-up needed.

### 2026-07-09 20:30 — P1-02 started
- What: Add a `make type-completeness` gate around pyright `--verifytypes ser` and
  record the current completeness baseline as the threshold.
- Evidence: `uv run --frozen --extra dev pyright --verifytypes ser --ignoreexternal
  --outputjson` → completeness score `0.9788235294117648` with zero diagnostics
  before adding the gate.
- Deviations / follow-ups: none.

### 2026-07-09 20:29 — P1-01 done
- What: Added `griffe` to the dev extra, `scripts/dump_public_api.py`, the tier-1
  JSON snapshot, and a contract test that diffs current griffe output against the
  checked-in snapshot.
- Evidence: `uv run --frozen --extra dev python scripts/dump_public_api.py --write`
  → generated `tests/suites/integration/architecture/public_api_snapshot.json`;
  `uv run --frozen --extra dev pytest -q tests/suites/integration/architecture/test_public_api_snapshot.py`
  → `1 passed`; synthetic `"SyntheticExport"` in `ser.api.__all__` made that test
  fail with `ValueError: ser.api.__all__ exports missing member 'SyntheticExport'`,
  then the synthetic edit was reverted; `make lock-check && make check` → lock
  fresh, lint/type/test all pass, `1026 passed in 119.60s`; `uv run --frozen --extra
  dev black --check scripts/dump_public_api.py` → unchanged; scoped
  `git diff` searches for `torch`, `torchaudio`, and `resolution-markers` in
  `uv.lock`/`pyproject.toml` → no matches.
- Deviations / follow-ups: Baseline snapshot includes `ser.profiles` public
  definitions because that tier-1 module has no `__all__`; explicit exports can be
  added in a later public-surface cleanup if desired.

### 2026-07-09 20:17 — P1-01 started
- What: Add the griffe-backed tier-1 public API snapshot script, checked-in JSON
  snapshot, and architecture drift contract test.
- Evidence: Current branch rebuilt from `main` plus P0 commits; `git status` clean
  before this task.
- Deviations / follow-ups: none.

### 2026-07-09 03:10 — P0-05 done
- What: Removed `use_profile_pipeline` from `ser.api.train`, the internal train/
  training-command chain, the CLI, and the restricted-backend gate plumbing;
  `pipeline_builder` is the sole training override. Gate-side booleans renamed to
  `profile_resolution_enabled`/`profile_routing_enabled` with identical value flow
  (CLI still feeds `profile_pipeline_enabled(settings)`), so no behavior change.
- Evidence: `rg -n "use_profile_pipeline" ser tests` → no matches (exit 1);
  `uv run --frozen --extra dev python -c "import ser; print(ser.__version__)"` →
  `1.0.0`; import-cost one-liner importing all P0-03/P0-04 symbols from `ser.api`
  → exit 0 with `torch` absent; `uv build` → wheel built, `unzip -l` shows
  `ser/py.typed`; verifier-lite gate sweep → `make lint` (all checks passed),
  `make type` (mypy 0 issues / pyright 0 errors), `make test-cov`
  (`1025 passed in 181.76s`, coverage gate held), `make import-lint`
  (16 passed), `make lock-check` (lock verified) — all exit 0.
- Deviations / follow-ups: none; NG6 respected (CLI flags untouched, only internal
  call plumbing updated).

### 2026-07-09 00:28 — P0-04 done
- What: Added `DatasetConsents(NamedTuple)`, returned it from consent APIs, exported it
  from `ser.domain` and `ser.api`, and added field-access regression assertions.
- Evidence: import/tuple-compatibility check for `ser.api.DatasetConsents` → exit 0
  with `torch` absent; targeted API/boundary/dataset-consent tests → `56 passed`; first
  targeted run failed only on `ser.api.__all__` ordering and passed after reorder;
  `rtk make check` → lint passed, mypy `Success: no issues found in 390 source files`,
  pyright `0 errors, 0 warnings, 0 informations`, pytest `1025 passed`.
- Deviations / follow-ups: verifier-lite unavailable due account usage limit; parent
  ran checks directly per stop-free fallback.

### 2026-07-09 00:34 — P0-05 started
- What: Remove the `train()` profile-pipeline escape hatch and make
  `pipeline_builder` the only training override.
- Evidence: `rtk rg -n "use_profile_pipeline" ser tests` showed public API, CLI,
  restricted-backend gate, command-wrapper, and tests still carrying the old flag.
- Deviations / follow-ups: no runtime behavior change intended; restricted-backend
  profile-routing flag is renamed, not semantically changed.

### 2026-07-08 23:47 — P0-04 started
- What: Add `DatasetConsents` tuple-compatible named return type and export it through
  `ser.domain` and `ser.api`.
- Evidence: consent APIs currently return anonymous `tuple[tuple[str, ...],
  tuple[str, ...]]`; existing API test only proves unpacking, not field access.
- Deviations / follow-ups: none.

### 2026-07-08 23:31 — P0-03 done
- What: Runtime-exported `InferenceRequest`, `InferenceExecution`, `SubtitleFormat`,
  `DiagnosticReport`, `DiagnosticFinding`, `DiagnosticSeverity`, `AppConfig`, and
  `ProfileName` from `ser.api`; updated API and tier-1 export snapshots.
- Evidence: verifier-lite `019f44b6-e188-7bb0-97f6-1ced57ed2667` import-cost command
  importing the P0-03 symbols from `ser.api` → exit 0 with `torch` absent; targeted API
  + boundary tests → `52 passed`; `rtk make lint` → passed. Parent reran `rtk make
  type` with a long window → mypy `Success: no issues found in 390 source files`;
  pyright `0 errors, 0 warnings, 0 informations`.
- Deviations / follow-ups: verifier-lite again used too-short `make type` wait; parent
  long-window rerun completed successfully.

### 2026-07-08 23:31 — P0-03 started
- What: Make `ser.api` runtime-export every type used by its own public signatures.
- Evidence: `ser.api` currently imports `InferenceRequest`, `InferenceExecution`,
  `SubtitleFormat`, and `DiagnosticReport` only under `TYPE_CHECKING`; `AppConfig` and
  `ProfileName` are imported but absent from `__all__`.
- Deviations / follow-ups: `from ser.runtime.contracts ...` and diagnostics domain
  imports kept `torch` absent in a subprocess, so no contract relocation needed.

### 2026-07-08 23:13 — P0-02 done
- What: Added `ser.__version__` via `importlib.metadata.version("ser")` with
  `PackageNotFoundError` fallback, exported it from root package, and updated the
  tier-1 export contract.
- Evidence: verifier-lite `019f44a7-7f9c-7221-a2c9-45f943e29bf2` ran
  `rtk uv run --frozen --extra dev python -c "import ser; print(ser.__version__)"` →
  `1.0.0`; import-cost check → exit 0 with `torch` absent; boundary test →
  `13 passed`; `rtk make lint` → passed. Parent reran `rtk make type` with a long
  window → mypy `Success: no issues found in 390 source files`; pyright `0 errors, 0
  warnings, 0 informations`.
- Deviations / follow-ups: verifier-lite's first `make type` timeout was too short for
  local pyright; parent rerun completed successfully.

### 2026-07-08 23:13 — P0-02 started
- What: Expose `ser.__version__` from package metadata without adding heavy imports.
- Evidence: `ser/__init__.py` currently only re-exports domain NamedTuples; tier-1
  export snapshot for `ser` contains no `__version__`.
- Deviations / follow-ups: none.

### 2026-07-08 22:07 — P0-01 done
- What: Added empty `ser/py.typed` and included it in the hatchling wheel manifest.
- Evidence: `rtk uv build` → built `dist/ser-1.0.0-py3-none-any.whl`;
  `rtk unzip -l dist/ser-1.0.0-py3-none-any.whl | rtk rg py.typed` → `0 ...
  ser/py.typed`; `rtk make lint` → passed; `rtk make type` → passed
  (`Success: no issues found in 390 source files`; `0 errors, 0 warnings, 0
  informations`) via verifier-lite `019f449e-1b52-7763-957e-2242f67db12b`.
- Deviations / follow-ups: Cold pyright on this WSL2 box took ~20 minutes; no task
  changes needed.

### 2026-07-08 22:07 — P0-01 started
- What: Ship the PEP 561 marker and include it in the wheel build manifest.
- Evidence: `git status --short --branch` → clean `refactor/public-api-hardening-p0`
  worktree before edits.
- Deviations / follow-ups: Working in clean sibling worktree
  `../ser-public-api-hardening-p0` to avoid unrelated dirty files in the original
  checkout.

### 2026-07-08 — Plan created
- What: Document authored from a live audit of the surface (facade files, boundary
  policy, packaging config, subpackage inventory) plus the PR #70 predecessor spec.
  No implementation work has started; all tasks TODO.
- Evidence: audit commands in the authoring session (file reads of `ser/api.py`,
  `ser/config.py`, `boundary_policy.toml`, `pyproject.toml`; `find`/`rg` inventories).
- Follow-ups: none.

## Appendix A — Phase 2 classification (populated by P2-01)

*(empty until P2-01)*
