# SER Architecture Refactor Roadmap

This roadmap prioritizes architectural work based on current code structure,
concentration points, and the refactoring already completed in the runtime,
transcription, and data subsystems.

## Priority 0: protect the current architecture

### Goal

Preserve the current gains and prevent regressions in boundary discipline.

### Actions

- Keep [`make import-lint`](../scripts/run_import_lint.sh) green and run `uv run pytest -q` during refactors.
- Track refactor-sensitive hotspots in [`refactor-hotspot-checks.md`](refactor-hotspot-checks.md) and keep executable gates focused on reusable contract lanes rather than curated hotspot subsets.
- Keep public facades narrow: [`ser/api.py`](../ser/api.py), [`ser/config.py`](../ser/config.py).

### Success criteria

- No new public API widening without explicit intent.
- No new large owner modules added without tests and contract coverage.

## Priority 1: keep the model subsystem decomposition stable

### Why this is first

The model subsystem was the last major architecture hotspot. Public training
entrypoint wiring now lives in
[`ser/models/training_entrypoints.py`](../ser/models/training_entrypoints.py),
shared preparation/evaluation lives in
[`ser/models/training_preparation.py`](../ser/models/training_preparation.py),
shared execution/reporting lives in
[`ser/models/training_execution.py`](../ser/models/training_execution.py), and
accurate-profile prepared-training execution assembly lives in
[`ser/models/accurate_training_execution.py`](../ser/models/accurate_training_execution.py),
while accurate-profile preparation/entrypoint helpers live in
[`ser/models/accurate_training_preparation.py`](../ser/models/accurate_training_preparation.py),
and
shared boundary helpers/types live in
[`ser/models/training_support.py`](../ser/models/training_support.py) and
[`ser/models/training_types.py`](../ser/models/training_types.py).
[`ser/models/emotion_model.py`](../ser/models/emotion_model.py) is now reduced
to a boundary-only facade, so the main remaining risk is backsliding into
reconcentration, not preserving a compatibility namespace.

### Remaining extraction slices

- keep `emotion_model.py` as a thin public boundary only
- keep cross-profile preparation, execution, reporting, and boundary support
  split across dedicated `training_*` modules
- keep load/predict boundary wiring delegated through canonical owner modules
  instead of drifting back into `emotion_model.py`
- keep consolidating ambient-settings resolution behind boundary-local helpers
  in model and transcription convenience entrypoints
- keep tests and internal code bound to canonical owner modules instead of
  reintroducing facade-only aliases

### Candidate target modules

- continue using `ser/models/training_*`
- use `ser/models/training_entrypoints.py` for public entrypoint wiring
- use `ser/models/accurate_training_preparation.py` for accurate-profile preparation and entrypoint helpers
- use `ser/models/accurate_training_execution.py` for accurate-profile prepared-training execution assembly
- use `ser/models/training_preparation.py` for shared payload preparation and evaluation
- use `ser/models/training_execution.py` for shared execution/reporting
- use `ser/models/training_support.py` for shared boundary helpers
- use `ser/models/training_types.py` for shared contracts
- continue using `ser/models/artifact_*`
- continue using `ser/models/profile_*`
- continue using `_internal/models/*` for entrypoint-only seams
- avoid creating a second mixed-responsibility owner while extracting

### Stop condition

- `emotion_model.py` stays boundary-first without a compatibility namespace, and
  cross-profile training logic remains split across `training_entrypoints.py`,
  `accurate_training_preparation.py`, `accurate_training_execution.py`,
  `training_preparation.py`, `training_execution.py`, `training_support.py`,
  and `training_types.py` instead of reconcentrating into one large owner.

## Priority 2: reduce ambient config breadth

### Why this matters

`get_settings()` is still widely used. Current execution scoping is already
safer than a process-global singleton because `settings_override()` uses
execution-context state, but ambient settings still weaken dependency
visibility and make execution paths harder to reason about in isolation.

### Actions

- Prefer explicit `settings: AppConfig` threading in public orchestrators and owner modules.
- Limit `get_settings()` to boundary adapters, convenience helpers, and
  compatibility shims.
- Prefer one boundary-local settings resolver per public module instead of
  repeating ambient lookups across many sibling entrypoints. `api.py`,
  `data_loader.py`, `feature_extractor.py`, `emotion_model.py`,
  `transcript_extractor.py`, `profiling.py`, `diagnostics/command.py`, and
  `runtime/profile_quality_gate.py` should follow that pattern.
- When extracting new helpers, pass settings explicitly rather than resolving ambient state inside them.

### Stop condition

- `get_settings()` is primarily a boundary convenience, not a core-domain dependency.

## Priority 3: continue slimming transcription orchestrators

### Why this is not Priority 1

The transcription subsystem is already fairly well decomposed. The remaining
issue is preventing public-boundary drift in `transcript_extractor.py`, not
missing internal owners. Recent extraction work already moved transcript
entrypoint orchestration, profiling calibration flow, profiling CLI dispatch,
and profiling public-boundary wiring into `_internal/transcription/*`, and
process-isolation boundary wiring plus in-process/runtime-profile public-boundary
wiring now also route through `_internal/transcription/public_boundary_process.py`
and `_internal/transcription/public_boundary_runtime.py`, so the remaining work
is opportunistic slimming instead of missing ownership seams.

### Target files

- `ser/transcript/transcript_extractor.py`
- `ser/transcript/profiling.py`

### Actions

- Keep moving pure logic and policy into `_internal/transcription/*`.
- Keep public modules focused on boundary adaptation and stable signatures.
- Avoid reintroducing backend-specific or process-isolation details into the public modules.

### Stop condition

- Public transcription modules are mostly policy wrappers and public type owners.

## Priority 4: harden public-to-internal dependency policy

### Current state

The `_internal` boundary is curated, not absolute. This is acceptable, but it
must remain intentional. The explicit allowlist already lives in
[`docs/subsystem-dependency-map.md`](subsystem-dependency-map.md), and
[`tests/test_api_import_boundary.py`](../tests/test_api_import_boundary.py)
already fails on drift. Priority 4 is now about maintaining that policy, not
inventing it from scratch.

### Actions

- Keep one explicit allowlist of public modules that are allowed to depend on
  `_internal` by design.
- Document that allowlist in the dependency map.
- Keep a test that fails on unexpected additions or silent drift in that set.
- Avoid widening `TID251` exceptions or boundary allowlists without updating the
  docs and contract tests in the same change packet.

### Stop condition

- The soft boundary stays explicit policy instead of drifting back into tacit convention.

## Priority 5: improve architecture documentation fidelity

### Why

The architecture docs now have dedicated contract coverage via
`make architecture-docs`, but fidelity still depends on updating the docs when
seams, counts, or policy move. The remaining risk is wording drift, not missing
guardrails.

### Actions

- Keep [`docs/codebase-architecture.md`](codebase-architecture.md) current when
  major seams change.
- Keep the diagram and dependency map in sync with code movement.
- Keep `make architecture-docs` green when docs or module counts move.
- Link new ADRs when architectural direction changes materially.

### Stop condition

- Docs and code no longer diverge in obvious ways.

## Recommended execution order

1. protect boundary and contract discipline
2. continue slimming the model subsystem public boundary
3. reduce ambient config breadth in touched paths
4. continue slimming transcription orchestrators only where value remains
5. maintain the explicit soft `_internal` dependency policy
6. maintain architecture docs as part of each change packet

## What not to do

- Do not try to convert the whole codebase to a pure hexagonal architecture in one pass.
- Do not remove profile-specific runtime modules if their behavior is still materially different.
- Do not replace pragmatic owner-module extraction with premature abstraction layers.
- Do not widen the public API just to make internal orchestration easier.

## Architectural end-state worth aiming for

- stable thin public facades
- explicit settings and env ownership at execution boundaries
- slim public orchestrators
- concentrated shared primitives for retries, worker lifecycle, and env management
- model subsystem decomposed to the same maturity level as runtime and data
