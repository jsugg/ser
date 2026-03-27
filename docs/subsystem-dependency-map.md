# SER Subsystem Dependency Map

This document describes the current subsystem dependency directions and the most
important coupling points in the codebase.

## Intended dependency direction

```text
CLI / Public API
        |
        v
Internal API / CLI owners
        |
        v
Runtime / Data / Diagnostics / Transcription / Model subsystems
        |
        v
Shared primitives, schemas, utils, external libraries
```

## Public boundaries

### `ser.api`

- Depends on: `_internal.api.data`, `_internal.api.runtime`, `_internal.api.diagnostics`
- Exposes: stable library workflows for datasets, training, inference, diagnostics
- Design note: thin facade with optional explicit settings injection

### `ser.config`

- Depends on: `_internal.config.bootstrap`, `_internal.config.schema`, `profiles`
- Exposes: stable typed config surface only
- Design note: public facade is intentionally narrow and no longer carries compatibility shims

### `ser.__main__`

- Depends on: `_internal.cli.data`, `_internal.cli.runtime`, `_internal.cli.diagnostics`
- Exposes: CLI only
- Design note: composition-heavy but now appropriately funneled through internal CLI support modules

## Runtime subsystem

### Primary nodes

- `runtime/pipeline.py`
- `runtime/backend_hooks.py`
- `runtime/registry.py`
- `runtime/*_inference.py`
- `_internal/runtime/accurate_public_boundary.py`
- `_internal/runtime/medium_public_boundary.py`
- `_internal/runtime/*`

### Inbound dependencies

- `_internal/api/runtime.py`
- `ser.api`
- `ser.__main__`

### Outbound dependencies

- `profiles.py`
- `transcript/transcript_extractor.py`
- `models/emotion_model.py`
- `utils/timeline_utils.py`
- `_internal/runtime/environment_plan.py`
- `_internal/runtime/process_env.py`

### Coupling assessment

- Strong internal cohesion around orchestration and operational behavior
- Medium coupling to config via `AppConfig` and boundary-level `reload_settings()` fallbacks
- Acceptable profile-local duplication, now mostly reduced to public-boundary wrappers and internal support owners

## Transcription subsystem

### Primary nodes

- `transcript/transcript_extractor.py`
- `transcript/profiling.py`
- `transcript/backends/*`
- `_internal/transcription/*`

### Inbound dependencies

- `runtime/pipeline.py`
- `diagnostics/service.py`
- CLI calibration flows

### Outbound dependencies

- transcription backend adapters
- runtime policy
- config/profile resolution
- process isolation and worker lifecycle helpers
- public-boundary process-isolation wiring via `_internal/transcription/public_boundary_process.py`
- public-boundary profiling wiring and CLI dispatch via `_internal/transcription/public_boundary_profiling.py`
- public-boundary runtime/setup/in-process wiring via `_internal/transcription/public_boundary_runtime.py`
- public-boundary wrapper orchestration via `_internal/transcription/public_boundary_support.py`

### Coupling assessment

- High internal complexity, but good decomposition
- `transcript_extractor.py` remains a concentration point, but most reusable
  wrapper orchestration now lives under `_internal/transcription/*`
- Profiling is well-factored into internal owners

## Data subsystem

### Primary nodes

- `data/application.py`
- `data/dataset_prepare.py`
- `data/dataset_registry.py`
- `data/catalog/public_datasets.py`
- `data/strategies/*`
- `data/adapters/*`
- `_internal/data/application/*`

### Inbound dependencies

- `ser.api`
- CLI data/configure flows
- diagnostics service for registry health checks

### Outbound dependencies

- download providers
- manifest builders
- label ontology
- consent and registry persistence

### Coupling assessment

- Good separation between declarative catalog, strategy behavior, and application workflows
- Strong architectural shape with low cross-subsystem surprise

## Model subsystem

### Primary nodes

- `models/emotion_model.py`
- `models/training_entrypoints.py`
- `models/accurate_training_preparation.py`
- `models/accurate_training_execution.py`
- `models/training_preparation.py`
- `models/training_execution.py`
- `models/training_support.py`
- `models/training_types.py`
- `models/fast_training.py`
- `_internal/models/model_loading.py`
- `_internal/models/fast_training_entrypoints.py`

### Inbound dependencies

- `runtime/pipeline.py`
- profile-specific inference modules
- training command/workflow paths

### Outbound dependencies

- artifact loading/persistence modules
- dataset loading
- runtime env planning
- sklearn model creation/evaluation

### Coupling assessment

- `emotion_model.py` is now a thin public boundary rather than a primary hotspot
- Public training wrappers and shared preparation, execution, reporting, and
  boundary helpers are split across `emotion_model.py`,
  `training_entrypoints.py`, `accurate_training_preparation.py`,
  `accurate_training_execution.py`, `training_preparation.py`,
  `training_execution.py`, `training_support.py`, and `training_types.py`
- Ambient settings lookup in the public API, data, feature, model, and
  transcription convenience wrappers is now consolidated behind one private
  boundary resolver per module
- Primary remaining risk is boundary drift or subsystem reconcentration, not
  one single owner module

Obsolete references:

- `ser/models/training_orchestration.py` is not present in this workspace
- `ser/_internal/apt/runtime.py` is not present in this workspace

## Diagnostics subsystem

### Primary nodes

- `diagnostics/service.py`
- `_internal/api/diagnostics.py`
- `_internal/cli/diagnostics.py`

### Inbound dependencies

- `ser.__main__`
- `ser.api`

### Outbound dependencies

- runtime capability registry
- transcription compatibility
- dataset registry health
- config/profile resolution

### Coupling assessment

- Good service-oriented design
- Strong operational value for a runtime-heavy ML tool

## Cross-cutting shared primitives

- Config schema and bootstrap: `_internal/config/*`
- Runtime worker lifecycle and retries: `_internal/runtime/*`, `runtime/policy.py`, `runtime/retry_primitives.py`
- Domain contracts: `domain.py`, `runtime/contracts.py`, `runtime/schema.py`
- Logging and timing: `utils/logger.py`, `runtime/phase_timing.py`

These are the architectural backbone. They reduce repeated operational logic and
make the system more testable.

## Concrete coupling observations

- Public modules importing `_internal` directly: `24`
- Public modules resolving default settings snapshots internally: concentrated in boundary-local helpers across API, models, transcript, data loader, diagnostics, quality gate, and utility paths

This means the system is not hard-layered. It is intentionally layered, but some
subsystems still know about their owner modules directly instead of depending on
smaller abstract service boundaries.

## Explicit soft-boundary allowlist

Public modules allowed to import `_internal` directly by design are defined in
[`boundary_policy.toml`](../boundary_policy.toml). That file is authoritative:
[`tests/suites/integration/architecture/test_api_import_boundary.py`](../tests/suites/integration/architecture/test_api_import_boundary.py)
loads it directly, and `make import-lint` enforces the same contract.

Each `[[public_internal_import]]` entry in `boundary_policy.toml` carries both
the allowed module path and the rationale for why that boundary exception
exists. Keep the file sorted by path and update it in the same change packet as
any intentional widening or narrowing of the soft boundary.

## Dependency risk summary

Low risk:

- config assembly and schema layering
- runtime environment planning
- dataset catalog and strategy organization
- diagnostics service separation

Medium risk:

- public-to-`_internal` imports as a soft boundary
- boundary-level settings fallback via `reload_settings()`, even though direct
  `get_settings()` lookups are gone from source modules
- transcription orchestrator concentration

Highest architectural risk:

- public boundary drift in `runtime/accurate_inference.py`,
  `runtime/medium_inference.py`, and `transcript/transcript_extractor.py`,
  plus convenience-layer settings fallback that still stops short of full DI
