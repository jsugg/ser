# SER Codebase Architecture

This document summarizes the current architecture of the `ser` codebase from the
source itself. It is intended as the narrative overview companion to the more
formal artifacts in [`docs/architecture-diagram.md`](architecture-diagram.md),
[`docs/subsystem-dependency-map.md`](subsystem-dependency-map.md), and
[`docs/refactor-hotspot-checks.md`](refactor-hotspot-checks.md).

## Scope and current state

These counts are a current working-tree snapshot taken on March 12, 2026.

- Source modules under `ser/`: `226`
- Test modules under `tests/`: `136`
- Public modules outside `_internal/`: `166`
- Internal owner/helper modules under `_internal/`: `60`
- Public modules importing `_internal` directly: `24`

This is a modular monolith with explicit subsystem seams. It is not a textbook
hexagonal architecture, but it is clearly designed: configuration is typed and
immutable, runtime behavior is profile-driven, and operationally sensitive code
paths use extracted owner modules for retries, process isolation, environment
application, and diagnostics.

Older roadmap references to `ser/models/training_orchestration.py` and
`ser/_internal/apt/runtime.py` do not match the current tree; neither module is
present in this workspace.

## Architectural style

The dominant architectural pattern is:

1. thin public facades for stable entrypoints
2. `_internal` owner modules for orchestration or shared primitives
3. profile- and backend-specific execution modules for heavy runtime behavior
4. typed configuration and schema models as the shared contract layer
5. strong architectural tests and contract gates in CI

Representative boundaries:

- Public API: [`ser/api.py`](../ser/api.py)
- Public config: [`ser/config.py`](../ser/config.py)
- CLI entrypoint: [`ser/__main__.py`](../ser/__main__.py)
- Runtime pipeline seam: [`ser/runtime/pipeline.py`](../ser/runtime/pipeline.py)
- Internal API owners: [`ser/_internal/api/runtime.py`](../ser/_internal/api/runtime.py), [`ser/_internal/api/data.py`](../ser/_internal/api/data.py), [`ser/_internal/api/diagnostics.py`](../ser/_internal/api/diagnostics.py)

## Top-level subsystem map

### 1. Configuration and profile resolution

Configuration is assembled through a clean internal pipeline:

- input capture and normalization: [`ser/_internal/config/settings_inputs.py`](../ser/_internal/config/settings_inputs.py)
- immutable `AppConfig` construction: [`ser/_internal/config/settings_builder.py`](../ser/_internal/config/settings_builder.py)
- typed schema definitions: [`ser/_internal/config/schema.py`](../ser/_internal/config/schema.py)
- runtime-scoped settings state: [`ser/_internal/config/bootstrap.py`](../ser/_internal/config/bootstrap.py)

Profile behavior is catalog-driven rather than hardcoded:

- profile catalog and YAML-backed definitions: [`ser/profiles.py`](../ser/profiles.py)

This is one of the strongest parts of the design. The code treats config as
data, not as a bag of mutable globals. `settings_override()` uses `ContextVar`
state, which is significantly safer than direct process-global mutation.

Tradeoff:

- the architecture still offers boundary-level fallback config resolution via
  `reload_settings()`. Direct `get_settings()` lookups are now removed from
  source modules, but full dependency injection is still incomplete because
  optional settings remain part of several public APIs.

## 2. Public API and CLI architecture

The public library surface is intentionally small:

- [`ser/api.py`](../ser/api.py)
- [`ser/config.py`](../ser/config.py)
- [`ser/__init__.py`](../ser/__init__.py)

The CLI delegates through internal CLI support modules instead of talking
directly to the library facade:

- [`ser/__main__.py`](../ser/__main__.py)
- [`ser/_internal/cli/runtime.py`](../ser/_internal/cli/runtime.py)
- [`ser/_internal/cli/data.py`](../ser/_internal/cli/data.py)
- [`ser/_internal/cli/diagnostics.py`](../ser/_internal/cli/diagnostics.py)

Those CLI modules then delegate into internal API owners:

- [`ser/_internal/api/runtime.py`](../ser/_internal/api/runtime.py)
- [`ser/_internal/api/data.py`](../ser/_internal/api/data.py)
- [`ser/_internal/api/diagnostics.py`](../ser/_internal/api/diagnostics.py)

This is good boundary design. The public library API stays narrower than the CLI
workflow surface, and CLI-specific orchestration does not leak back into the
library facade.

## 3. Runtime inference architecture

The central orchestration seam is [`ser/runtime/pipeline.py`](../ser/runtime/pipeline.py).
It coordinates:

- profile resolution
- backend capability checks
- environment application
- training and inference execution
- optional transcription
- timeline generation

Supporting runtime ownership is split between:

- backend hook construction: [`ser/runtime/backend_hooks.py`](../ser/runtime/backend_hooks.py)
- capability resolution: [`ser/runtime/registry.py`](../ser/runtime/registry.py)
- retry policy: [`ser/runtime/policy.py`](../ser/runtime/policy.py)
- shared retry/error primitives: [`ser/runtime/retry_primitives.py`](../ser/runtime/retry_primitives.py)
- process env planning: [`ser/_internal/runtime/environment_plan.py`](../ser/_internal/runtime/environment_plan.py)
- scoped env application: [`ser/_internal/runtime/process_env.py`](../ser/_internal/runtime/process_env.py)

The runtime design is profile-aware, not fully generic. Each profile still has
its own execution path:

- fast: [`ser/runtime/fast_inference.py`](../ser/runtime/fast_inference.py)
- medium: [`ser/runtime/medium_inference.py`](../ser/runtime/medium_inference.py)
- accurate: [`ser/runtime/accurate_inference.py`](../ser/runtime/accurate_inference.py)
- accurate-research: [`ser/runtime/accurate_research_inference.py`](../ser/runtime/accurate_research_inference.py)

The important architectural improvement is that medium and accurate are no
longer monolithic runtime modules. They now delegate heavily into owner modules:

- public-boundary orchestration: [`ser/_internal/runtime/accurate_public_boundary.py`](../ser/_internal/runtime/accurate_public_boundary.py), [`ser/_internal/runtime/medium_public_boundary.py`](../ser/_internal/runtime/medium_public_boundary.py)
- execution: [`ser/runtime/medium_execution.py`](../ser/runtime/medium_execution.py), [`ser/runtime/accurate_execution.py`](../ser/runtime/accurate_execution.py)
- setup/context: [`ser/runtime/medium_execution_context.py`](../ser/runtime/medium_execution_context.py), [`ser/runtime/accurate_operation_setup.py`](../ser/runtime/accurate_operation_setup.py)
- retry/execution flow: [`ser/runtime/medium_execution_flow.py`](../ser/runtime/medium_execution_flow.py), [`ser/runtime/accurate_execution_flow.py`](../ser/runtime/accurate_execution_flow.py)
- worker lifecycle: [`ser/runtime/medium_worker_lifecycle.py`](../ser/runtime/medium_worker_lifecycle.py), [`ser/runtime/accurate_worker_lifecycle.py`](../ser/runtime/accurate_worker_lifecycle.py)
- runtime support: [`ser/runtime/medium_runtime_support.py`](../ser/runtime/medium_runtime_support.py), [`ser/runtime/accurate_runtime_support.py`](../ser/runtime/accurate_runtime_support.py)

Underneath those, shared internal worker primitives provide the real reusable
mechanics:

- [`ser/_internal/runtime/worker_lifecycle.py`](../ser/_internal/runtime/worker_lifecycle.py)
- [`ser/_internal/runtime/retry_scaffold.py`](../ser/_internal/runtime/retry_scaffold.py)

This is a good example of pragmatic architecture: profile-local boundary modules
remain, but the complex reusable mechanics are extracted and tested separately.

## 4. Model and training architecture

The model subsystem is centered on [`ser/models/emotion_model.py`](../ser/models/emotion_model.py),
which is now a thin public boundary rather than a primary hotspot. It owns:

- fast, medium, accurate, and accurate-research training entrypoints
- model loading and compatibility filtering
- inference helpers

There has been meaningful decomposition:

- fast training workflow owner: [`ser/models/fast_training.py`](../ser/models/fast_training.py)
- public training-entrypoint wiring: [`ser/models/training_entrypoints.py`](../ser/models/training_entrypoints.py)
- accurate-profile preparation and entrypoint helpers: [`ser/models/accurate_training_preparation.py`](../ser/models/accurate_training_preparation.py)
- accurate-profile prepared-training execution assembly: [`ser/models/accurate_training_execution.py`](../ser/models/accurate_training_execution.py)
- shared training preparation/evaluation: [`ser/models/training_preparation.py`](../ser/models/training_preparation.py)
- shared training execution/reporting: [`ser/models/training_execution.py`](../ser/models/training_execution.py)
- shared boundary helpers: [`ser/models/training_support.py`](../ser/models/training_support.py)
- shared training contracts: [`ser/models/training_types.py`](../ser/models/training_types.py)
- fast training entrypoint seam: [`ser/_internal/models/fast_training_entrypoints.py`](../ser/_internal/models/fast_training_entrypoints.py)
- shared model loading entrypoint: [`ser/_internal/models/model_loading.py`](../ser/_internal/models/model_loading.py)

This subsystem is now materially closer to the runtime/data architecture shape:
[`ser/models/emotion_model.py`](../ser/models/emotion_model.py) stays
boundary-only, while profile-specific entrypoint wiring and shared
cross-profile logic live in dedicated `training_*` owner modules, and
accurate-profile preparation is now separated from accurate-profile
prepared-training execution between
[`ser/models/accurate_training_preparation.py`](../ser/models/accurate_training_preparation.py)
and
[`ser/models/accurate_training_execution.py`](../ser/models/accurate_training_execution.py)
instead of reconcentrating inside one mixed-responsibility owner. Internal
callers and tests now use canonical owner modules directly instead of routing
through a compatibility namespace on `emotion_model.py`. The remaining risk is
boundary drift or reconcentration, not alias sprawl.

## 5. Transcription architecture

Transcription is one of the most mature architectural subsystems.

Public orchestration lives in:

- [`ser/transcript/transcript_extractor.py`](../ser/transcript/transcript_extractor.py)

Public-boundary wrapper orchestration now also lives in:

- [`ser/_internal/transcription/public_boundary_support.py`](../ser/_internal/transcription/public_boundary_support.py)

Execution strategy is split between:

- in-process orchestration: [`ser/_internal/transcription/in_process_orchestration.py`](../ser/_internal/transcription/in_process_orchestration.py)
- process isolation: [`ser/_internal/transcription/process_isolation.py`](../ser/_internal/transcription/process_isolation.py)
- process-worker payload/runtime cleanup: [`ser/_internal/transcription/process_worker.py`](../ser/_internal/transcription/process_worker.py)
- public-boundary process wiring: [`ser/_internal/transcription/public_boundary_process.py`](../ser/_internal/transcription/public_boundary_process.py)
- public-boundary runtime/setup/in-process wiring: [`ser/_internal/transcription/public_boundary_runtime.py`](../ser/_internal/transcription/public_boundary_runtime.py)
- runtime-profile resolution: [`ser/_internal/transcription/runtime_profile.py`](../ser/_internal/transcription/runtime_profile.py)
- compatibility handling: [`ser/_internal/transcription/compatibility.py`](../ser/_internal/transcription/compatibility.py)

Backend abstraction lives behind:

- [`ser/transcript/backends/__init__.py`](../ser/transcript/backends/__init__.py)

Profiling and recommendation are also decomposed:

- public profiling orchestrator: [`ser/transcript/profiling.py`](../ser/transcript/profiling.py)
- public-boundary profiling wiring and CLI dispatch: [`ser/_internal/transcription/public_boundary_profiling.py`](../ser/_internal/transcription/public_boundary_profiling.py)
- benchmark owner: [`ser/_internal/transcription/default_benchmark.py`](../ser/_internal/transcription/default_benchmark.py)
- recommendation owner: [`ser/_internal/transcription/default_recommendation.py`](../ser/_internal/transcription/default_recommendation.py)
- text metrics: [`ser/_internal/transcription/text_metrics.py`](../ser/_internal/transcription/text_metrics.py)
- runtime calibration workflow: [`ser/_internal/transcription/runtime_calibration_workflow.py`](../ser/_internal/transcription/runtime_calibration_workflow.py)

Architecturally, transcription is well designed for operational complexity. It
explicitly models compatibility, setup, model load, process isolation, and
calibration rather than burying those concerns in one function.

## 6. Data architecture

The data subsystem is organized around a useful combination of facades,
application workflows, catalog data, strategy objects, and file-format adapters.

Public/application layer:

- [`ser/data/application.py`](../ser/data/application.py)
- [`ser/_internal/data/application/prepare.py`](../ser/_internal/data/application/prepare.py)
- [`ser/_internal/data/application/registry_snapshot.py`](../ser/_internal/data/application/registry_snapshot.py)

Preparation and registry orchestration:

- [`ser/data/dataset_prepare.py`](../ser/data/dataset_prepare.py)
- [`ser/data/dataset_registry.py`](../ser/data/dataset_registry.py)

Catalog and strategies:

- declarative catalog: [`ser/data/catalog/public_datasets.py`](../ser/data/catalog/public_datasets.py)
- strategy registry: [`ser/data/strategies/default.py`](../ser/data/strategies/default.py)
- extracted homogeneous CSV strategies: [`ser/data/strategies/auto_csv.py`](../ser/data/strategies/auto_csv.py)

Adapters:

- [`ser/data/adapters/public_csv_datasets.py`](../ser/data/adapters/public_csv_datasets.py)
- [`ser/data/adapters/csv_manifest_builder.py`](../ser/data/adapters/csv_manifest_builder.py)

This subsystem is architecturally strong. Metadata is declarative, behavior is
attached to strategies, and application workflows are separated from low-level
registry and manifest concerns.

## 7. Diagnostics and operational gates

Diagnostics are treated as a first-class subsystem:

- diagnostics service: [`ser/diagnostics/service.py`](../ser/diagnostics/service.py)
- diagnostics API owner: [`ser/_internal/api/diagnostics.py`](../ser/_internal/api/diagnostics.py)
- CLI doctor/preflight support: [`ser/_internal/cli/diagnostics.py`](../ser/_internal/cli/diagnostics.py)

This is a good design choice for an ML/runtime-heavy project. Runtime admission,
dependency capability, ffmpeg availability, transcription compatibility, and
dataset registry health are modeled as structured diagnostics rather than ad hoc
log spam.

## Boundary discipline and coupling

The codebase has real architectural discipline, but it is a soft discipline.

Strengths:

- narrow public API in [`ser/api.py`](../ser/api.py)
- stable public config facade in [`ser/config.py`](../ser/config.py)
- boundary tests in [`tests/test_api_import_boundary.py`](../tests/test_api_import_boundary.py)
- documented hotspot inventory in [`docs/refactor-hotspot-checks.md`](refactor-hotspot-checks.md)
- CI lanes enforcing lint, typing, tests, contracts, and build in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

Limits:

- the `_internal` boundary is not universally enforced
- some public modules still import `_internal` directly by design
- several public boundaries still offer optional settings fallbacks for
  convenience

This is a pragmatic boundary model. It is maintainable, but it depends on team
discipline more than on absolute architectural isolation.

## Design patterns in active use

- Facade: `ser.api`, `ser.config`, `ser.data.application`
- Strategy: dataset strategy registry and dataset-specific strategy classes
- Adapter: transcription adapters, representation backends, backend hooks
- Catalog/config-driven runtime selection: profile YAML definitions
- Owner-module extraction: runtime and transcription helper modules
- Scoped context execution: `settings_override`, temporary process env
- Contract testing: API snapshot, import boundary, import-lint policy

## Strongest architectural qualities

- typed immutable configuration pipeline
- declarative profile catalog
- explicit runtime pipeline seam
- strong process-isolation and retry abstractions
- mature transcription subsystem
- clean dataset catalog/strategy design
- unusually strong architectural testing and CI contract culture

## Main architectural liabilities

- [`ser/runtime/accurate_inference.py`](../ser/runtime/accurate_inference.py) and [`ser/runtime/medium_inference.py`](../ser/runtime/medium_inference.py) remain large public wrappers because they preserve public/runtime signatures, exception surfaces, and worker-entry orchestration seams, even though most compute, retry, and pooling ownership now lives in [`ser/_internal/runtime/accurate_public_boundary.py`](../ser/_internal/runtime/accurate_public_boundary.py), [`ser/_internal/runtime/medium_public_boundary.py`](../ser/_internal/runtime/medium_public_boundary.py), [`ser/runtime/accurate_execution.py`](../ser/runtime/accurate_execution.py), and [`ser/runtime/medium_execution.py`](../ser/runtime/medium_execution.py)
- [`ser/transcript/transcript_extractor.py`](../ser/transcript/transcript_extractor.py) remains a large public boundary, but process-isolation, runtime/setup, and wrapper orchestration now route through [`ser/_internal/transcription/public_boundary_process.py`](../ser/_internal/transcription/public_boundary_process.py), [`ser/_internal/transcription/public_boundary_runtime.py`](../ser/_internal/transcription/public_boundary_runtime.py), and [`ser/_internal/transcription/public_boundary_support.py`](../ser/_internal/transcription/public_boundary_support.py)
- [`ser/__main__.py`](../ser/__main__.py) is still a long CLI composition function
- optional boundary-level `reload_settings()` fallbacks remain, so DI is still
  incomplete even though direct `get_settings()` calls have been removed from
  source modules
- `_internal` is a convention-backed boundary, not a sealed dependency boundary
- architecture docs require active maintenance to stay current

## Judgment

This is a good architecture for a production-oriented ML/runtime tool: a
disciplined modular monolith with clear subsystem seams, strong tests, and
substantial refactoring maturity. It is not fully “clean architecture”, but it
is intentionally designed, operationally defensive, and moving in the right
direction. The runtime and data subsystems are in particularly good shape; the
main remaining concentration risk is public-boundary glue in runtime and
transcription, not the model subsystem.
