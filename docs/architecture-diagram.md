# SER Architecture Diagram

This document presents a text-form architecture diagram of the current codebase.

## Level 1: external interaction

```text
User / Library Caller
        |
        +--------------------------+
        |                          |
        v                          v
   ser CLI                    ser.api / ser.config
   ser/__main__.py            public library facade
        |                          |
        v                          v
  _internal/cli/*             _internal/api/*
        \                          /
         \                        /
          +------ workflow -------+
                     |
                     v
              runtime / data / diagnostics owners
```

## Level 2: runtime execution

```text
_internal/api/runtime.py
        |
        v
runtime/pipeline.py
        |
        +--> runtime/registry.py
        +--> runtime/backend_hooks.py
        +--> _internal/runtime/environment_plan.py
        +--> _internal/runtime/process_env.py
        |
        +--> fast_inference.py
        +--> medium_inference.py
        |      +--> medium_execution_context.py
        |      +--> medium_execution_flow.py
        |      +--> medium_execution.py
        |      +--> medium_worker_operation.py
        |      +--> medium_worker_lifecycle.py
        |      +--> medium_runtime_support.py
        |
        +--> accurate_inference.py
               +--> accurate_execution_flow.py
               +--> accurate_execution.py
               +--> accurate_operation_setup.py
               +--> accurate_worker_operation.py
               +--> accurate_worker_lifecycle.py
               +--> accurate_runtime_support.py
```

Shared runtime primitives:

```text
_internal/runtime/worker_lifecycle.py
_internal/runtime/retry_scaffold.py
runtime/policy.py
runtime/retry_primitives.py
```

## Level 3: transcription execution

```text
runtime/pipeline.py
        |
        v
transcript/transcript_extractor.py
        |
        +--> _internal/transcription/runtime_profile.py
        +--> _internal/transcription/compatibility.py
        +--> transcript/backends/*
        |
        +--> in-process path
        |      _internal/transcription/in_process_orchestration.py
        |
        +--> process-isolated path
               _internal/transcription/process_isolation.py
               _internal/transcription/process_worker.py
```

## Level 3.5: model training and loading

```text
runtime/pipeline.py
        |
        v
models/emotion_model.py
        |
        +--> models/training_entrypoints.py
        +--> models/accurate_training_execution.py
        +--> models/training_support.py
        +--> models/training_preparation.py
        +--> models/training_execution.py
        +--> models/training_types.py
        +--> _internal/models/fast_training_entrypoints.py
        +--> _internal/models/model_loading.py
```

Profiling and calibration:

```text
transcript/profiling.py
        |
        +--> _internal/transcription/default_benchmark.py
        +--> _internal/transcription/default_recommendation.py
        +--> _internal/transcription/profile_candidates.py
        +--> _internal/transcription/ravdess_references.py
        +--> _internal/transcription/text_metrics.py
        +--> _internal/transcription/runtime_calibration_workflow.py
```

## Level 4: data preparation

```text
ser.api.prepare_dataset()
        |
        v
_internal/api/data.py
        |
        v
data/application.py
        |
        +--> _internal/data/application/prepare.py
        +--> _internal/data/application/registry_snapshot.py
        +--> _internal/data/application/capability_snapshot.py
        |
        v
data/dataset_prepare.py
        |
        +--> data/catalog/public_datasets.py
        +--> data/strategies/default.py
        +--> data/strategies/auto_csv.py
        +--> data/adapters/*
        +--> data/dataset_registry.py
```

## Level 5: configuration and profile model

```text
environment variables + profile_defs.yaml
                |
                v
_internal/config/settings_inputs.py
                |
                v
_internal/config/settings_builder.py
                |
                v
_internal/config/schema.py -> AppConfig
                |
                v
_internal/config/bootstrap.py
                |
                +--> get_settings()
                +--> reload_settings()
                +--> settings_override()
```

Profile catalog:

```text
profiles.py
   |
   +--> backend id
   +--> required modules
   +--> transcription defaults
   +--> runtime defaults
   +--> feature-runtime defaults
```

## Design summary

- The system is a modular monolith.
- `ser.api` and `ser.config` are stable external boundaries.
- `_internal/*` contains owner modules and shared implementation scaffolding.
- `runtime/pipeline.py` is the central orchestration seam.
- `profiles.py` is the architectural policy source for runtime selection.
- Data and transcription remain the most decomposed subsystems, but the model
  subsystem now follows the same boundary-owner pattern more closely.
