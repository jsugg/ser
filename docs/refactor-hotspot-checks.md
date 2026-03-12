# Refactor Hotspot Checks

This page is a docs-only inventory of source modules that have been recently
decomposed or repeatedly called out as refactor-sensitive hotspots.

It is not executable policy, CI configuration, or an ownership map. It exists
only as a maintenance heuristic and architecture-reference aid.

## Current hotspot inventory

### Public runtime and model seams

- `ser/api.py`
- `ser/__main__.py`
- `ser/models/emotion_model.py`
- `ser/models/artifact_metadata.py`
- `ser/models/artifact_persistence.py`
- `ser/models/artifact_loading.py`
- `ser/models/training_types.py`
- `ser/models/accurate_training_execution.py`
- `ser/models/training_preparation.py`
- `ser/models/training_support.py`
- `ser/models/training_execution.py`
- `ser/models/training_reporting.py`
- `ser/runtime/profile_quality_gate.py`
- `ser/runtime/quality_gate_policy.py`
- `ser/runtime/quality_gate_evaluation.py`
- `ser/runtime/quality_gate_reporting.py`
- `ser/runtime/quality_gate_cli.py`
- `ser/runtime/medium_execution_context.py`
- `ser/runtime/medium_execution_flow.py`
- `ser/runtime/medium_process_operation.py`
- `ser/runtime/medium_retry_operation.py`
- `ser/runtime/retry_primitives.py`
- `ser/runtime/medium_runtime_support.py`
- `ser/runtime/medium_worker_lifecycle.py`
- `ser/runtime/medium_worker_operation.py`
- `ser/runtime/medium_inference.py`
- `ser/runtime/medium_execution.py`
- `ser/runtime/accurate_execution.py`
- `ser/runtime/accurate_execution_flow.py`
- `ser/runtime/accurate_operation_setup.py`
- `ser/runtime/accurate_runtime_support.py`
- `ser/runtime/accurate_worker_lifecycle.py`
- `ser/runtime/accurate_worker_operation.py`
- `ser/runtime/accurate_inference.py`
- `ser/transcript/transcript_extractor.py`

### Data and representation seams

- `ser/utils/dsp.py`
- `ser/features/feature_extractor.py`
- `ser/repr/handcrafted.py`
- `ser/data/application.py`
- `ser/data/catalog/__init__.py`
- `ser/data/catalog/public_datasets.py`
- `ser/data/adapters/csv_manifest_builder.py`
- `ser/data/adapters/public_csv_datasets.py`
- `ser/data/strategies/auto_csv.py`

### Internal helper seams tied to public boundaries

- `ser/_internal/api/runtime.py`
- `ser/_internal/api/data.py`
- `ser/_internal/api/diagnostics.py`
- `ser/_internal/cli/data.py`
- `ser/_internal/cli/diagnostics.py`
- `ser/_internal/cli/runtime.py`
- `ser/_internal/models/fast_training_entrypoints.py`
- `ser/_internal/models/model_loading.py`
- `ser/_internal/runtime/accurate_public_boundary.py`
- `ser/_internal/runtime/medium_public_boundary.py`
- `ser/_internal/transcription/default_benchmark.py`
- `ser/_internal/transcription/default_recommendation.py`
- `ser/_internal/transcription/public_boundary_profiling.py`
- `ser/_internal/transcription/public_boundary_process.py`
- `ser/_internal/transcription/public_boundary_runtime.py`
- `ser/_internal/transcription/public_boundary_support.py`
- `ser/_internal/transcription/profile_candidates.py`
- `ser/_internal/transcription/ravdess_references.py`
- `ser/_internal/transcription/text_metrics.py`
- `ser/_internal/transcription/runtime_calibration_workflow.py`
- `ser/transcript/profiling.py`

## Recently cleared or obsolete hotspot claims

- `ser/config.py` is now a narrow public facade and is no longer a current
  hotspot.
- `ser/models/emotion_model.py` is now a thin public boundary and is no longer
  a primary extraction target.
- `ser/models/training_orchestration.py` is not present in this workspace, so
  it should not be discussed as a live hotspot here.
- `ser/_internal/apt/runtime.py` is not present in this workspace, so it is not
  a valid current migration target.

## How to use this file

- Use it as a reminder of modules that deserve extra review care during refactors.
- When architecture docs mention recent hotspots, link here instead of creating executable lane-specific source-code orchestration.
- Do not treat membership here as a behavioral, runtime, packaging, or CI contract.
