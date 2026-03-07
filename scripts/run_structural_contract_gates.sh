#!/usr/bin/env bash
set -euo pipefail

# Structural ownership contract lane for PR-901..PR-903 extracted modules.
readonly OWNERSHIP_FILES=(
  ser/utils/dsp.py
  ser/features/feature_extractor.py
  ser/repr/handcrafted.py
  ser/models/emotion_model.py
  ser/models/artifact_metadata.py
  ser/models/artifact_persistence.py
  ser/models/artifact_loading.py
  ser/models/training_reporting.py
  ser/models/training_orchestration.py
  ser/runtime/profile_quality_gate.py
  ser/runtime/quality_gate_policy.py
  ser/runtime/quality_gate_evaluation.py
  ser/runtime/quality_gate_reporting.py
  ser/runtime/quality_gate_cli.py
  ser/api.py
  ser/_internal/api/runtime.py
  ser/_internal/api/data.py
  ser/_internal/api/diagnostics.py
  ser/__main__.py
  tests/test_feature_extractor.py
  tests/test_handcrafted_backend.py
  tests/test_emotion_model.py
  tests/test_accurate_training_artifact.py
  tests/test_provenance_metadata.py
  tests/test_profile_quality_gate.py
  tests/test_postprocessing.py
  tests/test_medium_quality_report.py
  tests/test_api.py
  tests/test_cli.py
  tests/test_api_import_boundary.py
)

readonly OWNERSHIP_TESTS=(
  tests/test_feature_extractor.py
  tests/test_handcrafted_backend.py
  tests/test_emotion_model.py
  tests/test_accurate_training_artifact.py
  tests/test_provenance_metadata.py
  tests/test_profile_quality_gate.py
  tests/test_postprocessing.py
  tests/test_medium_quality_report.py
  tests/test_api.py
  tests/test_cli.py
  tests/test_api_import_boundary.py
)

uv run ruff check "${OWNERSHIP_FILES[@]}"
uv run black --check "${OWNERSHIP_FILES[@]}"
uv run isort --check-only "${OWNERSHIP_FILES[@]}"
uv run mypy "${OWNERSHIP_FILES[@]}"
uv run pyright --pythonversion 3.12 "${OWNERSHIP_FILES[@]}"
uv run pytest -q "${OWNERSHIP_TESTS[@]}"
