#!/usr/bin/env bash
set -euo pipefail

run_training="$(printf '%s' "${SER_FULL_GATE_RUN_TRAINING:-false}" | tr '[:upper:]' '[:lower:]')"
require_pass="$(printf '%s' "${SER_FULL_GATE_REQUIRE_PASS:-true}" | tr '[:upper:]' '[:lower:]')"
archive_report="$(printf '%s' "${SER_FULL_GATE_ARCHIVE_REPORT:-true}" | tr '[:upper:]' '[:lower:]')"

dataset_glob="${SER_FULL_GATE_DATASET_GLOB:-${DATASET_FOLDER:-ser/dataset/ravdess}/Actor_*/*.wav}"
fast_model_file_name="${SER_FULL_GATE_FAST_MODEL_FILE_NAME:-ser_model_fast_full.pkl}"
fast_training_report_file_name="${SER_FULL_GATE_FAST_TRAINING_REPORT_FILE_NAME:-training_report_fast_full.json}"
medium_model_file_name="${SER_FULL_GATE_MEDIUM_MODEL_FILE_NAME:-ser_model_medium_full.pkl}"
medium_training_report_file_name="${SER_FULL_GATE_MEDIUM_TRAINING_REPORT_FILE_NAME:-training_report_medium_full.json}"
report_path="${SER_FULL_GATE_REPORT_PATH:-profile_quality_gate_report_full.json}"
progress_every="${SER_FULL_GATE_PROGRESS_EVERY:-120}"
models_dir="${SER_MODELS_DIR:-$HOME/Library/Application Support/ser/models}"

if [[ "$run_training" != "true" && "$run_training" != "false" ]]; then
  printf 'SER_FULL_GATE_RUN_TRAINING must be true or false, got: %s\n' "$run_training" >&2
  exit 2
fi
if [[ "$require_pass" != "true" && "$require_pass" != "false" ]]; then
  printf 'SER_FULL_GATE_REQUIRE_PASS must be true or false, got: %s\n' "$require_pass" >&2
  exit 2
fi
if [[ "$archive_report" != "true" && "$archive_report" != "false" ]]; then
  printf 'SER_FULL_GATE_ARCHIVE_REPORT must be true or false, got: %s\n' "$archive_report" >&2
  exit 2
fi
if ! [[ "$progress_every" =~ ^[0-9]+$ ]]; then
  printf 'SER_FULL_GATE_PROGRESS_EVERY must be a non-negative integer, got: %s\n' "$progress_every" >&2
  exit 2
fi
if ! compgen -G "$dataset_glob" > /dev/null; then
  printf 'No files match dataset glob: %s\n' "$dataset_glob" >&2
  exit 2
fi

if [[ "$run_training" == "false" ]]; then
  for artifact in \
    "$fast_model_file_name" \
    "$fast_training_report_file_name" \
    "$medium_model_file_name" \
    "$medium_training_report_file_name"; do
    if [[ ! -f "$models_dir/$artifact" ]]; then
      printf 'Missing required artifact (set SER_FULL_GATE_RUN_TRAINING=true or provide artifact): %s\n' "$models_dir/$artifact" >&2
      exit 2
    fi
  done
fi

if [[ "$run_training" == "true" ]]; then
  printf '[full-gate] training fast artifact: %s\n' "$fast_model_file_name"
  SER_RANDOM_STATE=42 \
  SER_TEST_SIZE=0.25 \
  SER_MODEL_FILE_NAME="$fast_model_file_name" \
  SER_TRAINING_REPORT_FILE_NAME="$fast_training_report_file_name" \
  uv run ser --train

  printf '[full-gate] training medium artifact: %s\n' "$medium_model_file_name"
  SER_ENABLE_PROFILE_PIPELINE=true \
  SER_ENABLE_MEDIUM_PROFILE=true \
  SER_RANDOM_STATE=42 \
  SER_TEST_SIZE=0.25 \
  SER_MODEL_FILE_NAME="$medium_model_file_name" \
  SER_TRAINING_REPORT_FILE_NAME="$medium_training_report_file_name" \
  uv run --extra medium ser --train
fi

gate_cmd=(
  uv run --extra medium python -m ser.runtime.profile_quality_gate
  --dataset-glob "$dataset_glob"
  --random-state 42
  --test-size 0.25
  --fast-model-file-name "$fast_model_file_name"
  --fast-training-report-file-name "$fast_training_report_file_name"
  --medium-model-file-name "$medium_model_file_name"
  --medium-training-report-file-name "$medium_training_report_file_name"
  --progress-every "$progress_every"
  --out "$report_path"
)
if [[ "$require_pass" == "true" ]]; then
  gate_cmd+=(--require-pass)
fi

printf '[full-gate] running quality gate (report: %s)\n' "$report_path"
"${gate_cmd[@]}"

if [[ "$archive_report" == "true" ]]; then
  mkdir -p "$models_dir"
  cp "$report_path" "$models_dir/$(basename "$report_path")"
  printf '[full-gate] archived report: %s/%s\n' "$models_dir" "$(basename "$report_path")"
fi

printf '[full-gate] completed successfully\n'
