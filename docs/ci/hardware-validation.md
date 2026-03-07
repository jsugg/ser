# Hardware Validation Workflows

Manual hardware validation is intentionally separated from default CI.

## MPS (GitHub-hosted)

- Workflow: `.github/workflows/macos15-mps-validation.yml`
- Trigger: `workflow_dispatch`
- Runner: `macos-15`
- Requirement: an Apple Silicon macOS 15 runner with MPS available.
  - The workflow fails fast if `torch.backends.mps.is_available()` or
    `torch.backends.mps.is_built()` is false.
- Coverage:
  - MPS-focused runtime policy tests.
  - `medium` and `accurate` train/predict smoke.
  - Optional `accurate-research` train/predict smoke.

Run with GitHub CLI:

```bash
gh workflow run .github/workflows/macos15-mps-validation.yml \
  -f python_version=3.12 \
  -f accurate_model_id=openai/whisper-tiny \
  -f run_accurate_research=false \
  -f accurate_research_model_id=iic/emotion2vec_plus_large
```

## CUDA and XPU (Self-hosted)

- Workflow: `.github/workflows/linux-selfhosted-gpu-validation.yml`
- Trigger: `workflow_dispatch`
- Runner: self-hosted Linux runners selected via JSON label inputs.
  - Default CUDA labels: `["self-hosted","linux","x64","cuda"]`
  - Default XPU labels: `["self-hosted","linux","x64","xpu"]`
- Requirements:
  - `ffmpeg` must be installed on the runner.
  - CUDA lane requires `torch.cuda.is_available() == true`.
  - XPU lane requires `torch.xpu.is_available() == true`.
- Coverage:
  - Device/policy runtime tests for selected lane.
  - `medium` and `accurate` train/predict smoke.
  - Optional `accurate-research` train/predict smoke.

Run CUDA lane:

```bash
gh workflow run .github/workflows/linux-selfhosted-gpu-validation.yml \
  -f python_version=3.12 \
  -f run_cuda=true \
  -f run_xpu=false \
  -f cuda_runner_labels_json='["self-hosted","linux","x64","cuda"]' \
  -f xpu_runner_labels_json='["self-hosted","linux","x64","xpu"]' \
  -f accurate_model_id=openai/whisper-tiny \
  -f run_accurate_research=false \
  -f accurate_research_model_id=iic/emotion2vec_plus_large
```

Run XPU lane:

```bash
gh workflow run .github/workflows/linux-selfhosted-gpu-validation.yml \
  -f python_version=3.12 \
  -f run_cuda=false \
  -f run_xpu=true \
  -f cuda_runner_labels_json='["self-hosted","linux","x64","cuda"]' \
  -f xpu_runner_labels_json='["self-hosted","linux","x64","xpu"]' \
  -f accurate_model_id=openai/whisper-tiny \
  -f run_accurate_research=true \
  -f accurate_research_model_id=iic/emotion2vec_plus_large
```

## Notes

- GitHub-hosted runner minimum for current macOS validation is `macos-15`.
- Project support policy still includes local macOS13 validation targets:
  - `darwin-x86_64-macos13-python3.12` (full profile support)
  - `darwin-x86_64-macos13-python3.13` (partial, fast profile only)
- Keep hardware lanes manual unless runtime/cost constraints change.
