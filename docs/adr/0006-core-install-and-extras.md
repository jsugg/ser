# ADR 0006: Core install and extras

Status: Accepted (D15).

The base package installs the CPU-only `fast` emotion workflow, including the
utterance analyser and batch CLI. It does not install PyTorch or speech-to-text.
`transcript` installs the transcription stack, `medium` adds the Torch and
Transformers emotion backends, and `full` includes both plus the research backend
and corpus tooling.
`all` aliases `full`; `dev` adds development tools to `all`.
The shared fast CLI and training-readiness path still uses `psutil`.

Extras can only add dependencies. The first tagged release will be `v1.1.0`
with the core as the default install. Existing transcription calls still request
transcripts by default and fail with an install hint if the extra is absent.
For emotion-only inference, pass `--no-transcript` or
`include_transcript=False`. The Python export surface is unchanged.

The Intel macOS constraints remain: the Torch-based paths use the Intel Torch
pin and NumPy 1 on Python 3.12. On Intel Python 3.13, use the `fast` emotion
profile and, if needed, the faster-whisper transcription backend.
