# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
after its first published distribution.

## [Unreleased]

### Added

- Public API stability policy for tier-1 modules.
- README Python API example coverage through a contract test.
- Changelog discipline before first publish.
- Internal masked uncertainty-weighted multitask loss for auxiliary training
  heads, combining only the targets each sample actually carries.
- Internal hierarchical utterance sampling primitives: square-root corpus mass,
  inverse-square-root class mass, and bounded seeded window selection.
- Feature backends resolve and expose the exact model revision they loaded, and
  accept a `repository@revision` model identifier to pin it.
- Manifest schema v2: categorical emotion is now optional, and rows can carry
  normalized VAD, social attitude, binary affect, language, transcript, per
  target source/confidence, speaker/session identity, native split, segment
  bounds, normalized-PCM SHA-256, dataset revision, license, and provenance.
  Schema v1 manifests remain readable.
- Versioned dataset recipes selected with `--dataset-recipe` or
  `SER_DATASET_RECIPE`, including the built-in `research-v1`. Recipes declare
  per-corpus label and task routing, and are audited into a leakage-safe split
  ledger with reproducible recipe, manifest, and ledger digests.
  Strictness is controlled by `--strict-dataset-audit` / `SER_STRICT_DATASET_AUDIT`.
- Model artifact envelope v3, persisting recipe and split-ledger digests, model
  revision, task heads, sampling policy, seed, and evaluation summary. Envelope
  v2 artifacts still load, and the public eight-emotion inference schema is
  unchanged.
- Mandatory training readiness contract on every CLI and library training
  entrypoint, with `--dry-run`, `--prepare-only`, `--prepared-plan`, and
  `--repair` modes documented in `docs/training-readiness.md`.
- Bias-aware quarantine budgets through `SER_MAX_FAILED_FILES`,
  `SER_MAX_FAILED_FILE_RATIO_PER_CORPUS`, `SER_MAX_FAILED_FILE_RATIO_PER_CLASS`,
  `SER_MAX_FAILURES_PER_REASON`, `SER_MIN_REMAINING_PER_CLASS_SPLIT`, and
  `SER_STRICT_QUARANTINE`.
- `SER_DEV_SIZE` (default `0.10`) reserves a deterministic, speaker-isolated
  development partition before windowing.

### Changed

- Audio read failures now raise the typed `AudioIntegrityError` and
  `AudioDecodeError` instead of a bare `OSError`, and preserve the underlying
  cause. Both subclass `OSError`, so existing handling keeps working.
- Preparing CREMA-D now requires Git and Git LFS up front and fails with an
  actionable message when either is missing.
- Training validation and policy failures now exit `2`; unexpected internal
  failures continue to exit `1` with a traceback.
- `ser doctor` and the training preflight now include training readiness checks.
- README Python API guidance now directs workflow users to `ser.api`.

### Fixed

- Unmaterialized Git LFS pointer files are now reported as a dataset integrity
  problem naming the repair steps, instead of surfacing as an opaque audio
  decode failure during feature extraction.
- CREMA-D audio is validated before manifest registration, so an incomplete
  checkout fails during dataset preparation rather than mid-training.
- Speaker partitioning during dataset splitting is now linear rather than
  quadratic, so large inventories no longer appear to hang.
- Import lint path collection now works on shells without `mapfile`, such as the
  Bash 3.2 that ships with macOS.

[Unreleased]: https://github.com/jsugg/ser/compare/HEAD...HEAD
