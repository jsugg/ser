# Training readiness, preparation, and fault containment

Every CLI and library training entrypoint runs the same mandatory readiness contract before a
classifier is created or fitted. `--preflight off` disables only the general startup preflight; it
does not disable training readiness.

## Commands

```bash
ser --train --profile medium --dry-run
ser --train --profile medium --dry-run --repair
ser --train --profile medium --prepare-only
ser --train --profile medium --prepare-only --repair
ser --train --profile medium --prepared-plan ~/.cache/ser/tmp/prepared-training-medium.json
```

- `--dry-run` validates configuration, registry/media integrity, split feasibility, write paths,
  disk/file-descriptor capacity, and a deterministic bounded real-backend sample. It writes
  `training-readiness-<profile>.json` under `SER_TMP_DIR` and never creates/fits a classifier,
  populates production embedding caches, writes a quarantine JSONL ledger, or writes model
  artifacts. Quarantine findings and audit timestamps remain in the atomic readiness report.
- `--prepare-only` first runs readiness, then performs complete feature/cache preparation. It
  atomically writes a non-pickle NPZ feature payload followed by
  `prepared-training-<profile>.json`. A failed/interrupted payload publication never produces a
  ready plan. Re-running is safe and reuses valid embedding-cache entries.
- `--prepared-plan` verifies the canonical plan digest, feature-payload digest, profile/settings,
  backend/model revision, device/dtype, registry, manifests, media fingerprints, recipe, split and
  quarantine ledgers, code-owned cache namespace/version, and content-bound keys. Backend plans
  require the exact commit resolved by the checked backend; an unverified mutable model identifier
  cannot produce or reuse a plan. Plans and payloads also bind actual train/dev/test sample and
  window ledgers plus included/quarantined/dropped dispositions, shapes, dtypes, labels, and current
  effective sample IDs. Any mismatch exits with validation status `2`; valid plans bypass completed
  feature extraction.
- `--repair` is accepted only with `--dry-run` or `--prepare-only`. Repairs are explicit,
  idempotent, recorded in the readiness report, and revalidated. The allowlist creates missing
  application-owned directories, removes abandoned application staging/probe files, invalidates
  corrupt NPZ entries only inside fixed application-owned cache namespaces, rebuilds missing
  manifests from intact registered sources,
  and hydrates compatible Git LFS checkouts. A revision-pinned Hugging Face model may be
  redownloaded only when `SER_TRAINING_REPAIR_ALLOW_NETWORK=1` explicitly permits network access.
  It never deletes datasets, changes labels or thresholds, accepts licenses/consents, or
  overwrites a valid model artifact.

Conflicting or non-training options fail during argument validation, before registry, backend, or
feature work. Validation/policy failures exit `2`; unexpected internal failures exit `1` with a
traceback.

## Fault containment

Failures carry a stable scope, reason, severity, and disposition. Git LFS pointers, invalid
configuration/manifests, leakage, insufficient class support, non-finite backend output, resource
shortages, and unknown exceptions always abort. Medium and accurate profiles may quarantine only
known sample-local decode failures or an exact missing media path proven inside a registered root.
Generic/cache/model `OSError` and unproven missing paths abort. Every proposed exclusion must satisfy
the absolute, global, per-corpus, per-class, per-reason, and minimum remaining class/split budgets.
Strict quarantine mode disables these exclusions.

The compatibility ratio `SER_MAX_FAILED_FILE_RATIO` remains supported. Bias-aware controls are
available through `SER_MAX_FAILED_FILES`, `SER_MAX_FAILED_FILE_RATIO_PER_CORPUS`,
`SER_MAX_FAILED_FILE_RATIO_PER_CLASS`, `SER_MAX_FAILURES_PER_REASON`,
`SER_MIN_REMAINING_PER_CLASS_SPLIT`, and `SER_STRICT_QUARANTINE`.

`SER_DEV_SIZE` (default `0.10`) reserves a deterministic, speaker-isolated development partition
before windowing. Together with `SER_TEST_SIZE`, it must leave a non-empty training ratio; readiness
validates minimum per-class support independently across train, dev, and test.

Quarantines are written as deterministic, bounded JSONL under `SER_TMP_DIR`; paths are represented
by SHA-256 digests. Source audio is never mutated. Silent/low-variance medium windows continue to
use the existing deterministic window-noise controls, and corrupt embedding-cache entries continue
to be atomically invalidated and recomputed. Optional secure artifact failure remains advisory only
after the primary artifact is valid; primary model/report persistence remains fatal.

## Side effects and sizing

Readiness uses a fixed sample cap and linear media inventory passes. Write/rename probes use unique
names and are removed in `finally`. Preparation concurrency remains bounded by configured worker
limits. Progress logs report processed/total, cache activity, quarantines, elapsed time, and a
credible ETA at bounded intervals. Backend smoke and permitted network repairs require hard-deadline
support; unsupported worker/platform execution is rejected rather than running unbounded. Reports,
quarantine ledgers, feature payloads, and plans use atomic publication; no partial document is
treated as ready. Quarantine ledger reuse identity excludes wall-clock fields while JSONL/report
timestamps remain available for audit.
