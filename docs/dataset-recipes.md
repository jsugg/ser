# Versioned cross-domain dataset recipes

Training can opt into an explicit recipe instead of concatenating every configured manifest:

```bash
ser --train --dataset-recipe research-v1 --strict-dataset-audit
```

The equivalent configuration is `SER_DATASET_RECIPE=research-v1`. Strict auditing defaults to
enabled whenever a recipe is configured; use `SER_STRICT_DATASET_AUDIT=false` only for exploratory
data repair, never benchmark generation.

## Manifest schema v2

Schema v2 keeps v1 JSONL readable and makes categorical emotion optional. Records can carry
normalized VAD, social attitude, binary affect, language, transcript, per-target source/confidence,
speaker/session identity, native split, segment bounds, normalized-PCM SHA-256, dataset revision,
license, policy, and source provenance.

Strict recipe training requires:

- a dataset revision and normalized-PCM SHA-256 on every row;
- unique sample IDs and content hashes across all corpora;
- at least two populated exact primary-emotion classes;
- zero speaker/session component or content overlap across train/dev/test; and
- exhaustive accepted/remapped/weak/dropped/missing/quarantined accounting.

Rows without a defensible speaker/session group are assigned `ssl_only`; they cannot enter
supervised validation or test. Approximate mappings are routed to `raw_emotion`, never the public
eight-class primary head. The `calm` output remains part of the runtime contract and is not inferred
from neutral or sleepy labels.

## Reproducibility artifacts

The audit emits canonical recipe, manifest, and split-ledger SHA-256 digests. Artifact envelope v3
can persist those digests with model revision, task heads, hierarchical sampling policy, seed, and
evaluation summary. The loader remains compatible with v2 envelopes; the public eight-emotion
inference schema is unchanged.

Frozen-encoder experiments can use the sampling primitives in
`ser._internal.models.utterance_sampling`: corpora receive mass proportional to the square root of
their utterance count, classes receive inverse-square-root mass within corpus, and long utterances
use a bounded seeded window subset per epoch. Evaluation should continue aggregating all windows to
one utterance prediction.
