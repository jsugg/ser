# SER benchmark protocol

Protocol identifier: `ser-bench-protocol`, version `1`.

This document fixes how any SER model is scored. It was written before any model was scored under
it. A change to any rule below creates a new protocol version; reports produced under different
versions are never compared with each other.

The key words "must", "must not", "is" and "are" state rules. Where a rule depends on a choice
that belongs to the project owner, the text says so and points to
[Pending owner decisions](#15-pending-owner-decisions); no rule here assumes an answer to them.

## 1. Purpose and claims

The benchmark measures how well a model assigns categorical emotion labels to single utterances of
speakers it has never heard, per dataset and per language, and how well its confidence matches its
accuracy.

A report produced under this protocol supports these claims and no others:

- the model's UAR, macro-F1, per-class recall, confusion matrix and calibration error on the test
  partition of each named dataset, with a confidence interval over speakers;
- the same figures aggregated per language, as defined in section 8.4;
- that model A beats model B, or does not, under the decision rule of section 10;
- that a model passes or fails the acceptance thresholds of section 10.4.

A report does not support any claim about:

- speech, speakers, recording conditions or languages outside the scored test partitions. Acted
  corpora say little about conversational or clinical speech;
- clinical, diagnostic or psychological validity of the labels;
- agreement with the owner's own reading of his recordings (that is measured separately, by the
  own-recordings kit of brief E3-4, and only recorded in the report);
- fairness across demographic groups (the protocol does not stratify by sex, age or accent);
- a single pooled number across datasets or languages. Pooled figures are never reported.

The only result on record before this protocol (RAVDESS accuracy 58.6%, UAR 56.8%, `fast`
profile) came from a split by clip, not by speaker, and is not comparable with any figure produced
here.

## 2. Input form

Every model is scored through the public utterance-analysis call of SER (the call defined by
workstream WS1), so the benchmark measures what a consumer calls. The harness does not call
internal model classes and does not aggregate windows itself.

For every scored row the harness:

1. decodes the source file (or the row's declared segment) to floating-point samples;
2. mixes all channels to one by the arithmetic mean of the channels, sample by sample;
3. resamples to 16,000 Hz with the canonical resampler named in the bundle manifest
   (`resampler` field, with its library version); a file already at 16,000 Hz is not resampled;
4. passes the resulting mono float32 array and the sample rate 16,000 to the public call, with one
   loaded analyser reused for all rows of a run;
5. records the returned status and the probability vector over the model's label set.

A row whose call returns any status other than success, raises, or exceeds the call timeout is a
**failed row**. Failed rows stay in the scored set (section 7.2).

The one exception is the native-rate baseline row of section 9, which feeds the file's own
sample rate; it is marked `input.sample_rate = "native"` in its report.

### 2.1 Audio content identity

Duplicate detection and the leakage audit compare audio by the content hash contract
`pcm16k-s16le-v1`:

1. steps 1 to 3 above, using the same resampler;
2. clip each sample to [-1.0, 1.0], multiply by 32767, round half to even to an integer;
3. serialise as signed 16-bit little-endian integers, no header;
4. SHA-256 of those bytes, lowercase hexadecimal: the row's `audio_sha256`.

A change of resampler or of any step above is a new contract identifier and requires recomputing
every hash in the bundle. This contract detects identical audio only; re-encoded or trimmed copies
of the same recording are not detected (section 13).

## 3. Datasets and roles

Roles:

- **train/dev/test**: split by speaker as in section 5; train and dev are available to model
  builders, test is scored.
- **test-only**: every eligible row is in test; no row of the dataset is used to build or tune a
  model.
- **excluded**: not in the bundle; not scored.

| Dataset id | Language | Kind | Speaker identity | Licence (catalogue policy) | Role under this protocol | Pending owner decision |
|---|---|---|---|---|---|---|
| `ravdess` | English | acted, 24 actors | yes (actor in filename) | CC BY-NC-SA 4.0 (non-commercial) | train/dev/test if P2 admits non-commercial training data; otherwise test-only or excluded | P2 |
| `crema-d` | English | acted, 91 actors | yes (actor in filename) | ODbL (open) | train/dev/test | none (English is in every option of P1) |
| `mesd` | Spanish (Mexican) | acted | to be verified by brief D4-2 | CC BY 4.0 (open) | train/dev/test if D4-2 verifies individual speaker identity; test-only (speaker-unverified) otherwise | P1 |
| `spanish-meacorpus-2023` | Spanish | natural | to be verified by D4-2 | CC BY-NC 4.0 (non-commercial) | train/dev/test if P2 admits non-commercial training data and D4-2 verifies speaker identity; test-only if P2 admits it only for evaluation; excluded otherwise | P1, P2 |
| `escorpus-pe` | Spanish (Peruvian) | — | — | CC BY 4.0 (open) | excluded: the plan records valence/arousal values only, no named emotion categories; a later protocol version can add it if D4-2 finds categorical labels | P1 |
| `coraa-ser` | Portuguese (Brazilian) | natural, 933 clips | — | research-only | excluded: its labels are neutral / non-neutral, not named emotions; a binary track would be a new protocol version | P1, P2 |
| `msp-podcast` third-party mirror | English | natural | no | academic-only | excluded from training and testing (below) | none |

Portuguese has no qualifying dataset today. Candidates outside the catalogue (Spanish and
Portuguese) are qualified by brief D4-2; a dataset enters the benchmark only through a new
protocol version that adds its row to this table and its label set to section 4.

**The MSP-Podcast mirror is excluded** from supervised training and from testing, whatever the
answer to P2. It is not the official release: it carries no speaker identity and no official split,
so neither a speaker-disjoint split nor a speaker-level bootstrap can be built; tens of thousands of
its rows carry labels that the mirror introduced rather than the source; and hundreds of
duplicate-content findings were recorded on it. SER's own research specification already bans it
from supervised training and testing. An official MSP-Podcast release would be a different dataset
and a new table row.

The bundle (`bench-v1`, built by workstream WS4) records for every dataset its source URL,
revision or checksum, licence identifier, native sample rate and row count in a dataset card.

## 4. Label policy

### 4.1 Label sets

Every label set has an identifier `<name>-<count>-v<version>`. Labels are lowercase ASCII ids and
are always ordered by Unicode code point.

| Label set id | Labels | Used by |
|---|---|---|
| `ser-8-v1` | angry, calm, disgust, fearful, happy, neutral, sad, surprised | SER's runtime label set (all current profiles) |
| `ravdess-8-v1` | angry, calm, disgust, fearful, happy, neutral, sad, surprised | `ravdess` |
| `crema-d-6-v1` | angry, disgust, fearful, happy, neutral, sad | `crema-d` (source codes ANG, DIS, FEA, HAP, NEU, SAD) |
| `mesd-6-v1` | angry, disgust, fearful, happy, neutral, sad | `mesd`, pending D4-2 confirmation of the source categories |
| `spanish-meacorpus-2023-6-v1` | angry, disgust, fearful, happy, neutral, sad | `spanish-meacorpus-2023`, pending D4-2 confirmation |

### 4.2 Renaming, not mapping

A source category is **renamed** to a label id only when the dataset's own documentation defines
it as that same emotion (for example CREMA-D `FEA` → `fearful`, MESD `anger` → `angry`). Renaming
is one-to-one.

A source category is never **mapped** onto a different category. In particular boredom → neutral,
sleepy → neutral, amused → happy, anxious → fearful, and neutral → calm are forbidden. Some mappings
of this kind exist in SER's dataset catalogue; the benchmark does not use them. A source category
without an exact label id is excluded for that dataset (ledger reason `label_not_scored`), never
coerced.

### 4.3 Scored label set of a model on a dataset

For dataset `d` with label set `S_d` and model `m` with label set `M_m`:

- the scored labels are `K = S_d ∩ M_m`, further reduced to the labels that have at least one row in
  `d`'s test partition;
- rows whose true label is not in `K` are not scored for this pair; their count is reported as
  `rows_label_not_in_model`;
- the model's probability vector is restricted to `K` and renormalised: `p_i(a) = q_i(a) / Σ_{b∈K} q_i(b)`
  for `a ∈ K`, where `q_i` is the vector returned by the call. If that sum is zero, the row is a
  failed row;
- the prediction is `ŷ_i = argmax_{a∈K} p_i(a)`; ties go to the label first in code-point order;
- the mean probability mass the model put outside `K`, `mean_i (1 − Σ_{b∈K} q_i(b))`, is reported as
  `mean_out_of_set_mass`.

When two models with different label sets are compared (section 10), both are rescored on
`K_AB = S_d ∩ M_A ∩ M_B` (restricted to labels with test rows), and the report declares `K_AB`.

## 5. Splits

### 5.1 Duplicate handling (before any split)

Across the whole bundle, rows are grouped by `audio_sha256`. For each group of two or more rows:

- all rows have the same dataset, the same label and the same speaker group: the row with the
  smallest `sample_id` (code-point order) is kept; the others get split `excluded`, reason
  `duplicate_same_label`;
- the rows disagree on label: all of them get `excluded`, reason `duplicate_conflicting_label`;
- the rows agree on label but disagree on dataset or speaker group: all of them get `excluded`,
  reason `duplicate_identity_conflict`.

Every excluded row stays in the split ledger, so the list of exclusions is part of the digest. The
three CREMA-D pairs of identical audio with different labels fall under the second case.

### 5.2 Speaker groups

A row's speaker group is the connected component of its speaker id and, where present, its
session id, as in SER's dataset audit: two rows sharing either belong to the same group. Ids are
namespaced by dataset (`crema-d:1001`). Where a dataset card declares that speakers of two datasets
are the same people, those ids are joined into one component. The group id is the smallest id in
its component in code-point order.

### 5.3 Assignment for train/dev/test datasets

Seed: `17`. For dataset `d` with eligible rows (not excluded by 5.1 or 4.2, with a speaker group):

1. Let the groups be `g_1 … g_n`. If `n < 3`, the dataset cannot have this role.
2. Order the groups by the lowercase hexadecimal SHA-256 of the UTF-8 string
   `"17:" + d + ":" + group_id`, ascending.
3. With `R` = round half to even to an integer:
   - `n_train = max(1, min(n − 2, R(0.6·n)))`
   - `n_dev = max(1, min(n − n_train − 1, R(0.2·n)))`
   - `n_test = n − n_train − n_dev`
4. The first `n_train` groups are train, the next `n_dev` dev, the rest test.

For `ravdess` (`n = 24`) this gives 14/5/5 speakers; for `crema-d` (`n = 91`) 55/18/18.

The seed is never changed to obtain a "better" split. A label with no test row after assignment is
dropped from that dataset's scored set and listed in the report (`labels_without_test_rows`).

A component that spans several datasets takes the split assigned by the first such dataset in
code-point order that has the train/dev/test role; its rows in the other datasets take the same
split, or are excluded with reason `speaker_in_training_partition` if the other dataset is
test-only and the split is train or dev.

A dataset's documented official split is used instead of 5.3 only if its dataset card shows it is
speaker-disjoint; the ledger then records reason `native_speaker_disjoint_split`.

### 5.4 Test-only datasets and datasets without speaker identity

Every eligible row of a test-only dataset gets split `test`, reason `test_only_dataset`.

A dataset without speaker identity is never train or dev. It is test-only only if no row of it, and
no row of the same source collection, is in any train or dev partition, and none of its audio
hashes occurs in train or dev. Its results carry the flag `speaker_unverified` and its bootstrap
resamples utterances (section 8.3). If these conditions cannot be shown, it is excluded.

### 5.5 Split ledger and digest

The split ledger (`ser.bench.split-ledger/v1`) is a JSON Lines file with one object per bundle row,
including excluded rows, with exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `audio_sha256` | string | content hash, section 2.1 |
| `dataset` | string | dataset id |
| `label` | string or null | scored label id after renaming; null if not scored |
| `language` | string | BCP 47 tag (`en`, `es`, `pt`) |
| `reason` | string | `assigned`, `test_only_dataset`, `native_speaker_disjoint_split`, `duplicate_same_label`, `duplicate_conflicting_label`, `duplicate_identity_conflict`, `label_not_scored`, `no_speaker_identity`, `speaker_in_training_partition` |
| `sample_id` | string | unique in the bundle |
| `speaker_group` | string or null | group id of 5.2; null if unknown |
| `split` | string | `train`, `dev`, `test` or `excluded` |

Serialisation: keys sorted, separators `,` and `:` with no spaces, non-ASCII characters written as
UTF-8 (not escaped), one object per line, each line ending in `\n`, lines sorted by `(dataset,
sample_id)` in code-point order. The **ledger digest** is the lowercase hexadecimal SHA-256 of the
whole file's bytes. Rebuilding the bundle from the same sources must give the same digest.

## 6. Leakage audit

The audit reads the bundle manifest, the split ledger and, if given, the model's provenance. It
runs before every benchmark run and on its own. Checks, over all datasets together:

| Check | Pass condition |
|---|---|
| A1 ledger digest | recomputed digest equals the digest recorded in the bundle manifest |
| A2 coverage | every manifest row appears exactly once in the ledger and vice versa |
| A3 shared speakers | 0 speaker groups appear in more than one of train, dev, test |
| A4 shared audio | 0 `audio_sha256` values appear in more than one of train, dev, test |
| A5 duplicate audio | 0 `audio_sha256` values appear twice among non-excluded rows |
| A6 model training data | the model's provenance names the bundle's ledger digest as its training ledger (it was built from train and dev only), or lists its training audio hashes and none is in test; otherwise the result is `unknown` |

A1 to A5 failing stops the run. A6 `unknown` does not stop it; it sets the flag
`training_overlap_unknown` on every result of that model (see P5). A6 finding any overlap stops
the run.

The audit prints exactly these lines, in this order, then the offending ids (one per line, sorted)
if any check failed, and exits with status 0 on pass, 2 on failure:

```text
ser-bench leakage-audit protocol=1 bundle=<bundle id>
ledger_sha256=<recorded> recomputed=<recomputed> match=<true|false>
rows=<n> train=<n> dev=<n> test=<n> excluded=<n>
shared_speakers train/dev=<n> train/test=<n> dev/test=<n>
shared_audio train/dev=<n> train/test=<n> dev/test=<n>
duplicate_audio_scored=<n>
model_training_overlap=<verified|unknown|found:<n>|not_checked>
result=<PASS|FAIL>
```

## 7. Scoring a row

### 7.1 Notation

For one model on one dataset's test partition: rows `i = 1 … N` (rows with a true label in `K`),
`k = |K|`, true label `y_i`, prediction `ŷ_i` (section 4.3) or `∅` for a failed row, confidence
`c_i = max_{a∈K} p_i(a)` rounded half to even to 6 decimal places. `N'` is the number of rows that
are not failed.

### 7.2 Failed rows

A failed row counts in every recall denominator and is never correct. It appears in the confusion
matrix in a final column `no_prediction`. It has no confidence and is left out of the calibration
error. The report states `n_failed`.

## 8. Metrics

### 8.1 Per dataset

All metrics are computed in exact rational arithmetic from integer counts and the 6-decimal
confidences, then rounded once, half to even, to 6 decimal places.

- **Confusion matrix** `C[a][b]` = number of rows with `y_i = a` and `ŷ_i = b`, for `a ∈ K` (rows)
  and `b ∈ K ∪ {no_prediction}` (columns), both in code-point order with `no_prediction` last.
- **Support** `n_a = Σ_b C[a][b]`.
- **Per-class recall** `R_a = C[a][a] / n_a`.
- **UAR** (unweighted average recall, the primary metric) `= (1/k) Σ_{a∈K} R_a`.
- **Per-class precision** `P_a = C[a][a] / Σ_{x∈K} C[x][a]`; `P_a = 0` if that column sum is 0.
- **Per-class F1** `F_a = 2·P_a·R_a / (P_a + R_a)`; `F_a = 0` if `P_a + R_a = 0`.
- **Macro-F1** `= (1/k) Σ_{a∈K} F_a`.
- **Accuracy** `= Σ_a C[a][a] / N` (reported for continuity; never used in a decision).
- **Expected calibration error** (top-label, 10 equal-width bins):
  - bin of row `i`: `m_i = min(10, ⌊10·c_i⌋ + 1)`, on the exact 6-decimal value, so bin `m` holds
    `[(m−1)/10, m/10)` and bin 10 holds `[0.9, 1.0]`;
  - for each non-empty bin: `acc_m` = fraction of its rows with `ŷ_i = y_i`, `conf_m` = mean of
    its `c_i`;
  - `ECE = Σ_m (|B_m| / N') · |acc_m − conf_m|`;
  - empty bins contribute 0 and are reported with count 0 and null accuracy and confidence;
  - if `N' = 0`, ECE is null and the result is flagged `no_predictions`.

Labels in `S_d` but absent from the test partition are not part of `K` and do not enter UAR as
zero. (SER's `compute_ser_metrics` in `ser/_internal/train/metrics.py` gives such a label a recall
of 0.0 when it is passed explicitly in `labels`; the benchmark does not pass it.) On rows without
failures, these definitions equal scikit-learn's macro recall and macro F1 with
`zero_division=0` over `K`.

### 8.2 What is reported per dataset

UAR, macro-F1, accuracy, per-class recall, precision and F1, the confusion matrix, ECE with its
per-bin table, `n_rows`, `n_speakers`, `n_failed`, `rows_label_not_in_model`,
`mean_out_of_set_mass`, and 95% confidence intervals for UAR, macro-F1 and ECE.

### 8.3 Bootstrap confidence intervals

- **Unit: the speaker.** Rows of one speaker are correlated (same voice, same session, same
  prompts); the claim is about new speakers, so speakers are resampled. Resampling utterances
  would understate the uncertainty. A `speaker_unverified` dataset resamples utterances instead,
  and its intervals carry that flag.
- **Resamples:** `B = 10,000`.
- **Random draws:** for each dataset, a generator `numpy.random.Generator(numpy.random.PCG64(s_d))`
  with `s_d` = the first 8 bytes, read as a big-endian unsigned integer, of SHA-256 of the UTF-8
  string `"17:bootstrap:" + d`. The test speakers are ordered by group id in code-point order, `S`
  of them. For `b = 0 … B−1`: `idx_b = rng.integers(0, S, size=S)`; resample `b` is the multiset of
  all rows of speakers `idx_b`, with repetition. The draws depend only on the dataset, never on the
  model, so every model and baseline sees the same resamples (paired).
- **Metric on a resample:** the formulas of 8.1, with `K` reduced to labels that have at least one
  row in that resample.
- **Interval:** sort the `B` values `v_0 ≤ … ≤ v_{B−1}`; for `q ∈ {0.025, 0.975}`, `h = q·(B−1)`,
  value `= v_⌊h⌋ + (h − ⌊h⌋)·(v_⌈h⌉ − v_⌊h⌋)` (the "linear" percentile). With `B = 10,000`,
  `h = 249.975` and `9748.025`.
- A dataset with fewer than 5 test speakers is flagged `underpowered`; its intervals are reported
  but are not reliable.

### 8.4 Per language, never pooled

For language `ℓ` with scored datasets `D_ℓ` (each weighted equally, whatever its size):

- `UAR_ℓ = mean_{d∈D_ℓ} UAR_d`, likewise `MacroF1_ℓ`, `ECE_ℓ`;
- `chance_ℓ = mean_{d∈D_ℓ} 1/k_d`;
- its interval uses, for each `b`, the mean over `d ∈ D_ℓ` of the resample-`b` values (datasets are
  resampled independently, stratified by dataset).

Rows of different datasets are never merged into one confusion matrix or one metric. No figure
across languages is computed.

## 9. Baselines

Every report includes these rows for every scored dataset, on the same test partition and scored
label set `K` as the model:

| Baseline | Definition | Values |
|---|---|---|
| `chance` | analytic expectation of uniform random guessing over `K` | `UAR = 1/k`; `MacroF1 = (1/k) Σ_a 2·π_a·(1/k) / (π_a + 1/k)` with `π_a = n_a/N`; accuracy `= 1/k`; ECE null |
| `majority` | always predicts the most frequent label of the dataset's train partition (of the test partition for a test-only dataset, flagged `majority_from_test`); ties to the first label in code-point order; probability 1 on that label | `UAR = 1/k`; `MacroF1 = (1/k)·2·π_m/(π_m + 1)`; accuracy `= π_m`; `ECE = 1 − π_m` |
| `fast@16k` | the existing `fast` model artifact (identified by its SHA-256), through the public call with 16 kHz input as in section 2 | measured |
| `fast@native` | the same artifact with the file's own sample rate | measured |

The `fast` artifact was trained on all 24 RAVDESS actors with a clip-level split, so every RAVDESS
test speaker was in its training data: its RAVDESS rows carry the flag `trained_on_test_speakers`.
Where a dataset's native rate is 16 kHz, the two `fast` rows are equal; both are reported.

If the public call at the scored code version always resamples to 16 kHz, `fast@native` is produced
at the last recorded commit whose inference path used the file's own rate, and the row carries the
flag `legacy_native_rate_path` and that commit.

## 10. Decision rule

### 10.1 Primary metric

Decisions use UAR only. Macro-F1, ECE and per-class figures are reported but never decide which
model is better (ECE enters only acceptance, 10.4).

### 10.2 "Model A beats model B"

Both models are scored on the same bundle, ledger digest and test rows, on `K_AB` (section 4.3),
with the same resamples (8.3). For each resample `b`: `Δ_b = UAR_A,b − UAR_B,b` per dataset, and
`Δ_ℓ,b` = mean of the dataset `Δ_b` over `D_ℓ` per language. With the interval of 8.3 on `Δ`:

- A beats B **in language ℓ** if the lower end of the `Δ_ℓ` interval is greater than 0;
- A beats B **overall** if A beats B in at least one scored language and B beats A in none;
- otherwise neither beats the other.

No correction for multiple comparisons is applied; the report lists every interval.

### 10.3 Ties and selection

When neither beats the other, the cheaper model is preferred, compared in this order, on numbers
measured on the same machine class by the CPU performance harness (workstream WS7): lower 95th
percentile processing time per second of audio at 2 threads; then lower peak memory; then smaller
model file. If all three are equal, the current default stays.

To choose a default among several candidates: drop every candidate that fails 10.4; among the
rest, choose the one that beats the most others; break ties by the cost order above.

### 10.4 Acceptance thresholds (checklist U11)

A model is acceptable as default only if, for every scored language `ℓ` (which languages: P1):

1. `UAR_ℓ ≥ 2 · chance_ℓ`;
2. `UAR_ℓ ≥ UAR_ℓ(fast reference) + 0.10`, both computed on `K_AB` of the model and `fast`, where
   the reference row is chosen by P3;
3. `ECE_ℓ ≤ 0.10`;
4. a check on at least 100 of the owner's own labelled turns in `ℓ` is recorded from the E3-4 kit
   (the report holds its status, not its content).

Whether conditions 1 to 3 apply to the point estimates or to the interval bounds is P4.

## 11. Report format

### 11.1 Files

A run writes two files with the same stem, chosen by the operator (for example
`benchmarks/reports/2026-12-baselines`): `<stem>.json` and `<stem>.md`, and a predictions file
`<stem>.predictions.jsonl` with one line per scored row (`dataset`, `sample_id`, `label`, `status`,
`probabilities` over the model's own label set at 6 decimals), sorted by `(dataset, sample_id)`.
Predictions of the owner's own recordings are never written by this harness.

### 11.2 JSON (`ser.bench.report/v1`)

| Field | Type | Content |
|---|---|---|
| `schema` | string | `"ser.bench.report/v1"` |
| `protocol` | object | `id` (`"ser-bench-protocol"`), `version` (`1`), `document_sha256` (SHA-256 of this file at the commit used) |
| `code` | object | `ser_version`, `git_commit` (40 hex), `git_dirty` (must be `false`) |
| `bundle` | object | `id`, `manifest_sha256`, `split_ledger_sha256`, `content_hash_contract` (`"pcm16k-s16le-v1"`), `resampler` (name and version) |
| `model` | object | `id`, `profile`, `artifact_sha256`, `label_set_id`, `labels`, `licence_class`, `training_ledger_sha256` (string or null), `training_overlap` (`"verified"`, `"unknown"`) |
| `input` | object | `sample_rate` (`16000` or `"native"`), `channels` (`1`), `call` (the public call's qualified name), `timeout_seconds` |
| `settings` | object | `split_seed` (`17`), `bootstrap_seed_rule` (`"17:bootstrap:<dataset>"`), `bootstrap_resamples` (`10000`), `ece_bins` (`10`) |
| `predictions_sha256` | string | SHA-256 of the predictions file |
| `audit` | object | the eight audit lines of section 6 as key/value pairs |
| `datasets` | array | one object per scored dataset: `dataset`, `language`, `label_set_id`, `scored_labels`, `labels_without_test_rows`, `n_rows`, `n_speakers`, `n_failed`, `rows_label_not_in_model`, `mean_out_of_set_mass`, `metrics`, `ci`, `flags` |
| `metrics` | object | `uar`, `macro_f1`, `accuracy`, `per_class` (object keyed by label: `support`, `recall`, `precision`, `f1`), `confusion` (`rows`, `columns`, `matrix` as array of integer arrays), `ece`, `ece_bins` (10 objects: `bin`, `lower`, `upper`, `count`, `accuracy`, `confidence`) |
| `ci` | object | `uar`, `macro_f1`, `ece`, each `[lower, upper]`; `unit` (`"speaker"` or `"utterance"`) |
| `languages` | array | per language: `language`, `datasets`, `uar`, `macro_f1`, `ece`, `chance_uar`, `ci` |
| `baselines` | array | per baseline of section 9: `id`, then the same `datasets` and `languages` structures |
| `acceptance` | object | per language: each condition of 10.4 with `value`, `threshold`, `pass`; `own_recordings` (`"recorded"` or `"missing"`); `pending` (list of the P-decisions still open, which leave the verdict `"undecided"`) |
| `comparisons` | array | optional; per pair: `a`, `b`, `labels` (`K_AB` per dataset), `delta_uar` and its interval per dataset and per language, `verdict` |

### 11.3 Byte-identical output

Two runs on the same bundle, model, code version and predictions produce identical bytes:

- no wall-clock time, host name, user name, absolute path or duration appears in any file;
- JSON is UTF-8, keys sorted at every level, two-space indentation, separators `", "` and `": "`,
  non-ASCII not escaped, a final newline;
- arrays are ordered: datasets and languages by id in code-point order, labels in code-point order,
  baselines in the order of section 9, ECE bins by bin number;
- every metric is computed exactly (section 8.1) and written as a JSON number with exactly six
  digits after the decimal point (`0.750000`); counts are integers; null is `null`;
- the bootstrap uses only the seeded generators of 8.3;
- the Markdown file is generated from the JSON file alone by a fixed template; it shows metrics to
  3 decimals, rounded half to even from the 6-decimal JSON value.

Model outputs can differ across processors and numerical libraries. The report is a pure function
of the predictions file; `predictions_sha256` shows whether two runs received the same outputs.

### 11.4 Markdown rendering

Sections, in order: identity (protocol, code, bundle, model, input); audit lines; per-language
table (model and baselines, UAR with interval, macro-F1, ECE); per-dataset tables (same columns,
plus per-class recall); one confusion matrix per dataset; ECE bin table per dataset; acceptance
table; comparisons; flags. Section 12.8 shows the rendering of the toy example.

## 12. Worked toy example

A toy dataset `toy` (language `und`, role test-only) of 20 rows: 4 speakers (S1 to S4), 3 labels,
label set `toy-3-v1` = angry, happy, neutral. The model's label set is the same, so `K` = {angry,
happy, neutral}, `k = 3`, and no renormalisation changes any vector. `audio_sha256` values are
stand-ins (`a01` … `a20`), not real hashes. Row `toy-s1-04` failed (status `too_short`).

### 12.1 The rows

| # | sample_id | speaker | true | p(angry) | p(happy) | p(neutral) | predicted | c | bin | correct |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | toy-s1-01 | S1 | angry | 0.95 | 0.03 | 0.02 | angry | 0.95 | 10 | yes |
| 2 | toy-s1-02 | S1 | angry | 0.30 | 0.60 | 0.10 | happy | 0.60 | 7 | no |
| 3 | toy-s1-03 | S1 | happy | 0.03 | 0.95 | 0.02 | happy | 0.95 | 10 | yes |
| 4 | toy-s1-04 | S1 | happy | — | — | — | no_prediction | — | — | no |
| 5 | toy-s1-05 | S1 | neutral | 0.10 | 0.10 | 0.80 | neutral | 0.80 | 9 | yes |
| 6 | toy-s2-01 | S2 | angry | 0.90 | 0.05 | 0.05 | angry | 0.90 | 10 | yes |
| 7 | toy-s2-02 | S2 | angry | 0.30 | 0.20 | 0.50 | neutral | 0.50 | 6 | no |
| 8 | toy-s2-03 | S2 | happy | 0.10 | 0.80 | 0.10 | happy | 0.80 | 9 | yes |
| 9 | toy-s2-04 | S2 | neutral | 0.05 | 0.05 | 0.90 | neutral | 0.90 | 10 | yes |
| 10 | toy-s2-05 | S2 | neutral | 0.80 | 0.10 | 0.10 | angry | 0.80 | 9 | no |
| 11 | toy-s3-01 | S3 | angry | 0.90 | 0.05 | 0.05 | angry | 0.90 | 10 | yes |
| 12 | toy-s3-02 | S3 | angry | 0.80 | 0.10 | 0.10 | angry | 0.80 | 9 | yes |
| 13 | toy-s3-03 | S3 | happy | 0.05 | 0.90 | 0.05 | happy | 0.90 | 10 | yes |
| 14 | toy-s3-04 | S3 | happy | 0.60 | 0.30 | 0.10 | angry | 0.60 | 7 | no |
| 15 | toy-s3-05 | S3 | neutral | 0.20 | 0.20 | 0.60 | neutral | 0.60 | 7 | yes |
| 16 | toy-s4-01 | S4 | angry | 0.80 | 0.10 | 0.10 | angry | 0.80 | 9 | yes |
| 17 | toy-s4-02 | S4 | angry | 0.60 | 0.20 | 0.20 | angry | 0.60 | 7 | yes |
| 18 | toy-s4-03 | S4 | happy | 0.20 | 0.60 | 0.20 | happy | 0.60 | 7 | yes |
| 19 | toy-s4-04 | S4 | neutral | 0.10 | 0.40 | 0.50 | neutral | 0.50 | 6 | yes |
| 20 | toy-s4-05 | S4 | neutral | 0.30 | 0.50 | 0.20 | happy | 0.50 | 6 | no |

Supports: angry 8 (rows 1, 2, 6, 7, 11, 12, 16, 17), happy 6 (3, 4, 8, 13, 14, 18), neutral 6
(5, 9, 10, 15, 19, 20). `N = 20`, `n_failed = 1`, `N' = 19`, `n_speakers = 4` (flag
`underpowered`, since 4 < 5).

### 12.2 Confusion matrix, recall, precision, F1

| true \ predicted | angry | happy | neutral | no_prediction | support |
|---|---|---|---|---|---|
| angry | 6 | 1 | 1 | 0 | 8 |
| happy | 1 | 4 | 0 | 1 | 6 |
| neutral | 1 | 1 | 4 | 0 | 6 |
| column sum (K only) | 8 | 6 | 5 | | |

- angry row: correct rows 1, 6, 11, 12, 16, 17 = 6; row 2 → happy; row 7 → neutral.
- happy row: correct rows 3, 8, 13, 18 = 4; row 14 → angry; row 4 failed.
- neutral row: correct rows 5, 9, 15, 19 = 4; row 10 → angry; row 20 → happy.

| label | recall | precision | F1 |
|---|---|---|---|
| angry | 6/8 = 0.750000 | 6/8 = 0.750000 | 2·(3/4)(3/4)/(3/4+3/4) = 3/4 = 0.750000 |
| happy | 4/6 = 0.666667 | 4/6 = 0.666667 | 2·(2/3)(2/3)/(4/3) = 2/3 = 0.666667 |
| neutral | 4/6 = 0.666667 | 4/5 = 0.800000 | 2·(4/5)(2/3)/(4/5+2/3) = (16/15)/(22/15) = 8/11 = 0.727273 |

- **UAR** = (3/4 + 2/3 + 2/3)/3 = (9/12 + 16/12)/3 = (25/12)/3 = 25/36 = **0.694444**.
- **Macro-F1** = (3/4 + 2/3 + 8/11)/3 = (99/132 + 88/132 + 96/132)/3 = (283/132)/3 = 283/396 =
  **0.714646**.
- **Accuracy** = (6 + 4 + 4)/20 = 14/20 = **0.700000**.

### 12.3 Calibration error

| bin | range | rows | count | correct | acc_m | conf_m | \|acc_m − conf_m\| | weight count/19 |
|---|---|---|---|---|---|---|---|---|
| 1–5 | [0.0, 0.5) | — | 0 | — | null | null | — | 0 |
| 6 | [0.5, 0.6) | 7, 19, 20 | 3 | 1 | 1/3 | (0.5+0.5+0.5)/3 = 0.5 | 1/6 | 3/19 |
| 7 | [0.6, 0.7) | 2, 14, 15, 17, 18 | 5 | 3 | 3/5 = 0.6 | 0.6 | 0 | 5/19 |
| 8 | [0.7, 0.8) | — | 0 | — | null | null | — | 0 |
| 9 | [0.8, 0.9) | 5, 8, 10, 12, 16 | 5 | 4 | 4/5 = 0.8 | 0.8 | 0 | 5/19 |
| 10 | [0.9, 1.0] | 1, 3, 6, 9, 11, 13 | 6 | 6 | 1 | (0.95+0.95+0.9·4)/6 = 5.5/6 = 11/12 | 1/12 | 6/19 |

Bin of each row: `⌊10·0.5⌋ + 1 = 6`, `⌊10·0.6⌋ + 1 = 7`, `⌊10·0.8⌋ + 1 = 9`,
`min(10, ⌊10·0.9⌋ + 1) = 10`, `min(10, ⌊10·0.95⌋ + 1) = 10`. Row 4 has no confidence and is left
out, so the weights are over `N' = 19`.

**ECE** = (3/19)(1/6) + (5/19)(0) + (5/19)(0) + (6/19)(1/12) = 0.5/19 + 0.5/19 = 1/19 =
**0.052632**.

### 12.4 Baselines on the toy

`π_angry = 8/20 = 0.4`, `π_happy = π_neutral = 6/20 = 0.3`. The toy is test-only, so `majority`
uses the test partition (flag `majority_from_test`): angry.

- `chance`: UAR = 1/3 = **0.333333**. Macro-F1 terms: angry `2·0.4·(1/3)/(0.4 + 1/3)` =
  `(4/15)/(11/15)` = 4/11; happy and neutral `2·0.3·(1/3)/(0.3 + 1/3)` = `(1/5)/(19/30)` = 6/19.
  Macro-F1 = (4/11 + 6/19 + 6/19)/3 = ((76 + 132)/209)/3 = 208/627 = **0.331738**.
- `majority`: recalls 1, 0, 0, so UAR = 1/3 = **0.333333**; precision of angry 8/20 = 0.4, F1 of
  angry `2·0.4·1/(0.4 + 1)` = 4/7, others 0; Macro-F1 = (4/7)/3 = 4/21 = **0.190476**;
  accuracy = **0.400000**; every confidence is 1.0, all 20 rows in bin 10 with accuracy 0.4, so
  ECE = |0.4 − 1.0| = **0.600000**.

### 12.5 One bootstrap resample, by hand

The real draws come from the seeded generator of 8.3 and cannot be done by hand, so the toy report
shows no interval. To show what one resample computes, take the draw (S3, S1, S3, S4). Per speaker,
correct/total by true label (from 12.1):

| speaker | angry | happy | neutral |
|---|---|---|---|
| S1 | 1/2 | 1/2 | 1/1 |
| S2 | 1/2 | 1/1 | 1/2 |
| S3 | 2/2 | 1/2 | 1/1 |
| S4 | 2/2 | 1/1 | 1/2 |

Resample recalls: angry (2+1+2+2)/(2+2+2+2) = 7/8; happy (1+1+1+1)/(2+2+2+1) = 4/7; neutral
(1+1+1+1)/(1+1+1+2) = 4/5. UAR_b = (7/8 + 4/7 + 4/5)/3 = ((245 + 160 + 224)/280)/3 = 629/840 =
0.748810. The interval of 8.3 is the 2.5th and 97.5th percentiles of 10,000 such values.

### 12.6 Split digest inputs

**As a test-only dataset** (the role used above), every row gets `split = "test"`,
`reason = "test_only_dataset"`. The ledger's first two lines are:

```text
{"audio_sha256":"a01","dataset":"toy","label":"angry","language":"und","reason":"test_only_dataset","sample_id":"toy-s1-01","speaker_group":"S1","split":"test"}
{"audio_sha256":"a02","dataset":"toy","label":"angry","language":"und","reason":"test_only_dataset","sample_id":"toy-s1-02","speaker_group":"S1","split":"test"}
```

The other 18 lines follow the same pattern in row order of 12.1 (which is already `sample_id`
order), with `audio_sha256` = `a03` … `a20` and the speaker and label of the row. The file is
3,232 bytes (14 `angry` or `happy` lines of 161 bytes and 6 `neutral` lines of 163 bytes, each
including its `\n`), and its SHA-256 is:

```text
ec120a9d63135996aaf48d7004f3fe11de24b64f7548441d181869fec9d7b66b
```

**If the same toy dataset had the train/dev/test role**, section 5.3 would apply with `n = 4`
groups. The sort keys are SHA-256 of `17:toy:S1` … `17:toy:S4`:

| group | SHA-256 of `"17:toy:<group>"` | order |
|---|---|---|
| S4 | `94224a5175afef0035cd0b975d7f578cfbfc489da720a8cfcb282fdcc2e259d0` | 1 |
| S1 | `96dbb426009a568fc27e7d37753f631da7f82ac04c5e18fa2c1121c6084d81bc` | 2 |
| S3 | `d66010bcb09984b3f15b485da5f5ed0450e36d8223bebf5d3569b08efdcf9390` | 3 |
| S2 | `e55ebc1b6bc1648a8d228c442d5939cccaef95eb08465f20f4b8f5d38387fc15` | 4 |

`n_train = max(1, min(2, R(2.4))) = 2`, `n_dev = max(1, min(4 − 2 − 1, R(0.8))) = 1`,
`n_test = 1`: train = S4, S1; dev = S3; test = S2.

Any SHA-256 tool reproduces these, for example `printf '%s' '17:toy:S1' | shasum -a 256`.

### 12.7 Sample report (JSON excerpt)

```json
{
  "datasets": [
    {
      "ci": {"ece": null, "macro_f1": null, "uar": null, "unit": "speaker"},
      "dataset": "toy",
      "flags": ["majority_from_test", "underpowered"],
      "label_set_id": "toy-3-v1",
      "labels_without_test_rows": [],
      "language": "und",
      "mean_out_of_set_mass": 0.000000,
      "metrics": {
        "accuracy": 0.700000,
        "confusion": {
          "columns": ["angry", "happy", "neutral", "no_prediction"],
          "matrix": [[6, 1, 1, 0], [1, 4, 0, 1], [1, 1, 4, 0]],
          "rows": ["angry", "happy", "neutral"]
        },
        "ece": 0.052632,
        "ece_bins": [
          {"accuracy": null, "bin": 1, "confidence": null, "count": 0, "lower": 0.000000, "upper": 0.100000},
          "... bins 2 to 5 likewise empty ...",
          {"accuracy": 0.333333, "bin": 6, "confidence": 0.500000, "count": 3, "lower": 0.500000, "upper": 0.600000},
          {"accuracy": 0.600000, "bin": 7, "confidence": 0.600000, "count": 5, "lower": 0.600000, "upper": 0.700000},
          {"accuracy": null, "bin": 8, "confidence": null, "count": 0, "lower": 0.700000, "upper": 0.800000},
          {"accuracy": 0.800000, "bin": 9, "confidence": 0.800000, "count": 5, "lower": 0.800000, "upper": 0.900000},
          {"accuracy": 1.000000, "bin": 10, "confidence": 0.916667, "count": 6, "lower": 0.900000, "upper": 1.000000}
        ],
        "macro_f1": 0.714646,
        "per_class": {
          "angry": {"f1": 0.750000, "precision": 0.750000, "recall": 0.750000, "support": 8},
          "happy": {"f1": 0.666667, "precision": 0.666667, "recall": 0.666667, "support": 6},
          "neutral": {"f1": 0.727273, "precision": 0.800000, "recall": 0.666667, "support": 6}
        },
        "uar": 0.694444
      },
      "n_failed": 1,
      "n_rows": 20,
      "n_speakers": 4,
      "rows_label_not_in_model": 0,
      "scored_labels": ["angry", "happy", "neutral"]
    }
  ]
}
```

The `ci` values are null only in this toy, because its draws are not reproducible by hand; a real
report always fills them. The `"..."` line marks an omission in this excerpt; a real report lists
all ten bins.

### 12.8 Sample report (Markdown excerpt)

```markdown
## Dataset `toy` (und) — flags: majority_from_test, underpowered

| row | UAR | macro-F1 | accuracy | ECE | failed |
|---|---|---|---|---|---|
| model | 0.694 | 0.715 | 0.700 | 0.053 | 1 |
| chance | 0.333 | 0.332 | 0.333 | — | — |
| majority | 0.333 | 0.190 | 0.400 | 0.600 | 0 |

| label | support | recall | precision | F1 |
|---|---|---|---|---|
| angry | 8 | 0.750 | 0.750 | 0.750 |
| happy | 6 | 0.667 | 0.667 | 0.667 |
| neutral | 6 | 0.667 | 0.800 | 0.727 |
```

Rounding to 3 decimals: 0.714646 → 0.715, 0.052632 → 0.053, 0.331738 → 0.332, 0.190476 → 0.190.

## 13. Limitations of version 1

- Exact-content duplicates only; near-duplicates (re-encoded, trimmed, re-amplified copies) and
  shared prompt texts across partitions are not detected.
- Acted corpora dominate; no conversational test set exists yet.
- No correction for multiple comparisons (10.2).
- Equal dataset weights per language (8.4) give a small dataset as much weight as a large one.

## 14. Relation to the existing quality gate

The profile quality gate (`ser/_internal/runtime/profile_quality_gate.py` and
`quality_gate_*.py`) is not reused. It loads one already trained artifact per profile and then, for
every grouped fold, scores that same artifact on the fold's test indices; the fold's train indices
are never used (`quality_gate_evaluation.py`, `for _, test_indices in folds`). Whether the
artifact was trained on a test speaker is never checked, so its folds are repeated partitioned
inference, not cross-validation. It also derives speakers from RAVDESS file names only, discovers
samples by a file glob, reports only pooled metrics, has no calibration or interval, and stamps
its report with the wall-clock time, so two runs are not byte-identical.

## 15. Pending owner decisions

Each of these is open. Until it is decided, the affected rows of section 3 are not built and the
acceptance verdict of section 10.4 is `"undecided"`.

| Id | Question | Options | Affects |
|---|---|---|---|
| P1 | Which languages are scored (decision D2) | (a) English only; (b) English and Spanish; (c) English, Spanish and Portuguese (Portuguese has no qualifying dataset today, so (c) means own recordings only until one qualifies) | which languages 10.4 applies to; whether `mesd`, `spanish-meacorpus-2023` enter the bundle |
| P2 | Which licence classes are acceptable (decision D3) | (a) open only (CC BY, ODbL, CC0); (b) also non-commercial and research-only data for a privately used model; (c) also the academic-only MSP-Podcast mirror. A sub-question under (a): whether non-open datasets may still be used as test-only for private evaluation | role of `ravdess`, `spanish-meacorpus-2023`, `coraa-ser`. Under (c) the mirror stays excluded here for the technical reasons of section 3 |
| P3 | Which `fast` row is the reference for the "+10 points" clause of U11 | (a) `fast@16k`, the input consumers send; (b) `fast@native`; (c) the higher of the two per language | 10.4 condition 2 |
| P4 | Whether U11 thresholds apply to point estimates or to interval bounds | (a) point estimates; (b) lower interval bound for UAR conditions and upper bound for ECE | 10.4 conditions 1–3 |
| P5 | How results of a model whose training data cannot be checked against the test partitions (`training_overlap_unknown`, for example a pretrained model such as emotion2vec+) are treated | (a) counted like any other, flagged; (b) counted only on datasets it provably never saw, and on the owner's own recordings; (c) reported but never used by 10.2–10.4 | 10.2, 10.3, 10.4 for such models |
