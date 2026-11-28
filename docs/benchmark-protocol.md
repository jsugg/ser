# SER benchmark protocol

Protocol identifier: `ser-bench-protocol`, version `1`.

This document fixes how any SER model is scored. It was written, and revised after review, before
any model was scored under it. A change to any rule below creates a new protocol version; reports
produced under different versions are never compared with each other.

The key words "must", "must not", "is" and "are" state rules. The owner's decisions that this
protocol depends on are recorded in [section 15](#15-pending-owner-decisions).

## 1. Purpose and claims

The benchmark measures how well a model assigns categorical emotion labels to single utterances,
per dataset and per benchmark language (English `en` and Spanish `es`), and how well its
confidence matches its accuracy. It chooses SER's first default model (section 10.5); no existing
default is presumed. SER's eight labels (`ser-8-v1`, section 4.1) are the shipped output contract.

A report produced under this protocol supports these claims and no others:

- the model's UAR, macro-F1, per-class recall, confusion matrix and calibration error on the test
  partition of each named dataset, with a confidence interval;
- for speaker-verified datasets (section 5.4) only: that these figures hold for speakers whose
  speech was in no train or dev partition of the bundle and, for `verified` provenance, in no data
  the model was fitted, tuned or calibrated on (section 6.1);
- the same figures aggregated per benchmark language, as defined in section 8.4;
- that model A beats model B, or does not, under the decision rule of section 10;
- that a model passes or fails the acceptance thresholds of section 10.4.

A report does not support any claim about:

- speech, speakers, recording conditions or languages outside the scored test partitions. Acted
  corpora say little about conversational or clinical speech;
- Portuguese. Portuguese is checked only on the owner's own recordings, by the own-recordings kit
  of brief E3-4; the report records only that check's status;
- clinical, diagnostic or psychological validity of the labels;
- agreement with the owner's own reading of his recordings (measured by the E3-4 kit, not here);
- fairness across demographic groups (the protocol does not stratify by sex, age or accent);
- a single pooled number across datasets or languages. Pooled figures are never reported.

Results flagged `speaker_unverified`, `identity_unresolved`, `overlap_unverified`, `contaminated`
or `provenance_unknown` are descriptive only: they are reported and never enter sections 10.2 to
10.5.

The only result on record before this protocol (RAVDESS accuracy 58.6%, UAR 56.8%, `fast`
profile) came from a split by clip, not by speaker, and is not comparable with any figure produced
here.

## 2. Input form

Every model is scored through the public utterance-analysis call of SER (the call defined by
workstream WS1), so the benchmark measures what a consumer calls. The harness does not call
internal model classes and does not aggregate windows itself.

### 2.1 Preprocessing contract `ser-preproc-v1`

The bundle manifest pins, under `preprocessing`, the decoder (name and version), the resampler
(name, version and every option it accepts, written out even where it equals the library default)
and the call options and timeout of section 2.2. A change to any of them is a new bundle id. For
every row the bundle builder and the harness:

1. identify the row by its `sample_id`, assigned once when the bundle is built from the dataset id
   and the source file's path relative to the dataset root (plus the segment bounds, if any), and
   never reused or renumbered within a bundle id;
2. decode the source file with the pinned decoder to float64 samples; integer PCM of `b` bits is
   divided by `2^(b−1)`, floating-point PCM is taken as is;
3. if the row declares a segment `[start, end)` in seconds (written as decimal strings), keep the
   frames `i` with `start ≤ i / r < end`, that is frames `⌈start·r⌉` up to but excluding
   `⌈end·r⌉`, computed exactly, where `r` is the file's native rate;
4. mix all channels to one by the arithmetic mean of the channels, sample by sample, in float64;
5. resample to 16,000 Hz with the pinned resampler in float64; a file already at 16,000 Hz is not
   resampled.

A row whose decode fails, yields zero frames after step 3, or contains a non-finite sample is
excluded when the bundle is built (ledger reason `decode_invalid`), so the scored rows are the same
for every model.

### 2.2 The call

For every scored row the harness passes the result of 2.1, cast to float32, and the sample rate
16,000 to the public call, with the call options pinned in the manifest and one loaded analyser
reused for all rows of a run. It records the returned status and the probability vector over the
model's label set (section 11.1). Every model receives identical input. The one exception is the
`fast@native` baseline row of section 9, which feeds the output of steps 2 to 4 at the file's own
rate; its report row records `input.sample_rate = "native"`.

A row whose call returns any status other than success, raises, exceeds the pinned timeout
(`timeout_seconds`, 120 in `bench-v1`), or returns an invalid vector (section 4.3) is a **failed
row**. Failed rows stay in the scored set (section 7.2).

### 2.3 Audio content identity

Duplicate detection and the leakage audit compare audio by the content hash contract
`pcm16k-s16le-v1`:

1. steps 1 to 5 of section 2.1;
2. clip each sample to [-1.0, 1.0], multiply by 32767, round half to even to an integer;
3. serialise as signed 16-bit little-endian integers, no header;
4. SHA-256 of those bytes, lowercase hexadecimal: the row's `audio_sha256`.

A change of preprocessing contract or of any step above is a new contract identifier and requires
recomputing every hash in the bundle. This contract detects identical audio only; re-encoded or
trimmed copies of the same recording are not detected (section 13).

## 3. Datasets and roles

Roles:

- **train/dev/test**: split by speaker as in section 5; train and dev are available to model
  builders, test is scored.
- **test-only**: every eligible row is in test; no row of the dataset is used to build or tune a
  model.
- **training-only**: rows may be used to build models; no row is ever in dev or test or scored.
- **excluded**: not in the bundle; not scored.

| Dataset id | Language | Kind | Speaker identity | Licence id (catalogue policy) | Role in `bench-v1` |
|---|---|---|---|---|---|
| `ravdess` | `en` | acted, 24 actors | yes (actor in filename) | `cc-by-nc-sa-4.0` (`noncommercial`) | train/dev/test |
| `crema-d` | `en` | acted, 91 actors | yes (actor in filename) | `odbl-1.0` (`share_alike`) | train/dev/test |
| `mesd` | `es` (Mexican) | acted | to be verified by brief D4-2 | `cc-by-4.0` (`open`) | by the admission rule below |
| `spanish-meacorpus-2023` | `es` | natural | to be verified by D4-2 | `cc-by-nc-4.0` (`noncommercial`) | by the admission rule below |
| `escorpus-pe` | `es` (Peruvian) | — | — | `cc-by-4.0` (`open`) | excluded: valence/arousal values only, no named emotion categories |
| `coraa-ser` | `pt` (Brazilian) | natural, 933 clips | — | `custom-research-only` (`research_only`) | excluded: neutral / non-neutral labels only, and Portuguese is not a benchmark language |
| `msp-podcast` third-party mirror | `en` | natural | no | `msp-academic-license` (`academic_only`) | training-only (below) |

**Admission of `mesd` and `spanish-meacorpus-2023`.** These datasets are in the catalogue but not
yet qualified. Each enters a bundle only through this single route: brief D4-2 records, for the
pinned source revision, (a) the raw category inventory and (b) whether each row carries an
individual speaker id. If (a) differs from the dataset's map in section 4.1, the dataset is
excluded until a new protocol version changes the map. Otherwise its role is: train/dev/test if
(b) holds and it has at least 3 speaker groups; test-only and `speaker_unverified` (section 5.4)
if (b) does not hold. The bundle manifest records the D4-2 record's SHA-256. A dataset not listed
in this table enters only through a new protocol version.

**Permission to train is separate from label qualification.** All catalogued data may be used to
train private models. Whether a dataset's labels are benchmark ground truth is decided only by
sections 3 and 4. A model trained on any dataset whose catalogue policy is not `open` is recorded
with `distribution: "private"` (section 11.2).

**The MSP-Podcast mirror is training-only.** It is not the official release and carries no speaker
identity and no official split, so it is never in dev or test and never scored. Its 56,157 rows
whose label the mirror introduced rather than the source are ledgered with `label: null` and
reason `label_not_from_source`; they are never gold labels, and a model that uses their mirror
labels as targets lists them as such in its provenance (section 6.1). The mirror is not exempt
from the checks of section 6: its hashes are compared with every dev and test row, and where its
dataset card cannot rule out source recordings or speakers shared with a test dataset, the results
on that test dataset of every model trained on the mirror carry `overlap_unverified`. An official
MSP-Podcast release would be a different dataset and a new table row.

The bundle (`bench-v1`, built by workstream WS4) records for every dataset a dataset card with its
source URL, revision or checksum, licence id, catalogue policy, native sample rate, row count, and
the source collections and speaker populations it may share with other datasets.

## 4. Label policy

### 4.1 Label sets and source maps

Every label set has an identifier `<name>-<count>-v<version>`. Labels are lowercase ASCII ids and
are always ordered by Unicode code point.

| Label set id | Labels | Used by |
|---|---|---|
| `ser-8-v1` | angry, calm, disgust, fearful, happy, neutral, sad, surprised | SER's shipped output contract (all profiles) |
| `ravdess-8-v1` | angry, calm, disgust, fearful, happy, neutral, sad, surprised | `ravdess` |
| `crema-d-6-v1` | angry, disgust, fearful, happy, neutral, sad | `crema-d` |
| `mesd-6-v1` | angry, disgust, fearful, happy, neutral, sad | `mesd` |
| `spanish-meacorpus-2023-6-v1` | angry, disgust, fearful, happy, neutral, sad | `spanish-meacorpus-2023` |

The label of a row is read from the raw source category at the pinned source revision, never from
a label already converted by SER's catalogue (whose maps include aliases and coercions):

| Dataset | Raw category source | Raw category → label |
|---|---|---|
| `ravdess` | third `-`-separated field of the file name | `01` neutral, `02` calm, `03` happy, `04` sad, `05` angry, `06` fearful, `07` disgust, `08` surprised |
| `crema-d` | third `_`-separated field of the file name | `ANG` angry, `DIS` disgust, `FEA` fearful, `HAP` happy, `NEU` neutral, `SAD` sad |
| `mesd` | category named by the source (D4-2 records where), case-folded | `anger` angry, `disgust` disgust, `fear` fearful, `happiness` happy, `neutral` neutral, `sadness` sad |
| `spanish-meacorpus-2023` | category named by the source (D4-2 records where), case-folded | `anger` angry, `disgust` disgust, `fear` fearful, `joy` happy, `neutral` neutral, `sadness` sad |

A raw category not in its dataset's map is excluded for that dataset (ledger reason
`label_not_scored`).

### 4.2 Renaming, not mapping

Every entry of 4.1 is a **renaming**: the source's category and the label id name the same
emotion. Renaming is one-to-one.

A source category is never **mapped** onto a different category. In particular boredom → neutral,
sleepy → neutral, amused → happy, anxious → fearful, and neutral → calm are forbidden. Some mappings
of this kind exist in SER's dataset catalogue; the benchmark does not use them. The same rule
applies to a model's output labels: a model declares a label set whose ids are renamings of its
outputs.

### 4.3 Scored label set of a model on a dataset

For dataset `d` with label set `S_d` and model `m` with label set `M_m`:

- the scored labels are `K = S_d ∩ M_m`, further reduced to the labels that have at least one row in
  `d`'s test partition;
- rows whose true label is not in `K` are not scored for this pair; their count is reported as
  `rows_label_not_in_model`;
- the returned vector `q_i` is **invalid** if its length differs from `|M_m|`, any value is not
  finite or is negative, `|Σ_{b∈M_m} q_i(b) − 1| > 10^-4`, or `Σ_{b∈K} q_i(b) = 0`. An invalid
  vector makes the row a failed row;
- otherwise it is restricted to `K` and renormalised: `p_i(a) = q_i(a) / Σ_{b∈K} q_i(b)` for
  `a ∈ K`;
- the prediction is `ŷ_i = argmax_{a∈K} p_i(a)`; ties go to the label first in code-point order;
- `mean_out_of_set_mass` = the mean of `1 − Σ_{b∈K} q_i(b)` over the rows with a valid vector;
  null if there is none.

If `|K| = 0` the dataset is not scored for the model (`scored_labels: []`, metrics null). If
`|K| = 1` the metrics are reported with the flag `degenerate_label_set`, and the result enters no
language aggregate and no decision.

**Selection basis.** For selection (sections 10.2 to 10.5) the label set of dataset `d` is fixed
for all candidates: `K*_d = S_d ∩ ser-8-v1`, reduced to labels with at least one test row. A
candidate whose `K` differs from `K*_d` on a required dataset is not selectable (flag
`label_coverage_incomplete`). For a descriptive comparison of two models with different label sets,
both are rescored on `K_AB = S_d ∩ M_A ∩ M_B` (restricted to labels with test rows), and the
report declares `K_AB`.

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

### 5.2 Identity ledger and speaker groups

Identity is fixed before any split, in a pinned identity ledger (`ser.bench.identity-ledger/v1`,
its SHA-256 in the manifest). It holds:

- the nodes of every bundle row, in separate namespaces: `<dataset>:speaker:<id>`,
  `<dataset>:session:<id>` and `source:<id>` (one id per source recording);
- alias records, each joining two nodes and citing the dataset card that states they are the same
  person or the same source recording;
- for every pair of datasets whose cards say they may share speakers or source recordings, whether
  their identities are `resolved` (every shared node is aliased) or `unresolved`.

The speaker groups are the connected components of the graph whose edges join each row to its
nodes, plus the alias records. Every row of the bundle contributes edges, including rows excluded
by 4.1 or 5.1, so an excluded row can bridge two groups. Rows without any node are not in any group;
in a train/dev/test dataset they are excluded, reason `no_speaker_identity`. The group id is the smallest node id of its component in code-point order.

Results on a dataset that belongs to an `unresolved` pair carry `identity_unresolved` and are
descriptive only; the audit reports the count (section 6) and never claims zero shared speakers for
them.

### 5.3 Assignment

Seed: `17`. Assignment happens per component, in this order of precedence:

1. **Official splits.** A dataset's documented official split is adopted only if its dataset card
   shows it is speaker-disjoint (ledger reason `native_speaker_disjoint_split`). A component with
   rows of such a dataset takes that split. If a component has rows with two different official
   assignments, the bundle build fails.
2. **Generated splits.** Every other component is assigned by the first dataset, in code-point
   order, of its rows that has the train/dev/test role. For dataset `d`, let its remaining groups
   be `g_1 … g_n`:
   - if the dataset has fewer than 3 groups in total, it takes no part in assignment and all its
     rows are excluded, reason `too_few_speaker_groups`;
   - order the groups by the lowercase hexadecimal SHA-256 of the UTF-8 string
     `"17:" + d + ":" + group_id`, ascending;
   - with `R` = round half to even to an integer:
     `n_train = max(1, min(n − 2, R(0.6·n)))`, `n_dev = max(1, min(n − n_train − 1, R(0.2·n)))`,
     `n_test = n − n_train − n_dev`;
   - the first `n_train` groups are train, the next `n_dev` dev, the rest test.
3. **Other roles.** A row of a test-only dataset whose component was assigned train or dev is
   excluded, reason `speaker_in_training_partition`. A row of a training-only dataset whose
   component was assigned dev or test is excluded, reason `speaker_in_evaluation_partition`.
4. **Check.** If any train/dev/test dataset ends with an empty train, dev or test partition, or any
   test-only dataset ends with an empty test partition, the bundle build fails.

For `ravdess` (`n = 24`) this gives 14/5/5 speakers; for `crema-d` (`n = 91`) 55/18/18.

The seed is never changed to obtain a "better" split. A label with no test row after assignment is
dropped from that dataset's scored set and listed in the report (`labels_without_test_rows`).

### 5.4 Test-only, training-only and speaker-unverified rows

Every eligible row of a test-only dataset gets split `test`, reason `test_only_dataset`. Every
eligible row of a training-only dataset gets split `train`, reason `training_only_dataset`, unless
5.1, 5.3 or section 3 excludes it or nulls its label.

A dataset without individual speaker identity is never train or dev. It is test-only only if no row
of it, and no row of the same source collection, is in any train or dev partition, and none of its
audio hashes occurs in train or dev; otherwise it is excluded. Its results carry the flag
`speaker_unverified`, its bootstrap resamples utterances (section 8.3), and they are descriptive
only.

### 5.5 Split ledger and digest

The split ledger (`ser.bench.split-ledger/v1`) is a JSON Lines file with one object per bundle row,
including excluded rows, with exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `audio_sha256` | string | content hash, section 2.3 |
| `dataset` | string | dataset id |
| `label` | string or null | scored label id after renaming; null if not scored or not gold |
| `language` | string | BCP 47 tag (`en`, `es`) |
| `reason` | string | one of `assigned`, `decode_invalid`, `duplicate_conflicting_label`, `duplicate_identity_conflict`, `duplicate_same_label`, `label_not_from_source`, `label_not_scored`, `native_speaker_disjoint_split`, `no_speaker_identity`, `speaker_in_evaluation_partition`, `speaker_in_training_partition`, `test_only_dataset`, `too_few_speaker_groups`, `training_only_dataset` |
| `sample_id` | string | unique in the bundle (section 2.1) |
| `speaker_group` | string or null | group id of 5.2; null if the row has no node |
| `split` | string | `train`, `dev`, `test` or `excluded` |

The file follows the JSON Lines byte contract of section 11.3, with lines sorted by `(dataset,
sample_id)` in code-point order. The **ledger digest** is the lowercase hexadecimal SHA-256 of the
whole file's bytes. Rebuilding the bundle from the same sources must give the same digest.

### 5.6 Relation to SER's training recipe

The ledger above is authoritative for training, tuning and scoring every benchmark candidate.
SER's dataset recipe (`ser/_internal/data/dataset_audit.py`, `docs/dataset-recipes.md`) stays the
default for other training; it differs on purpose: 70/15/15 ratios, components per corpus,
`ssl_only` for ungrouped rows outside its official-split path, and, for duplicate hashes, rejection
in strict mode or quarantine of all members otherwise. Both enforce the same invariants: no speaker
group and no audio hash in more than one partition. A benchmark candidate built from any other
ledger, including one reconstructed by the recipe, is not selectable unless its provenance
(section 6.1) proves that none of its training, tuning or calibration rows shares a speaker group
or an audio hash with this ledger's test partition.

## 6. Leakage audit

The audit reads the bundle manifest, the split ledger, the identity ledger and, in a benchmark
run, the model card (section 6.1). It runs before every benchmark run and on its own. Checks, over
all datasets together:

| Check | Pass condition |
|---|---|
| A1 ledger digests | the recomputed split and identity ledger digests equal those recorded in the manifest |
| A2 coverage | every manifest row appears exactly once in the split ledger and vice versa |
| A3 shared speakers | 0 speaker groups appear in more than one of train, dev, test |
| A4 shared audio | 0 `audio_sha256` values appear in more than one of train, dev, test |
| A5 duplicate audio | 0 `audio_sha256` values appear twice among non-excluded rows |
| A6 model data | section 6.1 |

A1 to A5 failing stops the run. `unresolved_pairs` greater than 0 does not stop it; the affected
datasets carry `identity_unresolved`. A6 never stops the run; it sets the model's provenance level
and flags.

### 6.1 Model provenance

Each model has a model card that records one provenance level, the lowest of its own and, where
it is built on a base model, the base model's:

- **`verified`**: a list of every row used to fit, tune or calibrate the artifact, each with
  dataset, `sample_id` and `audio_sha256`, whose SHA-256 is recorded in the artifact's metadata
  when it is trained (artifact-bound). The audit compares the listed rows with the test partition
  by audio hash and by speaker group (resolved through the identity ledger), and their source
  collections with each test dataset's;
- **`declared`**: the publisher's documented training-data list, quoted in the card with its URL,
  which names no test dataset and no source collection of one. The audit compares at dataset and
  source-collection level only. Results carry `provenance_declared`;
- **`unknown`**: anything else, including an incomplete list. Results carry `provenance_unknown`.

If any comparison finds an overlap, the model is **`contaminated`** and all its results carry
`contaminated`. `verified` and `declared` models may enter confirmatory selection; `unknown` and
`contaminated` ones are reported descriptively and cannot be selected. The owner's own recordings
(E3-4) are free of contamination by construction; for a `declared` model they are the check that
matters most.

The legacy `fast` artifact was trained on all 24 RAVDESS actors with a clip-level split, so its
card is `contaminated` on `ravdess`; section 9 gives its use.

### 6.2 Audit output

The audit prints exactly these lines, in this order, then the offending ids (one per line, sorted)
if any check failed, and exits with status 0 on pass, 2 on failure:

```text
ser-bench leakage-audit protocol=1 bundle=<bundle id>
ledger_sha256=<recorded> recomputed=<recomputed> match=<true|false>
identity_ledger_sha256=<recorded> recomputed=<recomputed> match=<true|false>
manifest_rows=<n> ledger_rows=<n> missing_from_ledger=<n> missing_from_manifest=<n>
rows=<n> train=<n> dev=<n> test=<n> excluded=<n>
shared_speakers train/dev=<n> train/test=<n> dev/test=<n> unresolved_pairs=<n>
shared_audio train/dev=<n> train/test=<n> dev/test=<n>
duplicate_audio_scored=<n>
model_provenance=<verified|declared|unknown|contaminated|not_checked> overlap_rows=<n|null> overlap_groups=<n|null> overlap_datasets=<n|null>
result=<PASS|FAIL>
```

`not_checked` appears only when the audit runs on its own without a model card; a benchmark run
without a model card is refused. The overlap counts are null for `unknown` and `not_checked`;
`overlap_rows` and `overlap_groups` are also null for `declared`.

## 7. Scoring a row

### 7.1 Notation

For one model on one dataset's test partition: rows `i = 1 … N` (rows with a true label in `K`),
`k = |K|`, true label `y_i`, prediction `ŷ_i` (section 4.3) or `∅` for a failed row, confidence
`c_i = max_{a∈K} p_i(a)` as an exact rational (section 11.1). `N'` is the number of rows that are
not failed; `coverage = N'/N`.

### 7.2 Failed rows

A failed row counts in the recall denominator of its true label and is never correct. It appears
in the confusion matrix in a final column `no_prediction`. It has no confidence and is left out of
the calibration error. The report states `n_failed` and `coverage`. ECE is therefore conditional
on success: a model that fails on hard rows can look better calibrated. It is a diagnostic, and
acceptance requires `n_failed = 0` (section 10.4).

### 7.3 Fitted settings

Any probability calibrator, threshold or hyperparameter of a model is fitted on train or dev only
and frozen before the model sees any test row. The model card identifies the calibrator (SHA-256)
and the ledger rows used for tuning; the report records them (section 11.2).

## 8. Metrics

### 8.1 Per dataset

All metrics are computed in exact rational arithmetic from integer counts and the exact
confidences of 7.1. Point values and interval bounds are rounded once, half to even, to 6 decimal
places, only when written (section 11.3).

- **Confusion matrix** `C[a][b]` = number of rows with `y_i = a` and `ŷ_i = b`, for `a ∈ K` (rows)
  and `b ∈ K ∪ {no_prediction}` (columns), both in code-point order with `no_prediction` last.
- **Support** `n_a = Σ_b C[a][b]`: every row whose true label is `a`, failed or not.
- **Per-class recall** `R_a = C[a][a] / n_a`.
- **UAR** (unweighted average recall, the primary metric) `= (1/k) Σ_{a∈K} R_a`.
- **Per-class precision** `P_a = C[a][a] / Σ_{x∈K} C[x][a]`; `P_a = 0` if that column sum is 0.
- **Per-class F1** `F_a = 2·P_a·R_a / (P_a + R_a)`; `F_a = 0` if `P_a + R_a = 0`.
- **Macro-F1** `= (1/k) Σ_{a∈K} F_a`.
- **Accuracy** `= Σ_a C[a][a] / N` (reported for continuity; never used in a decision).
- **Expected calibration error** (top-label, 10 equal-width bins):
  - bin of row `i`: `m_i = min(10, ⌊10·c_i⌋ + 1)`, so bin `m` holds `[(m−1)/10, m/10)` and bin 10
    holds `[0.9, 1.0]`;
  - for each non-empty bin: `acc_m` = fraction of its rows with `ŷ_i = y_i`, `conf_m` = mean of
    its `c_i`;
  - `ECE = Σ_m (|B_m| / N') · |acc_m − conf_m|`;
  - empty bins contribute 0 and are reported with count 0 and null accuracy and confidence;
  - if `N' = 0`, ECE is null and the result is flagged `no_predictions`.

Labels in `S_d` but absent from the test partition are not part of `K` and do not enter UAR as
zero. SER's `compute_ser_metrics` (`ser/_internal/train/metrics.py`) differs in two ways and is
never applied directly to benchmark rows that include failed rows: it gives a label passed in
`labels` but absent from the data a recall of 0.0, and it has no failure column, so with
`labels = K` it drops rows whose prediction is outside `K` and its recalls diverge from the
benchmark's. On rows without failures and with all labels present, these definitions equal
scikit-learn's macro recall and macro F1 with `zero_division=0` over `K`.

### 8.2 What is reported per dataset

UAR, macro-F1, accuracy, per-class recall, precision and F1, the confusion matrix, ECE with its
per-bin table, `n_rows`, `n_speakers`, `n_failed`, `coverage`, `rows_label_not_in_model`,
`mean_out_of_set_mass`, and confidence intervals for UAR, macro-F1 and ECE. `n_speakers` is the
number of test speaker groups; it is null for a `speaker_unverified` dataset. A dataset with fewer
than 5 test speaker groups is flagged `underpowered` on the model and every baseline row; its
intervals are reported but are not reliable.

### 8.3 Bootstrap confidence intervals

- **Unit: the speaker group.** Rows of one speaker are correlated (same voice, same session, same
  prompts); the claim is about new speakers, so groups are resampled. A `speaker_unverified`
  dataset resamples utterances instead, and its intervals carry that flag.
- **Universe.** The units of dataset `d` are fixed by the ledger before any model is scored: all
  test speaker groups of `d` in code-point order of group id (or, for utterance resampling, all
  test rows of `d` in code-point order of `sample_id`), `S` of them, whatever their labels.
- **Resamples:** `B = 10,000`.
- **Random draws:** for each dataset, a generator `numpy.random.Generator(numpy.random.PCG64(s_d))`
  with `s_d` = the first 8 bytes, read as a big-endian unsigned integer, of SHA-256 of the UTF-8
  string `"17:bootstrap:" + d`. For `b = 0 … B−1`:
  `idx_b = rng.integers(0, S, size=S, dtype=numpy.int64)`; resample `b` is the multiset of all rows
  of units `idx_b`, with repetition. The report records the numpy version. The draws depend only on
  the dataset, never on the model, so every model and baseline sees the same resamples (paired).
- **Metric on a resample:** the formulas of 8.1, over the resample's rows whose label is in the
  scored set, with `K` reduced to labels that have at least one such row. A resample with no such
  row, or (for ECE) no successful row, has no value.
- **Interval:** if any resample has no value, the interval is null and flagged `ci_undefined`.
  Otherwise sort the `B` exact values `v_0 ≤ … ≤ v_{B−1}`; for `q` the lower and upper tail
  levels, `h = q·(B−1)`, value `= v_⌊h⌋ + (h − ⌊h⌋)·(v_⌈h⌉ − v_⌊h⌋)` (the "linear" percentile).
  Descriptive intervals use `q ∈ {0.025, 0.975}`; with `B = 10,000`, `h = 249.975` and `9749.025`.
  Familywise-adjusted intervals are defined in 10.2.

### 8.4 Per language, never pooled

For benchmark language `ℓ`, `D_ℓ` is its datasets whose results are not descriptive-only (section
1) and not `degenerate_label_set`, each weighted equally whatever its size:

- `UAR_ℓ = mean_{d∈D_ℓ} UAR_d`, likewise `MacroF1_ℓ`, `ECE_ℓ`; null if `D_ℓ` is empty or any of its
  values is null;
- `chance_ℓ = mean_{d∈D_ℓ} 1/k_d`;
- `n_rows`, `n_failed` and `coverage` are summed over `D_ℓ` (`coverage = Σ N' / Σ N`);
- its interval uses, for each `b`, the mean over `d ∈ D_ℓ` of the resample-`b` values (datasets are
  resampled independently, stratified by dataset). It is null if any dataset interval is null. If
  a speaker group is in the test partition of two datasets of `D_ℓ`, independent draws misstate
  the uncertainty: the language intervals are null and flagged `ci_unavailable_shared_groups`.

Rows of different datasets are never merged into one confusion matrix or one metric. No figure
across languages is computed.

## 9. Baselines

Every report includes these rows for every scored dataset, on the same test partition and scored
label set `K` as the model:

| Baseline | Definition | Values |
|---|---|---|
| `chance` | analytic expectation of uniform random guessing over `K`; its confusion matrix is the expected one, `E[C[a][b]] = n_a/k` | `UAR = 1/k`; **macro-F1 of the expected confusion matrix** `= (1/k) Σ_a 2·π_a·(1/k) / (π_a + 1/k)` with `π_a = n_a/N` (not the expected macro-F1 of random predictions); accuracy `= Σ_a π_a·(1/k) = 1/k`; per-class recall `1/k`, precision `π_a`, F1 as above; confusion, ECE, intervals and `n_failed` null |
| `majority` | always predicts `m`, the label of `K` with the most rows in the dataset's train partition (for a test-only dataset, in its test partition, flagged `majority_from_test`: an oracle, not a deployable baseline); ties to the first label in code-point order; probability 1 on `m` | `UAR = 1/k`; `MacroF1 = (1/k)·2·π_m/(π_m + 1)`; accuracy `= π_m`; `ECE = 1 − π_m`; all fields of 8.2 filled, intervals by 8.3 |
| `fast@16k` | the legacy `fast` artifact (SHA-256 pinned in the selection plan), through the public call with the input of section 2.2 | measured |
| `fast@native` | the same artifact with the file's own sample rate | measured |

Both `fast` rows run at the code revision of the report. If the public call at that revision
cannot take the native rate, `fast@native` runs at the last commit whose inference path used it;
that row records the commit and the rate actually fed and carries `confounded_code_revision`, and
its difference from `fast@16k` is not attributed to the sample rate. Where a dataset's native rate
is 16 kHz, the two `fast` rows are equal; both are reported.

The `fast` rows are `contaminated` (section 6.1) and also carry `trained_on_test_speakers` on
`ravdess`. They are descriptive, with one use: `fast@16k` is the frozen reference of U11
condition 2 (section 10.4), a flagged historical hurdle. No clean rerun is required; the
contamination can only raise the reference on `ravdess`, which makes the hurdle harder. `fast` is
never a selection candidate.

A baseline value that cannot be computed on a dataset (for example, every `fast` row failed) is
null, and nulls propagate as in 8.4.

## 10. Decision rule

### 10.1 Primary metric

Decisions use UAR only. Macro-F1, ECE and per-class figures are reported but never decide which
model is better (ECE enters only acceptance, 10.4).

### 10.2 "Model A beats model B"

**Confirmatory comparisons** are made only between candidates of a frozen selection plan (10.5),
on the required datasets, on `K*_d` (section 4.3), with the same resamples (8.3). For each
resample `b`: `Δ_b = UAR_A,b − UAR_B,b` per dataset, and `Δ_ℓ,b` = mean of the dataset `Δ_b` over
`D_ℓ` per language. The interval of `Δ_ℓ` is familywise-adjusted by Bonferroni: with `M` = the
number of candidate pairs times the number of benchmark languages, fixed in the plan, the tail
levels are `q = 0.025/M` and `1 − 0.025/M` (8.3). Then:

- A beats B **in language ℓ** if the lower end of the adjusted `Δ_ℓ` interval is greater than 0;
- A beats B **overall** if A beats B in at least one benchmark language and B beats A in none;
- otherwise neither beats the other. A null interval is no win for either side.

**Descriptive comparisons** use unadjusted 95% intervals, on `K*_d` or on `K_AB`; they are labelled
descriptive and decide nothing.

### 10.3 Cost order

When candidates must be ordered by cost, they are compared in this order, on numbers measured on
the same machine class by the CPU performance harness (workstream WS7) and frozen in a cost
evidence file: lower 95th percentile processing time per second of audio at 2 threads; then lower
peak memory; then smaller model file; then smaller `artifact_sha256` in code-point order.

### 10.4 Acceptance thresholds (checklist U11)

A model is acceptable only if, for every benchmark language `ℓ` (`en` and `es`), on `K*_d`:

1. `UAR_ℓ ≥ 2 · chance_ℓ`;
2. `UAR_ℓ ≥ UAR_ℓ(fast@16k) + 0.10`, the reference being the `fast@16k` row of section 9;
3. `ECE_ℓ ≤ 0.10`;
4. `n_failed = 0` on every dataset of `D_ℓ`;
5. a check on at least 100 of the owner's own labelled turns in `ℓ` is recorded from the E3-4 kit,

and the E3-4 check on the owner's Portuguese recordings is recorded. The report holds each check's
status and turn count, not its content.

Conditions 1 to 3 are evaluated on point estimates: each side is rounded half to even to 6 decimal
places, then compared exactly. A null value fails its condition. Intervals are always reported
next to them but do not enter acceptance.

### 10.5 Selection

1. Before any candidate is scored on a test partition, a selection plan
   (`ser.bench.selection-plan/v1`) is committed. It lists the candidate artifacts (SHA-256) with
   their provenance level, the required datasets per benchmark language, `K*_d` for each, the
   `fast` artifact and its call options, the thresholds and precision of 10.4, `M`, and the cost
   evidence file's SHA-256. Every report used for selection records the plan's SHA-256.
2. A candidate is eligible if its provenance is `verified` or `declared`, it carries none of the
   descriptive-only flags of section 1 and no `label_coverage_incomplete`, it has a result on
   every required dataset, and it passes 10.4.
3. If no candidate is eligible, nothing is selected. No default is presumed.
4. Otherwise the selected candidate is the eligible one that beats (10.2, overall) the most other
   eligible candidates; ties go by the cost order of 10.3.

## 11. Report format

### 11.1 Inputs and files

A report is a pure function of these inputs, each identified in the report by its SHA-256: this
document; the bundle manifest, split ledger and identity ledger; the model card and the `fast`
card; one predictions file per configuration (`model`, `fast@16k`, `fast@native`); the selection
plan and cost evidence file, if the report is used for selection; and the E3-4 status file
(`ser.bench.own-recordings-status/v1`: per language `language`, `status` (`recorded` or
`missing`), `turns`). `chance` and `majority` need no predictions: they are computed from the
ledger. The audit lines are recomputed from these inputs.

The harness writes the predictions file `<stem>.predictions.<config>.jsonl`
(`ser.bench.predictions/v1`), one line per scored row, keys `dataset`, `probabilities`,
`sample_id`, `status`, lines sorted by `(dataset, sample_id)`. `status` is one of `ok`,
`not_success`, `raised`, `timeout`. `probabilities` is null unless `status` is `ok`; otherwise it is
the returned vector in the model's label order, each value written as the shortest decimal string
that converts back to the same IEEE 754 double (Python `repr(float(x))`). Scoring reads these
strings as exact rationals, so the archived file reproduces every argmax, confidence and bin.
Predictions of the owner's own recordings are never written by this harness.

A run writes `<stem>.json` and `<stem>.md` next to the predictions files, with a stem chosen by the
operator (for example `benchmarks/reports/2026-12-baselines`).

### 11.2 JSON (`ser.bench.report/v1`)

Types: `str`; `int`; `num` (fixed 6-decimal number, 11.3); `hex64` (64 lowercase hex characters);
`[T]` an array; `T|null` nullable. Every field is always present.

```text
report        {schema: "ser.bench.report/v1", protocol, code, inputs, bundle, model, input,
               settings, audit, datasets: [dataset], languages: [language], baselines: [baseline],
               acceptance, comparisons: [comparison]}
protocol      {id: "ser-bench-protocol", version: 1, document_sha256: hex64}
code          {git_commit: str (40 hex), git_dirty: false, ser_version: str}
inputs        {manifest_sha256: hex64, split_ledger_sha256: hex64, identity_ledger_sha256: hex64,
               model_card_sha256: hex64, fast_card_sha256: hex64,
               predictions: [{config: "model"|"fast@16k"|"fast@native", sha256: hex64}],
               selection_plan_sha256: hex64|null, cost_evidence_sha256: hex64|null,
               own_recordings_status_sha256: hex64}
bundle        {id: str, content_hash_contract: "pcm16k-s16le-v1",
               preprocessing_contract: "ser-preproc-v1", preprocessing_sha256: hex64}
model         {id: str, artifact_sha256: hex64, label_set_id: str, labels: [str],
               distribution: "open"|"private",
               provenance: "verified"|"declared"|"unknown"|"contaminated",
               provenance_url: str|null, calibrator_sha256: hex64|null,
               tuning_rows_sha256: hex64|null, flags: [flag]}
input         {call: str, call_options: {str: str}, channels: 1, sample_rate: 16000|"native",
               timeout_seconds: int}
settings      {bootstrap_resamples: 10000, bootstrap_seed_rule: "17:bootstrap:<dataset>",
               ece_bins: 10, familywise_contrasts: int|null, numpy_version: str, split_seed: 17}
audit         {lines: [str]}                       the 10 lines of 6.2, verbatim
dataset       {dataset: str, language: str, label_set_id: str, scored_labels: [str],
               labels_without_test_rows: [str], n_rows: int, n_speakers: int|null,
               n_failed: int|null, coverage: num|null, rows_label_not_in_model: int,
               mean_out_of_set_mass: num|null, metrics, ci, flags: [flag]}
metrics       {uar: num|null, macro_f1: num|null, accuracy: num|null,
               per_class: {<label>: {support: int, recall: num, precision: num, f1: num}},
               confusion: {rows: [str], columns: [str], matrix: [[int]]}|null,
               ece: num|null, ece_bins: [bin]|null}
bin           {bin: int, lower: num, upper: num, count: int, accuracy: num|null,
               confidence: num|null}
ci            {uar: [num, num]|null, macro_f1: [num, num]|null, ece: [num, num]|null,
               unit: "speaker"|"utterance"|null}
language      {language: str, datasets: [str], n_rows: int, n_failed: int|null,
               coverage: num|null, uar: num|null, macro_f1: num|null, ece: num|null,
               chance_uar: num|null, ci: ci, flags: [flag]}
baseline      {id: "chance"|"majority"|"fast@16k"|"fast@native", git_commit: str,
               sample_rate: 16000|"native", datasets: [dataset], languages: [language],
               flags: [flag]}
acceptance    {languages: [{language: str, conditions: [condition], pass: bool}],
               own_recordings: [{language: str, status: "recorded"|"missing", turns: int|null}],
               pass: bool}
condition     {id: "chance"|"fast_delta"|"ece"|"coverage"|"own_recordings", value: num|null,
               threshold: num|null, pass: bool}
comparison    {a: hex64, b: hex64, basis: "confirmatory"|"descriptive",
               languages: [{language: str, delta_uar: num|null, ci: [num, num]|null,
                            ci_adjusted: [num, num]|null, verdict: "a"|"b"|"none"}]}
flag          one of: ci_undefined, ci_unavailable_shared_groups, confounded_code_revision,
               contaminated, degenerate_label_set, identity_unresolved, label_coverage_incomplete,
               majority_from_test, no_predictions, overlap_unverified, provenance_declared,
               provenance_unknown, speaker_unverified, trained_on_test_speakers, underpowered
```

For `chance`, `metrics.confusion`, `metrics.ece`, `metrics.ece_bins`, `ci`, `n_failed` and
`coverage` are null. For the `coverage` and `own_recordings` conditions `value` and `threshold`
are null; `coverage` passes when every `n_failed` in the language is 0. A selection run also writes
`<stem>.selection.json` (`ser.bench.selection/v1`: `plan_sha256`, `candidates` and `eligible` as
sorted arrays of artifact SHA-256, `wins` as an object keyed by artifact SHA-256, `selected`
(hex64 or null), `reason` (`selected` or `no_eligible_candidate`)).

### 11.3 Byte contract

Two runs on the same inputs produce identical bytes.

- All files are UTF-8 without byte order mark; lines end with LF only; the file ends with exactly
  one LF; no line has trailing spaces; no wall-clock time, host name, user name, absolute path or
  duration appears.
- **JSON Lines** (ledgers, predictions): each line is the bytes of Python
  `json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`, no blank lines.
- **JSON** (report, selection): the bytes of Python
  `json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)` followed by LF, except that every
  `num` is written as the exact value rounded half to even to 6 decimal places, with exactly six
  digits after the point, no exponent, and `-0.000000` written `0.000000` (for example
  `0.750000`). Integers have no decimal point. Probability strings in predictions are strings, not
  numbers.
- **Array order:** `datasets` and per-language `datasets` by dataset id; `languages` and
  `own_recordings` by tag; labels, `scored_labels`, `labels_without_test_rows`, confusion rows and
  columns as in 8.1; `baselines` in the order of section 9; `ece_bins` by bin; `conditions` in the
  order of 10.4; `flags` unique and sorted; `predictions` by config in the order `model`,
  `fast@16k`, `fast@native`; `comparisons` by `(a, b)`; `audit.lines` as printed.

### 11.4 Markdown rendering

The Markdown file is generated from the JSON file alone by this template. Rules: lines as in
11.3; exactly one blank line between blocks; tables written `| a | b |` with one space inside each
border and a separator row of `|---|` per column; a `num` shown to 3 decimals, rounded half to even
from its 6-decimal JSON value; an interval shown `[l, u]`; null shown `—`; a flag list shown
comma-and-space separated, or `none` if empty. Repeated blocks follow the JSON array order.

~~~text
# SER benchmark report `<model.id>`

- protocol: `ser-bench-protocol` v1, document `<document_sha256>`
- code: `<git_commit>`, ser `<ser_version>`
- bundle: `<bundle.id>`, split ledger `<split_ledger_sha256>`
- model: `<artifact_sha256>`, labels `<label_set_id>`, provenance `<provenance>`, distribution `<distribution>`, flags <flags>
- input: <sample_rate> Hz, mono, call `<call>`, timeout <timeout_seconds> s

## Audit

```text
<audit.lines, one per line>
```

## Languages

| language | row | UAR | UAR CI | macro-F1 | ECE | coverage | flags |
|---|---|---|---|---|---|---|---|
<one line per language: model row, then one row per baseline>

## Dataset `<dataset>` (<language>)

| row | UAR | macro-F1 | accuracy | ECE | failed | flags |
|---|---|---|---|---|---|---|
<model row, then one row per baseline>

| label | support | recall | precision | F1 |
|---|---|---|---|---|
<one row per scored label, model only>

| true \ predicted | <columns> |
|---|<one ---| per column>
<one row per confusion row>

| bin | range | count | accuracy | confidence |
|---|---|---|---|---|
<ten rows>

## Acceptance

| language | condition | value | threshold | pass |
|---|---|---|---|---|
<one row per condition per language; then one row per own_recordings entry with condition own_recordings>

## Comparisons

| a | b | basis | language | ΔUAR | CI | adjusted CI | verdict |
|---|---|---|---|---|---|---|---|
<one row per comparison per language, or the single line "none">
~~~

Section 12.8 shows part of the rendering of the toy example.

## 12. Worked toy example

A toy dataset `toy` (language `und`, role test-only) of 20 rows: 4 speakers (S1 to S4), 3 labels,
label set `toy-3-v1` = angry, happy, neutral. The model's label set is the same, so `K` = {angry,
happy, neutral}, `k = 3`, and no renormalisation changes any vector. Row `toy-s1-04` failed
(status `raised`).

The toy departs from the real contract in these ways only: `audio_sha256` values are stand-ins
(`a01` … `a20`), not real hashes; speaker group ids are `S1` … `S4`, not namespaced node ids
(5.2); the dataset, language `und` and label set are synthetic; its intervals are not computed,
because the seeded draws cannot be done by hand (12.5); and the excerpts of 12.7 and 12.8 omit
fields and blocks, as marked there.

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
(5, 9, 10, 15, 19, 20). `N = 20`, `n_failed = 1`, `N' = 19`, `coverage = 19/20 = 0.950000`,
`n_speakers = 4` (flag `underpowered`, since 4 < 5). Every valid vector sums to 1 on `K`, so
`mean_out_of_set_mass = 0.000000`.

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
**0.052632**. Because row 4 failed, this ECE is conditional on success, and the model would fail
acceptance condition 4 of 10.4 whatever its ECE.

### 12.4 Baselines on the toy

`π_angry = 8/20 = 0.4`, `π_happy = π_neutral = 6/20 = 0.3`. The toy is test-only, so `majority`
uses the test partition and its row is flagged `majority_from_test` (an oracle): angry.

- `chance`: UAR = 1/3 = **0.333333**. Accuracy = Σ_a π_a·(1/3) = (0.4 + 0.3 + 0.3)/3 = 1/3 =
  **0.333333**. Macro-F1 of the expected confusion matrix: angry `2·0.4·(1/3)/(0.4 + 1/3)` =
  `(4/15)/(11/15)` = 4/11; happy and neutral `2·0.3·(1/3)/(0.3 + 1/3)` = `(1/5)/(19/30)` = 6/19;
  (4/11 + 6/19 + 6/19)/3 = ((76 + 132)/209)/3 = 208/627 = **0.331738**.
- `majority`: recalls 1, 0, 0, so UAR = 1/3 = **0.333333**; precision of angry 8/20 = 0.4, F1 of
  angry `2·0.4·1/(0.4 + 1)` = 4/7, others 0; Macro-F1 = (4/7)/3 = 4/21 = **0.190476**;
  accuracy = **0.400000**; every confidence is 1.0, all 20 rows in bin 10 with accuracy 0.4, so
  ECE = |0.4 − 1.0| = **0.600000**.

### 12.5 One bootstrap resample, by hand

The real draws come from the seeded generator of 8.3 (for `toy`, `s_toy = 4825966949488780857`)
and cannot be done by hand. To show what one resample computes, take the draw (S3, S1, S3, S4).
Per speaker, correct/total by true label (from 12.1):

| speaker | angry | happy | neutral |
|---|---|---|---|
| S1 | 1/2 | 1/2 | 1/1 |
| S2 | 1/2 | 1/1 | 1/2 |
| S3 | 2/2 | 1/2 | 1/1 |
| S4 | 2/2 | 1/1 | 1/2 |

Resample recalls: angry (2+1+2+2)/(2+2+2+2) = 7/8; happy (1+1+1+1)/(2+2+2+1) = 4/7; neutral
(1+1+1+1)/(1+1+1+2) = 4/5. UAR_b = (7/8 + 4/7 + 4/5)/3 = ((245 + 160 + 224)/280)/3 = 629/840 =
0.748810. The descriptive interval of 8.3 interpolates the sorted 10,000 such values at
`h = 0.025·9999 = 249.975` and `h = 0.975·9999 = 9749.025`.

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
including its LF), and its SHA-256 is:

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

The excerpt shows `datasets[0]` in full and, from `baselines`, only the `majority` entry with
some of its fields. The `ece_bins` objects and some small objects are written on one line, and bins
2 to 5 are replaced by a marker; a real report writes every field and object in the layout of 11.3
and all ten bins.

```json
{
  "baselines": [
    {
      "datasets": [
        {
          "ci": {"ece": null, "macro_f1": null, "uar": null, "unit": "speaker"},
          "coverage": 1.000000,
          "dataset": "toy",
          "flags": ["majority_from_test", "underpowered"],
          "metrics": {"accuracy": 0.400000, "ece": 0.600000, "macro_f1": 0.190476, "uar": 0.333333},
          "n_failed": 0
        }
      ],
      "id": "majority"
    }
  ],
  "datasets": [
    {
      "ci": {"ece": null, "macro_f1": null, "uar": null, "unit": "speaker"},
      "coverage": 0.950000,
      "dataset": "toy",
      "flags": ["underpowered"],
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

The majority row's `coverage` is 20/20 = 1.000000: it predicts every row.

### 12.8 Sample report (Markdown excerpt)

The excerpt shows the first two tables of the dataset block, with the `fast` rows omitted.

```markdown
## Dataset `toy` (und)

| row | UAR | macro-F1 | accuracy | ECE | failed | flags |
|---|---|---|---|---|---|---|
| model | 0.694 | 0.715 | 0.700 | 0.053 | 1 | underpowered |
| chance | 0.333 | 0.332 | 0.333 | — | — | underpowered |
| majority | 0.333 | 0.190 | 0.400 | 0.600 | 0 | majority_from_test, underpowered |

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
- Spanish acceptance needs at least one speaker-verified Spanish dataset. If D4-2 finds no
  individual speaker identity in `mesd` or `spanish-meacorpus-2023`, `D_es` is empty, `UAR_es` is
  null, and no model can be accepted (10.4).
- `declared` provenance cannot be checked row by row; the owner's own recordings are the only
  contamination-free evidence for such a model.
- Intervals depend on numpy's PCG64 stream; a numpy version that changes it changes the intervals,
  which is why the report records the version.
- A wall-clock timeout can fail a row on a slow machine; acceptance treats it as any other
  failure.
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

None.

Recorded owner decisions (2026-11-28), applied above:

- Benchmark languages are English and Spanish. Portuguese is checked only on the owner's own
  recordings (sections 1, 3, 10.4).
- All catalogued data, including the MSP-Podcast mirror, may be used to train private models;
  models trained on restricted data stay private (section 3).
- The benchmark chooses the first model, and SER's eight labels are the shipped output contract
  (sections 1, 4.1, 10.5).
