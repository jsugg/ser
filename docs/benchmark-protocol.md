# SER benchmark protocol

Protocol identifier: `ser-bench-protocol`, version `1`. A change to any rule below creates a new
protocol version; reports produced under different versions are never compared with each other.

## 1. Purpose and claims

The benchmark measures how well a model assigns categorical emotion labels to single utterances,
per dataset and per benchmark language (English `en` and Spanish `es`), and how well its
confidence matches its accuracy. It chooses SER's first default model (section 10.5); no existing
default is presumed. SER's eight labels (`ser-8-v1`, section 4.1) are the shipped output contract.

A report supports these claims and no others: the model's UAR, macro-F1, per-class recall,
confusion matrix and calibration error, with confidence intervals, on the test partition of each
named dataset and aggregated per benchmark language (section 8.4), for speakers whose speech was in
no train or dev partition of the bundle (for a dataset without speaker identity: whose source
collection is in no training set, 5.4) and, for `verified` provenance, in no data the model was
fitted, tuned or calibrated on (section 6.1); whether model A beats model B (section 10.2); and
whether the model passes the acceptance thresholds (section 10.4).

It supports no claim about speech, speakers, recording conditions or languages outside the scored
test partitions (acted corpora say little about conversational or clinical speech); about
Portuguese, which is checked only on the owner's own recordings by the kit of brief E3-4, whose
status alone the report records; about clinical or psychological validity of the labels; about
agreement with the owner's own reading of his recordings (E3-4); or about fairness across
demographic groups. No figure is pooled across datasets or languages.

**Descriptive-only flags.** A result carrying a dataset flag (`identity_unresolved`, 5.2;
`degenerate_label_set`, 4.3; `source_in_training`, 6) or a model flag (`provenance_unknown`,
`test_overlap`, `overlap_unverified`, `separation_unverified`, 6.1) is reported and never enters
sections 10.2 to 10.5, with the single exception of the U11 comparator of section 9.

The only earlier result (RAVDESS accuracy 58.6%, UAR 56.8%, `fast` profile) came from a split by
clip, not by speaker, and is not comparable with any figure produced here.

## 2. Input form

Every model is scored through SER's public utterance-analysis call (workstream WS1), so the
benchmark measures what a consumer calls; the harness neither calls internal model classes nor
aggregates windows. All rows come from a **benchmark bundle** built by workstream WS4; this
version's bundle is `bench-v1`. Its manifest pins every input of sections 2 to 5, and a change to
any of them is a new bundle id.

### 2.1 Preprocessing contract `ser-preproc-v1`

The manifest pins, under `preprocessing`, the decoder (name and version), the resampler (name,
version and every option it accepts, written out even where it equals the library default), and
the call options and timeout of section 2.2. Every row has a `sample_id`, assigned once when the
bundle is built and never reused or renumbered within a bundle id:

```text
sample_id = <dataset id> ":" <path> [ "#" <start> "-" <end> ]  |  <dataset id> "!" <n>
```

`<path>` is the source file's POSIX path relative to the dataset root (`/` separators, no leading
`./`, Unicode NFC). The suffix is present only for a segment row: `<start>` and `<end>` are the
source's frame indices at the file's native rate, as decimal integers without leading zeros; the
id does not validate them. A source row whose path or bounds cannot be written in that form gets
the source-row id `<dataset id> "!" <n>`, with `<n>` its zero-based position, in decimal, in the
dataset's pinned source listing. Two rows with the same `sample_id` fail the bundle build.

For every row the bundle builder and the harness:

1. decode the source file with the pinned decoder to float64 samples; integer PCM of `b` bits is
   divided by `2^(b−1)`, floating-point PCM is taken as is;
2. for a segment row, keep frames `start` up to but excluding `end`;
3. mix all channels to one by the arithmetic mean of the channels, sample by sample, in float64;
4. resample to 16,000 Hz with the pinned resampler in float64; a file already at 16,000 Hz is not
   resampled.

A row with a source-row id, whose decode fails, whose segment bounds do not satisfy
`0 ≤ start < end ≤` the decoded frame count, or whose audio has zero frames or a non-finite
sample is excluded when the bundle is built (ledger reason `decode_invalid`), so the
scored rows are the same for every model.

### 2.2 The call

The harness passes the result of 2.1, cast to float32, with sample rate 16,000 and the pinned call
options, to one loaded analyser reused for all rows of a run, and records the returned status and
probability vector. Only the `fast@native` baseline (section 9) instead receives the output of
steps 1 to 3 at the file's own rate.

A row whose call returns any status other than success, raises, exceeds the pinned timeout
(`timeout_seconds`, 120 in `bench-v1`), or returns an invalid vector (section 4.3) is a **failed
row**. Failed rows stay in the scored set (section 7).

### 2.3 Audio content identity

Duplicate detection and the leakage audit compare audio by the content hash contract
`pcm16k-s16le-v1`:

1. steps 1 to 4 of section 2.1;
2. clip each sample to [-1.0, 1.0], multiply by 32767, round half to even to an integer;
3. serialise as signed 16-bit little-endian integers, no header;
4. SHA-256 of those bytes, lowercase hexadecimal: the row's `audio_sha256`.

A `decode_invalid` row has `audio_sha256` null; it stays in the ledger with its reason and takes
no part in duplicate detection or any hash comparison. A change of preprocessing contract or of
any step above is a new contract identifier and requires recomputing every hash in the bundle.
This contract detects identical audio only (section 13).

## 3. Datasets and roles

Roles: **train/dev/test** (split by speaker as in section 5; train and dev are available to model
builders, test is scored); **test-only** (every eligible row is in test; none is used to build or
tune a model); **training-only** (rows may be used to build models; none is ever in dev or test or
scored); **excluded** (not in the bundle).

| Dataset id | Language | Kind | Speaker identity | Licence id (catalogue policy) | Role in `bench-v1` |
|---|---|---|---|---|---|
| `ravdess` | `en` | acted, 24 actors | yes (actor in filename) | `cc-by-nc-sa-4.0` (`noncommercial`) | train/dev/test |
| `crema-d` | `en` | acted, 91 actors | yes (actor in filename) | `odbl-1.0` (`share_alike`) | train/dev/test |
| `mesd` | `es` (Mexican) | acted | none known; verified by brief D4-2 | `cc-by-4.0` (`open`) | test-only candidate (admission rule below) |

**Source policy.** The datasets of `bench-v1`, their roles and their label maps are fixed by its
source policy (`ser.bench.source-policy/v1`): one immutable JSON object with the keys `schema`
(that id), `nonce` (32 random bytes in lowercase hexadecimal), `may_train_unlisted` (below) and
`datasets`, which maps each dataset id to an entry with these keys:

- every entry: `language` (BCP 47 tag), `role` (`train/dev/test`, `test-only candidate`,
  `training-only` or `excluded`), `licence_id` and `catalogue_policy` (the catalogue's, as in the
  table above) and `may_train` (below);
- a `train/dev/test` or `training-only` entry: `speaker_identity_verified` (whether every row's
  speaker identity is verified at the pinned source revision); a candidate has no such key, as
  its identity comes from D4-2 only (below);
- a `train/dev/test` or `test-only candidate` entry: `label_set` and `labels` (4.1),
  `raw_category_source` (where the raw category is read), `normalisation` (`none` or `casefold`,
  4.1) and `raw_map` (every normalised raw category to its label id);
- a `training-only` entry: `copy_of` (the original source's URL), `copy_source` (the copy's
  source URL), `copy_revision` (the copy's revision or checksum), `official_split` (false),
  `introduced_label_rows` (the number of rows whose label the copy introduced rather than the
  source) and `introduced_label_sample_ids_sha256` (the digest of their list, below);
  `copy_revision` and `introduced_label_sample_ids_sha256` are null until pinned;
- an `excluded` entry: `exclusion` (the ids of the exclusions below that apply).

The policy file is the UTF-8 bytes of Python
`json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` followed by LF. The
source policy of `bench-v1` is the file whose lowercase hexadecimal SHA-256 is

```text
676e3bb6fccd89373237e31d873057231584d254f1f6f1387f52e33fa1621f2e
```

and the manifest records that value as `source_policy_sha256`. The digest is taken over the
file's bytes as read, never over a re-serialisation of the decoded object; separately, those bytes
must equal the canonical serialisation of the object they decode to, and every entry must have
exactly the keys its role requires. A change to the policy is a new protocol version.

**Bundle membership.** A bundle may hold any policy entries and no other datasets, each bound
exactly as its entry binds it and never rebound: a candidate only if the admission route admits it;
a `training-only` entry or not, as its builder chooses; never an `excluded` entry. Which datasets a
decision needs is decided only by the language cohorts `D_ℓ` of section 8.4, taken from the bundle,
and by 10.4: an empty cohort has null language values (8.5) and fails 10.4. A bundle is invalid, so
its build fails and the harness refuses it, if its `source_policy_sha256` differs from the value
above or the policy file fails the checks above; if its datasets break that rule; if a dataset has a
language, label set, raw category source, normalisation or map other than its entry's, or a role
other than the one its entry and the admission route give it; if a dataset card's licence id or
catalogue policy differs from its entry's; or if it holds a training-only entry that fails a check
of the training-only paragraph below.

The table lists the policy entries this document names. The policy may hold further entries that
it does not list; every rule of this protocol applies to them through their role and properties,
and no dataset outside the policy enters a bundle under this protocol version.

**Exclusions.** An entry is `excluded` when its source has dimensional values only and no named
emotion categories (`no_named_categories`), when its language is not a benchmark language
(`not_benchmark_language`), or when it names fewer emotion categories than the six labels of a
candidate (4.1; `too_few_categories`).

**Admission of candidates.** `mesd` and every other entry whose role is `test-only candidate` are
in the catalogue but not yet qualified. Each enters a bundle only through this route: brief D4-2
records, for the pinned source revision, (a) the raw category inventory, read from the entry's raw
category source and normalised as the entry says (4.1), and (b) whether each row carries an
individual speaker id. If (a) differs from the dataset's map in its policy entry (for `mesd`, also
the map of section 4.1), the dataset is excluded until a new protocol version changes the map.
Otherwise its role is train/dev/test if (b) holds, and test-only if not (section 5.4). The
manifest records the D4-2 record's SHA-256; the build and the harness hash the record's bytes and
check that they match it and that the record is for that dataset at the source revision the
manifest pins. A candidate whose record is missing or fails those checks is not admitted and is
absent from the bundle; the bundle stays valid. The source policy asserts no speaker identity for
a candidate: its identity, and so its role, come only from that record. A third-party copy of a
source is never a candidate. A dataset outside the source policy enters only through a new
protocol version.

**Permission to train is separate from label qualification.** Which catalogued datasets may be
used to train private models is fixed by the source policy, not by this document: an entry's
`may_train`, and `may_train_unlisted` for a catalogued dataset without an entry. Every model
trained for the benchmark is still fitted, tuned and calibrated on the split ledger's train and
dev rows only (5.6). Whether a dataset's labels are benchmark ground truth is decided only by
sections 3 and 4. A model trained on any dataset whose catalogue policy is not `open` is private;
the catalogue policy of a policy entry is its `catalogue_policy`, that of any other catalogued
dataset the catalogue's.

**Training-only datasets.** An entry whose role is `training-only` is a third-party copy of a
source, not its official release, with no per-clip speaker identity (`speaker_identity_verified`
false) and no official split. A bundle may leave it out. A bundle that holds it holds it in that
role only, never rebound; under section 5.4 it is never in dev or test and never scored. Such a
bundle is invalid if `copy_revision` or `introduced_label_sample_ids_sha256` is null, or if the
entry's dataset card does not give `copy_source` as its source URL, `copy_revision` as its
revision or checksum and `copy_of` as the source it copies. The rows whose label the copy
introduced rather than the source are those whose `sample_id` is in the entry's introduced-label
list: a file holding one line for each such `sample_id` (2.1), once, the UTF-8 bytes of Python
`json.dumps(sample_id, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` followed by LF,
lines sorted by the decoded id in code-point order, with no other bytes, whose lowercase
hexadecimal SHA-256 is `introduced_label_sample_ids_sha256`.
Before the row exclusions of sections 2.1, 5.1 and 5.3, the build checks that digest, that every
listed id is a `sample_id` of the entry's rows, and that the list holds `introduced_label_rows`
ids; a failure invalidates the bundle. Of the listed rows, the otherwise eligible ones are in
train with `label: null` and reason `label_not_from_source`; an exclusion of section 2.1, 5.1 or
5.3 takes precedence, with its own reason. They are never gold labels (4.1), and a model that
uses the introduced labels as targets says so in its provenance list (section 6.1). Pinning a null
key is a policy change, and so is an official release of the source, which would be a different
dataset and a new policy entry; each is a new protocol version.

Each bundle dataset has a dataset card: source URL, revision or checksum, licence id, catalogue
policy, native sample rate, row count, the source collections and speakers it may share with
other datasets, and, for a third-party copy, the URL of the source it copies.

A training-only entry that the bundle does not hold may have the same card supplied with an
evaluation, beside the model card, for the overlap check of section 6.1. It is not a bundle input
and the manifest does not pin it. The audit hashes the card's bytes as read and the report records
that SHA-256 (section 11); the audit uses the card only if it gives `copy_source` as its source URL,
`copy_revision` as its revision or checksum and `copy_of` as the source it copies, as a bundle
card of that entry must, and otherwise treats it as absent.

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

The label of a row is read from the raw source category at the pinned source revision, never from
a label already converted by SER's catalogue (whose maps include aliases and coercions):

| Dataset | Raw category source | Raw category → label |
|---|---|---|
| `ravdess` | third `-`-separated field of the file name | `01` neutral, `02` calm, `03` happy, `04` sad, `05` angry, `06` fearful, `07` disgust, `08` surprised |
| `crema-d` | third `_`-separated field of the file name | `ANG` angry, `DIS` disgust, `FEA` fearful, `HAP` happy, `NEU` neutral, `SAD` sad |
| `mesd` | category named by the source (D4-2 records where), case-folded | `anger` angry, `disgust` disgust, `fear` fearful, `happiness` happy, `neutral` neutral, `sadness` sad |

A candidate that this table does not list (section 3) has the label set `<dataset id>-6-v1`
(angry, disgust, fearful, happy, neutral, sad); its raw category source, normalisation and complete
raw-category map, a one-to-one renaming (4.2), are those of its source-policy entry. Case-folded,
and the normalisation `casefold`, mean Python `str.casefold()` applied to the raw value as read,
with no other change, alike to the D4-2 inventory and to every row.

A label that a third-party copy of a source introduced rather than the original source is never a
gold label, in any dataset or role: its row has `label: null` and is never scored.

A raw category not in its dataset's map is excluded for that dataset (ledger reason
`label_not_scored`). Training-only rows are never scored, so they have no map and `label: null`.

### 4.2 Renaming, not mapping

Every entry of 4.1 is a one-to-one **renaming**: the source's category and the label id name the
same emotion. A category is never **mapped** onto a different one (for example boredom → neutral,
sleepy → neutral, amused → happy, anxious → fearful, neutral → calm), although SER's catalogue has
such mappings. The same rule applies to a model's output labels.

### 4.3 Scored label set of a model on a dataset

For dataset `d` with label set `S_d` and model `m` with label set `M_m`:

- the scored labels are `K = S_d ∩ M_m`, further reduced to the labels that have at least one row in
  `d`'s test partition; labels of `S_d` without a test row are listed as
  `labels_without_test_rows`;
- rows whose true label is not in `K` are not scored for this pair; their count is reported as
  `rows_label_not_in_model`;
- the returned vector `q_i` is **invalid**, and the row failed, if its length differs from
  `|M_m|`, a value is not finite or is negative, `|Σ_{b∈M_m} q_i(b) − 1| > 10^-4`, or
  `Σ_{b∈K} q_i(b) = 0`; otherwise `p_i(a) = q_i(a) / Σ_{b∈K} q_i(b)` for `a ∈ K`;
- the prediction is `ŷ_i = argmax_{a∈K} p_i(a)`; ties go to the label first in code-point order;
- `mean_out_of_set_mass` = the mean of `1 − Σ_{b∈K} q_i(b)` over the rows with a valid vector.

If `|K*_d| ≤ 1` (below) the dataset carries `degenerate_label_set`, a flag set from the bundle
alone. A model whose own `|K| ≤ 1` has that result reported as descriptive only; this never
changes `D_ℓ` or any denominator.

For selection and acceptance, the label set of dataset `d` is the same for every candidate:
`K*_d = S_d ∩ ser-8-v1`, reduced to labels with at least one test row. Because an eligible model's
label set is exactly `ser-8-v1` (section 10.5), its `K` equals `K*_d`. For a descriptive comparison
of two models with different label sets, both are rescored on `K_AB = S_d ∩ M_A ∩ M_B` (restricted
to labels with test rows), and the report declares `K_AB`.

## 5. Splits

### 5.1 Duplicate handling (before any split)

Across the whole bundle, rows with a non-null `audio_sha256` are grouped by it. For each group of
two or more rows:

- all rows have the same dataset, the same non-null label and the same speaker group: the row with
  the smallest `sample_id` (code-point order) is kept; the others are excluded, reason
  `duplicate_same_label`;
- otherwise, if the labels differ: all rows are excluded, reason `duplicate_conflicting_label`;
- otherwise: all rows are excluded, reason `duplicate_identity_conflict`.

So a training-only row whose audio equals a dev or test row removes both. The three CREMA-D pairs
of identical audio with different labels fall under the second case.

### 5.2 Identity ledger and speaker groups

Identity is fixed before any split, in a pinned identity ledger (`ser.bench.identity-ledger/v1`,
its SHA-256 in the manifest). It gives every bundle row its identity nodes, in separate namespaces
(`<dataset>:speaker:<id>`, `<dataset>:session:<id>`, `source:<id>` for one source recording);
alias records, each joining two nodes and citing the dataset card that states they are the same
person or recording; and, for every pair of datasets whose cards say they may share speakers or
source recordings, whether that pair is `resolved` (every shared node is aliased) or `unresolved`.

The speaker groups are the connected components of the graph whose edges join each row to its
nodes, plus the alias records. Every row contributes edges, including excluded rows, so an
excluded row can bridge two groups. The group id is the smallest node id of its component in
code-point order. A row without any node is in no group.

A dataset in an `unresolved` pair carries `identity_unresolved`; the audit never claims zero
shared speakers for it.

### 5.3 Assignment

Seed: `17`. Every component is assigned by the first of these rules that applies:

1. **Quarantine.** A component with a row of a training-only dataset and a row of any other
   dataset, or with rows of two different official assignments: all its rows are excluded,
   reason `identity_quarantined`.
2. **Official split.** A component with rows of a dataset whose documented official split its
   dataset card shows to be speaker-disjoint takes that split (reason
   `native_speaker_disjoint_split`).
3. **Generated split.** Any other component with a row of a train/dev/test dataset is assigned by
   the first such dataset `d`, in code-point order. Let `n` be the number of components assigned
   by `d`. If `n < 3`, all their rows are excluded, reason `too_few_speaker_groups`. Otherwise
   order those components by the lowercase hexadecimal SHA-256 of the UTF-8 string
   `"17:" + d + ":" + group_id`, ascending, and, with `R` = round half to even to an integer:

   `n_dev = n_test = max(1, R(0.2·n))`, `n_train = n − n_dev − n_test`.

   The first `n_train` components are train, the next `n_dev` dev, the rest test. For `n ≥ 3`
   every count is at least 1. This rule, and its `n < 3` exclusion, apply only to train/dev/test
   datasets, which have speaker identity; a dataset without it bypasses them (5.4).
4. **Role only.** A component with rows of test-only datasets only is test (reason
   `test_only_dataset`); one with rows of training-only datasets only is train (reason
   `training_only_dataset`).

Then a test-only row in a train or dev component is excluded (`speaker_in_training_partition`),
and a dataset left without one of its required partitions (train, dev and test; test for a
test-only dataset) has all its rows excluded (`empty_partition`). For `ravdess` (`n = 24`) the rule
gives 14/5/5 speakers; for `crema-d` (`n = 91`) 55/18/18. The seed is never changed to obtain a "better" split.

### 5.4 Rows without speaker identity

A dataset without per-clip speaker identity cannot be split, so it is used whole, in exactly one
role, never split and never dev. By default it is **test-only**: none of it is used for fitting,
tuning or calibration, and an official split it is distributed with is ignored (its official
training part is not used for training). It counts for acceptance and selection like any other
test set if it passes audit check A7 and the audio-hash checks of 5.1 and A4, and a model's
result on it also needs the separation evidence of 6.1. It is **training-only** only where the
source policy (section 3) assigns that role and its labels do not qualify for testing; it is then
never in dev or test and never scored. A row without a node is
test, reason `test_only_dataset`, in a test-only dataset; train, reason `training_only_dataset`,
in a training-only dataset; otherwise excluded, reason `no_speaker_identity`.

### 5.5 Split ledger and digest

The split ledger (`ser.bench.split-ledger/v1`) is a JSON Lines file with one object per bundle row,
including excluded rows, with exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `audio_sha256` | string or null | content hash, section 2.3; null for `decode_invalid` |
| `dataset` | string | dataset id |
| `label` | string or null | scored label id after renaming; null if not scored or not gold |
| `language` | string | BCP 47 tag (`en`, `es`) |
| `reason` | string | one of `assigned`, `decode_invalid`, `duplicate_conflicting_label`, `duplicate_identity_conflict`, `duplicate_same_label`, `empty_partition`, `identity_quarantined`, `label_not_from_source`, `label_not_scored`, `native_speaker_disjoint_split`, `no_speaker_identity`, `speaker_in_training_partition`, `test_only_dataset`, `too_few_speaker_groups`, `training_only_dataset` |
| `sample_id` | string | section 2.1 |
| `speaker_group` | string or null | group id of 5.2; null if the row has no node |
| `split` | string | `train`, `dev`, `test` or `excluded` |

The first applicable reason is recorded, in the order of sections 2.1, 3, 4.1, 5.1, 5.3, 5.4,
except that an exclusion reason precedes `label_not_from_source`. Each line is the UTF-8 bytes of
Python
`json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` followed by LF,
lines sorted by `(dataset, sample_id)` in code-point order. The **ledger digest** is the lowercase
hexadecimal SHA-256 of the whole file. Rebuilding the bundle from the same sources must give the
same digest.

### 5.6 Relation to SER's training recipe

This ledger is mandatory for fitting, tuning and calibrating every model trained for the
benchmark: they use its train and dev rows only. Already-trained external candidates fall under
the provenance rule of section 6.1; there is no other ledger route. SER's dataset recipe
(`ser/_internal/data/dataset_audit.py`, `docs/dataset-recipes.md`, 70/15/15) stays the default for
other training.

## 6. Leakage audit

The audit reads the manifest, the source policy (section 3), both ledgers and, in a benchmark run,
the model card (6.1) and any dataset card supplied with it (section 3). It runs
before every benchmark run and on its own. Checks, over all datasets together:

| Check | Pass condition |
|---|---|
| A1 ledger digests | the recomputed split and identity ledger digests equal those recorded in the manifest |
| A2 coverage | every manifest row appears exactly once in the split ledger and vice versa |
| A3 shared speakers | 0 speaker groups appear in more than one of train, dev, test |
| A4 shared audio | 0 non-null `audio_sha256` values appear in more than one of train, dev, test |
| A5 duplicate audio | 0 non-null `audio_sha256` values appear twice among non-excluded rows |
| A6 model data | section 6.1 |
| A7 test-only sources | for each test-only dataset without speaker identity (5.4), no dataset with train rows shares a source collection with it, according to the dataset cards |

A1 to A5 failing stops the run. A6 never stops the run; it sets the model's provenance level and
flags. A7 never stops the run; a dataset failing it carries `source_in_training`.

### 6.1 Model provenance

Each model card records one provenance level, covering every stage of fitting, tuning and
calibration, including a base model's:

- **`verified`**: an artifact-bound list (its SHA-256 recorded in the artifact's metadata when it
  is trained) of every row used in any of those stages, each a bundle row given by `sample_id`
  and present in the identity ledger. If any such row is outside the bundle, the model is at best
  `declared`;
- **`declared`**: the publisher's documented training-data list, quoted in the card with its URL.
  Results carry `provenance_declared`;
- **`unknown`**: anything else. Results carry `provenance_unknown`.

Whatever the level, every known row of every stage is checked: a bundle row, or an external file
whose `audio_sha256` (2.3) the card lists. Demotion to `declared` never drops these checks on the
part that can be audited. The flag `test_overlap` is set on a dataset's results when a known row is
in that dataset's test partition, shares its speaker group with a test row, or has the
`audio_sha256` of a test row, or when the card or publisher list names that dataset, its source
collection, or a speaker or recording of its test partition. A result on a test-only dataset without
speaker identity (5.4) also needs auditable separation: every stage's data is listed by source
collection, none is that dataset's, and every training row is a known row; otherwise it carries
`separation_unverified`. The flag `overlap_unverified` is set on a test dataset's results when the
model card or publisher list shows training, at any stage, on a training-only entry of the source
policy (section 3), known as the copy at its `copy_source` of its `copy_of`, at any revision,
whatever the evaluated bundle holds, and that entry's dataset card cannot rule out source recordings
or speakers shared with that dataset. That card is the bundle's card when the bundle holds the
entry, and otherwise the card supplied with the evaluation (section 3); where neither exists, the
flag is set. These flags remove the result from clean comparison; they are flags, not provenance
levels.

`verified` and `declared` models may enter confirmatory selection; `unknown` ones cannot. The
owner's own recordings (E3-4) are free of overlap by construction; for a `declared` model they are
the check that matters most. The legacy `fast` artifact was trained on all 24 RAVDESS actors, so
its results on `ravdess` carry `test_overlap`.

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
model_provenance=<verified|declared|unknown|not_checked> test_overlap_datasets=<n|null>
result=<PASS|FAIL>
```

`not_checked` appears only when the audit runs on its own without a model card, and then
`test_overlap_datasets` is null; a benchmark run without a model card is refused.

## 7. Scoring a row

For one model on one dataset's test partition: rows `i = 1 … N` (rows with a true label in `K`),
`k = |K|`, true label `y_i`, prediction `ŷ_i` (section 4.3) or `∅` for a failed row, confidence
`c_i = max_{a∈K} p_i(a)` as an exact rational (section 11). `N'` is the number of rows that are not
failed; `coverage = N'/N`.

A failed row counts in the recall denominator of its true label, is never correct, and appears in
the confusion matrix in a final column `no_prediction`. It has no confidence and is left out of the
calibration error, so ECE is conditional on success: a model that fails on hard rows can look
better calibrated. ECE is a diagnostic, and acceptance requires `n_failed = 0` (section 10.4).

Any calibrator, threshold or hyperparameter is fitted on train or dev only and frozen before the
model sees any test row; the model card identifies the calibrator (SHA-256) and the tuning rows.

## 8. Metrics

### 8.1 Per dataset

All metrics are computed in exact rational arithmetic from integer counts and the exact
confidences of section 7. Point values and interval bounds are rounded once, half to even, to 6
decimal places, only when written.

- **Confusion matrix** `C[a][b]` = number of rows with `y_i = a` and `ŷ_i = b`, for `a ∈ K` (rows)
  and `b ∈ K ∪ {no_prediction}` (columns), both in code-point order with `no_prediction` last.
- **Support** `n_a = Σ_b C[a][b]`: every row whose true label is `a`, failed or not.
- **Per-class recall** `R_a = C[a][a] / n_a`.
- **UAR** (unweighted average recall, the primary metric) `= (1/k) Σ_{a∈K} R_a`.
- **Per-class precision** `P_a = C[a][a] / Σ_{x∈K} C[x][a]`; `P_a = 0` if that column sum is 0.
- **Per-class F1** `F_a = 2·P_a·R_a / (P_a + R_a)`; `F_a = 0` if `P_a + R_a = 0`.
- **Macro-F1** `= (1/k) Σ_{a∈K} F_a`.
- **Accuracy** `= Σ_a C[a][a] / N` (reported for continuity; never used in a decision).
- **Expected calibration error** (top-label, 10 equal-width bins): row `i` is in bin
  `m_i = min(10, ⌊10·c_i⌋ + 1)`, so bin `m` holds `[(m−1)/10, m/10)` and bin 10 holds `[0.9, 1.0]`;
  `ECE = Σ_m (|B_m| / N') · |acc_m − conf_m|`, with `acc_m` the fraction of bin `m`'s rows with
  `ŷ_i = y_i` and `conf_m` the mean of their `c_i`. Empty bins contribute 0 and are reported with
  count 0 and null accuracy and confidence.

Labels in `S_d` but absent from the test partition are not part of `K` and do not enter UAR as
zero. SER's `compute_ser_metrics` (`ser/_internal/train/metrics.py`) is not used: it scores an
absent label as 0.0 and has no failure column. Without failures and with all labels present, these
definitions equal scikit-learn's macro recall and macro F1 with `zero_division=0` over `K`.

### 8.2 What is reported per dataset

UAR, macro-F1, accuracy, per-class recall, precision and F1, the confusion matrix, ECE with its
per-bin table, `n_rows`, `n_speakers`, `n_failed`, `coverage`, `rows_label_not_in_model`,
`mean_out_of_set_mass`, and confidence intervals for UAR, macro-F1 and ECE. `n_speakers` is the
number of test speaker groups, null for a dataset without speaker identity. A dataset with fewer
than 5 test resampling units (8.3) is flagged `underpowered`; its intervals are reported but are
not reliable.

### 8.3 Bootstrap confidence intervals

- **Unit: the speaker group**, because rows of one speaker are correlated and the claim is about
  new speakers. The units of dataset `d` are its `S` test speaker groups in code-point order of
  group id, fixed by the ledger, whatever their labels. For a dataset without speaker identity
  (5.4) the unit is the row, its `S` test rows in `sample_id` order, and its intervals are flagged
  `ci_utterance_unit`: such an interval is narrower than a speaker-level one would be, because
  rows of one speaker are correlated. `B = 10,000` resamples.
- **Random draws:** for each dataset, a generator `numpy.random.Generator(numpy.random.PCG64(s_d))`
  with `s_d` = the first 8 bytes, read as a big-endian unsigned integer, of SHA-256 of the UTF-8
  string `"17:bootstrap:" + d`. For `b = 0 … B−1`:
  `idx_b = rng.integers(0, S, size=S, dtype=numpy.int64)`; resample `b` is the multiset of all rows
  of units `idx_b`, with repetition. The report records the numpy version. The draws depend only on
  the dataset, never on the model, so every model and baseline sees the same resamples (paired).
- **Metric on a resample:** the formulas of 8.1, over the resample's rows whose label is in the
  scored set. Predictions, confidences and the full prediction columns stay those of the
  comparison's original `K`; only true classes with no row in the draw are left out of the macro
  averages (and `k`). No prediction is removed from a denominator or re-decided on fewer labels.
  Check: two true `angry` rows predicted `angry` and `happy`, no true `happy` row: UAR `= 1/2`.
  A resample with no such row, or (for ECE) no successful row, has no value.
- **Interval:** if any resample has no value, the interval is null and flagged `ci_undefined`.
  Otherwise sort the `B` exact values `v_0 ≤ … ≤ v_{B−1}`; for `q` the lower and upper tail
  levels, `h = q·(B−1)`, value `= v_⌊h⌋ + (h − ⌊h⌋)·(v_⌈h⌉ − v_⌊h⌋)` (the "linear" percentile).
  Descriptive intervals use `q ∈ {0.025, 0.975}`; with `B = 10,000`, `h = 249.975` and `9749.025`.
  Familywise-adjusted intervals are defined in 10.2.

### 8.4 Per language, never pooled

For benchmark language `ℓ`, the **required datasets** `D_ℓ` are its datasets with a train/dev/test
or test-only role in the bundle that carry no dataset flag of section 1 (each set from the bundle
alone: 5.2, 4.3 on `K*_d`, A7); model flags never change it. `D_ℓ` depends only on
the bundle, is the same for every model and baseline, and is frozen in the selection plan (10.5).
Each dataset is weighted equally whatever its size:

- `UAR_ℓ = mean_{d∈D_ℓ} UAR_d`, likewise `MacroF1_ℓ`, `ECE_ℓ`;
- `chance_ℓ = mean_{d∈D_ℓ} 1/k_d`;
- `n_rows` and `n_failed` are summed over `D_ℓ`, and `coverage = Σ N' / Σ N`;
- its interval uses, for each `b`, the mean over `d ∈ D_ℓ` of the resample-`b` values (datasets are
  resampled independently, stratified by dataset). If a speaker group is in the test partition of
  two datasets of `D_ℓ`, independent draws misstate the uncertainty: the language intervals are
  null and flagged `ci_unavailable_shared_groups`. If `D_ℓ` mixes speaker-level and
  utterance-level units, the language intervals are null and flagged `ci_mixed_units`. Point
  values are unaffected, so acceptance (10.4) is too; comparisons in that language are made per
  dataset (10.2).

Rows of different datasets are never merged into one confusion matrix or one metric. No figure
across languages is computed.

### 8.5 Null policy

| Value | Null when | Never null because of |
|---|---|---|
| every metric, count-derived ratio and interval of a dataset | the dataset has no eligible row for the model (`N = 0`, including `|K| = 0`) | failed rows: they are counted, lower recall and block acceptance |
| ECE and its interval | `N' = 0` | — |
| an interval | some resample has no value (`ci_undefined`), shared groups (`ci_unavailable_shared_groups`), or mixed units (`ci_mixed_units`) | — |
| a language value | `D_ℓ` is empty or any dataset value it averages is null | — |
| a field the reference does not define | always: `chance` is analytic (its values are expectations under uniform guessing), so its confusion matrix, ECE, bins, intervals, `n_failed` and `coverage` are null | — |

A null value fails every threshold and every comparison it enters.

## 9. Baselines

Every report includes these rows for every scored dataset, on the same test partition and scored
label set `K` as the model:

| Baseline | Definition | Values |
|---|---|---|
| `chance` | uniform random guessing over `K` | `UAR = 1/k`; **macro-F1 of the expected confusion matrix** `E[C[a][b]] = n_a/k`, `= (1/k) Σ_a 2·π_a·(1/k) / (π_a + 1/k)` with `π_a = n_a/N` (not the expected macro-F1 of random predictions); accuracy `= Σ_a π_a·(1/k) = 1/k`; per-class recall `1/k`, precision `π_a`, F1 as above |
| `majority` | always predicts `m`, the label of `K` with the most rows in the dataset's train partition (for a test-only dataset, in its test partition, flagged `majority_from_test`: an oracle, not a deployable baseline); ties to the first label in code-point order; probability 1 on `m` | `UAR = 1/k`; `MacroF1 = (1/k)·2·π_m/(π_m + 1)`; accuracy `= π_m`; `ECE = 1 − π_m`; all fields of 8.2 filled, intervals by 8.3 |
| `fast@16k` | the legacy `fast` artifact through the public call with the input of section 2.2 | measured |
| `fast@native` | the same artifact fed at each file's own sample rate | measured, descriptive only |

**The U11 comparator.** `fast@16k` is the one explicit exception to the descriptive-only flags:
its artifact SHA-256, call options, code revision, `D_ℓ` and `K*_d` are frozen in the selection
plan before any score is computed, its own included; it is scored on exactly `D_ℓ` and `K*_d` of
each language; and its `UAR_ℓ` is used only in U11 condition 2 (section 10.4). It is never a
selection candidate.

`fast@native` runs at the same code revision as `fast@16k`. If the public call there cannot take a
native rate, it runs at the last commit whose inference path did and carries
`confounded_code_revision`: its difference from `fast@16k` is not attributed to the sample rate.

## 10. Decision rule

### 10.1 Primary metric

Decisions use UAR only. Macro-F1, ECE and per-class figures never decide which model is better;
ECE enters only acceptance (10.4).

### 10.2 "Model A beats model B"

**Confirmatory comparisons** are made only between candidates of a frozen selection plan (10.5),
on `D_ℓ` and `K*_d`, with the same resamples (8.3). For each resample `b`:
`Δ_b = UAR_A,b − UAR_B,b` per dataset, and `Δ_ℓ,b` = mean of the dataset `Δ_b` over `D_ℓ`. The
**required contrasts** are `Δ_ℓ` for each benchmark language, except that a language whose `D_ℓ`
mixes units (8.4) contributes the dataset `Δ` of each `d ∈ D_ℓ` instead. Intervals are
familywise-adjusted by Bonferroni: with `M` = the number of candidate pairs times the number of
required contrasts, fixed in the plan, the tail levels are `q = 0.025/M` and `1 − 0.025/M` (8.3).
A beats B if every required contrast's adjusted interval is non-null and its upper end is not
below 0, and at least one's lower end is above 0. Otherwise A does not beat B.

**Descriptive comparisons** use unadjusted 95% intervals, on `K*_d` or on `K_AB`; they are labelled
descriptive and decide nothing.

### 10.3 Cost order

When candidates must be ordered by cost, they are compared in this order, on numbers measured on
the same machine class by the CPU performance harness (workstream WS7) and frozen in a cost
evidence file: lower 95th percentile processing time per second of audio at 2 threads; then lower
peak memory; then smaller model file; then smaller `artifact_sha256` in code-point order.

### 10.4 Acceptance thresholds (checklist U11)

A model is acceptable only if its label set is exactly `ser-8-v1` and, for every benchmark
language `ℓ` (`en` and `es`), on `D_ℓ` and `K*_d`:

1. `UAR_ℓ ≥ 2 · chance_ℓ`;
2. `UAR_ℓ ≥ UAR_ℓ(fast@16k) + 0.10`, with the comparator of section 9;
3. `ECE_ℓ ≤ 0.10`;
4. `n_failed = 0` on every dataset of `D_ℓ`;
5. a check on at least 100 of the owner's own labelled turns in `ℓ` is recorded from the E3-4 kit,

and the E3-4 check on the owner's Portuguese recordings is recorded (status and turn count only).
Conditions 1 to 3 are evaluated on point estimates, each side rounded half to even to 6 decimal
places, then compared exactly; a null fails. Intervals are reported beside them but do not decide.

### 10.5 Selection

1. Before any score is computed, a selection plan (`ser.bench.selection-plan/v1`) is committed. It
   lists the candidate artifacts (SHA-256) with their provenance level, `D_ℓ` and `K*_d` for each
   benchmark language, the comparator of section 9, the thresholds and precision of 10.4, `M`, and
   the cost evidence file's SHA-256. Every report used for selection records the plan's SHA-256.
2. A candidate is eligible if its label set is exactly `ser-8-v1`, its provenance is `verified`
   or `declared`, none of its results on `D_ℓ` carries a model flag of section 1, and it passes
   10.4.
3. If no candidate is eligible, nothing is selected. No default is presumed.
4. Otherwise the selected candidate is the eligible one that beats (10.2) the most other eligible
   candidates; ties go by the cost order of 10.3.

## 11. Report requirements

A report contains: the identity of the protocol (this document's SHA-256), the model (artifact
SHA-256, label set, provenance, distribution), the bundle (id, manifest SHA-256, contract ids), the
split and identity ledger digests, and the code version (commit, clean tree, SER and numpy
versions); for each configuration scored (model, `fast@16k`, `fast@native`), the call options,
timeout, code revision and sample rates actually fed; every value of sections 8.2 and 8.4 with its
interval and flags; the baselines of section 9; the audit output of 6.2 verbatim, and the SHA-256 of
any dataset card supplied with the evaluation (section 3); the acceptance result of 10.4 by
condition, with the E3-4 status; and, for every comparison, its basis, its cohort of datasets, its
label set (`K*_d` or `K_AB`), and the SHA-256 of every input it reads, including the other
candidate's predictions and card and the selection plan.

Two runs on the same inputs produce byte-identical files, with no wall-clock time, host name, user
name, absolute path or duration. Probabilities are archived losslessly: each returned value is
written as the shortest decimal string that converts back to the same IEEE 754 double (Python
`repr(float(x))`), and scoring reads these strings as exact rationals, so the archive reproduces
every argmax, confidence and bin. A vector with a non-finite value is invalid (4.3) and archived as
a failed row, not parsed. Predictions of the owner's own recordings are never written here.

**Deferred to the harness.** The JSON schema of the report, predictions and selection files, the
literal shape of every null, the Markdown rendering, the replay manifest that binds all inputs,
and the remaining byte-level serialisation rules are defined by the benchmark harness and pinned by
its tests (brief E3-2). They implement this section and section 8.5 and may not change a number.

## 12. Worked toy example

A toy dataset `toy` (language `und`, role test-only) of 20 rows: 4 speakers (S1 to S4), label set
`toy-3-v1` = angry, happy, neutral, also the model's, so `K` = {angry, happy, neutral}, `k = 3`, and
no renormalisation changes any vector. Row `toy-s1-04` failed (status `raised`). The toy departs
from the contract only in this: `sample_id` values do not follow the grammar of 2.1,
`audio_sha256` values are stand-ins (`a01` … `a20`), group ids are `S1` … `S4` rather than node ids
(5.2), dataset, language and label set are synthetic, and intervals are not computed (12.5).

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

Row 4 has no confidence and is left out, so the weights are over `N' = 19`.

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
  ECE = |0.4 − 1.0| = **0.600000**; it predicts every row, so its coverage is 20/20 = 1.000000.

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

`n_dev = n_test = max(1, R(0.8)) = 1`, `n_train = 4 − 1 − 1 = 2`: train = S4, S1; dev = S3;
test = S2.

## 13. Limitations of version 1

Only exact-content duplicates are detected, not re-encoded, trimmed or re-amplified copies or
shared prompt texts. Acted corpora dominate; no conversational test set exists yet. If D4-2 finds
no speaker identity in any Spanish dataset admitted for scoring, Spanish is tested only on test-only
datasets with utterance-level intervals, which overstate precision (8.3), and their rows are
withheld from fitting. `declared` provenance cannot be checked row by row. Intervals depend on
numpy's PCG64 stream, hence the recorded numpy version. A wall-clock timeout can fail a row on a
slow machine.

## 14. Relation to the existing quality gate

The profile quality gate (`ser/_internal/runtime/profile_quality_gate.py`) is not reused: it scores
one already trained artifact on every grouped fold without checking whether it was trained on a
test speaker (`quality_gate_evaluation.py`, `for _, test_indices in folds`), reports only pooled
metrics, has no calibration or interval, and stamps wall-clock time.

## 15. Pending owner decisions

None. The decisions that govern sources are recorded in the source policy (section 3) and in the
project's private record; the rules that apply them are in sections 1, 3, 4.1 and 10.
