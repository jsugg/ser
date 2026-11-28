# Repository External Asset Inventory

This document is the checkable inventory of external runtime assets (datasets, manifests, trained
model artifacts, and cache layout) required by a working SER data directory. Use it to verify a
restore or a fresh dataset acquisition on any machine.

## Directory layout (platform-neutral)

SER resolves its default data and cache roots from platform conventions in
`ser/_internal/config/schema.py` (`_platform_data_base_dir`, `_platform_cache_base_dir`,
`_default_data_root`, `_default_cache_root`) and supports environment-variable overrides in
`ser/_internal/config/settings_inputs.py`:

| Logical root | Env override | Linux default | macOS default | Windows default |
| --- | --- | --- | --- | --- |
| `<SER_DATA_DIR>` | `SER_DATA_DIR` | `${XDG_DATA_HOME:-~/.local/share}/ser` | `~/Library/Application Support/ser` | `%APPDATA%\ser` |
| `<SER_CACHE_DIR>` | `SER_CACHE_DIR` | `${XDG_CACHE_HOME:-~/.cache}/ser` | `~/Library/Caches/ser` | `%LOCALAPPDATA%\ser` |

Subdirectories resolved relative to those roots:

- `<SER_DATA_DIR>/datasets/<dataset_id>/` — downloaded dataset trees
  (`ser/_internal/data/dataset_prepare.py:default_dataset_root`)
- `<SER_DATA_DIR>/manifests/<dataset_id>.jsonl` — prepared utterance manifests
  (`ser/_internal/data/dataset_prepare.py:default_manifest_path`)
- `<SER_DATA_DIR>/models/` — persisted model artifacts and training reports
  (override: `SER_MODELS_DIR`)
- `<SER_DATA_DIR>/transcripts/` — transcript exports (override: `SER_TRANSCRIPTS_DIR`)
- `<SER_DATA_DIR>/.ser/dataset_registry.json` — registered dataset roots and manifest pointers
  (`ser/_internal/data/dataset_registry.py:_registry_path`)
- `<SER_DATA_DIR>/.ser/dataset_consents.json` — persisted dataset policy/licence consents
  (override: `SER_DATASET_CONSENTS_FILE`; `ser/_internal/data/dataset_consents.py:_consents_path`)
- `<SER_DATA_DIR>/restricted_backend_consents.json` — persisted restricted-backend consents
  (override: `SER_RESTRICTED_BACKENDS_CONSENT_FILE`;
  `ser/_internal/license_check.py:_consent_store_path`)
- `<SER_CACHE_DIR>/model-cache/` — downloaded foundation-model weights
  (override: `SER_MODEL_CACHE_DIR`)
- `<SER_CACHE_DIR>/tmp/` — temporary readiness reports, quarantine ledgers, and embedding caches
  (override: `SER_TMP_DIR`)

### Observed cache directory contents

All items under `<SER_CACHE_DIR>` are regenerable caches or dry-run artifacts:

| Relative path under `<SER_CACHE_DIR>` | Regular files | Total bytes | Notes |
| --- | ---: | ---: | --- |
| `model-cache/OpenAI/whisper/models--Systran--faster-distil-whisper-large-v3` | 6 | 1,516,479,668 | Cached Whisper transcription weights |
| `model-cache/huggingface/models--facebook--wav2vec2-xls-r-300m` | 8 | 1,269,617,260 | Cached XLS-R 300M encoder weights |
| `tmp/medium_embeddings/hf_xlsr/facebook__wav2vec2-xls-r-300m` | 5 | 5,825,370 | Cached medium-profile `.npz` embeddings |
| `tmp/training-readiness-accurate.json` | 1 | 7,230,972 | Dry-run readiness report (`accurate` profile) |
| `tmp/training-readiness-medium.json` | 1 | 7,230,970 | Dry-run readiness report (`medium` profile) |
| **Total `<SER_CACHE_DIR>`** | **21** | **2,806,384,240** | |

## Datasets

### Tree digest definition

Each dataset's tree digest is the SHA-256 of a UTF-8, LF-terminated TSV listing every regular file
under `<SER_DATA_DIR>/datasets/<dataset_id>/`:

1. Each line has three tab-separated fields: `<sha256_hex>\t<size_bytes>\t<relative_posix_path>`.
2. `<relative_posix_path>` is relative to the dataset root, uses `/` separators, and has no leading
   `./`.
3. Lines are sorted by `<relative_posix_path>` in byte-lexicographic (`LC_ALL=C`) order.
4. The tree digest is the SHA-256 of the resulting byte stream.

Note: `ser/_internal/data/catalog/public_datasets.py` declares the CSV-labelled public dataset
catalogue; the three datasets below are registered in `ser/_internal/data/dataset_prepare.py`
(`SUPPORTED_DATASETS`) and their dedicated adapters (`ser/_internal/data/adapters/{ravdess,crema_d,msp_podcast}.py`).

### Summary

| Dataset | Licence ID | Policy ID | Official source URL | Pinned revision / record | Total files | Audio WAVs | Total bytes | Manifest rows | Tree digest (SHA-256) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `ravdess` | `cc-by-nc-sa-4.0` | `noncommercial` | `https://zenodo.org/records/1188976` | Zenodo record `1188976` (`Audio_Speech_Actors_01-24.zip`) | 1,443 | 1,440 | 798,818,035 | 1,440 | `b544b5363acedc8cfe2ae15f731ad64131a2b9c5b27de7333e103b4c96cd1d19` |
| `crema-d` | `odbl-1.0` | `share_alike` | `https://github.com/CheyneyComputerScience/CREMA-D` | Git commit `1658cd342dff90010aa843eaeebd53610a08b1dc` | 44,682 | 7,442 | 7,435,537,796 | 7,442 | `36acae23c77cc1382d1560425b20a495898e96979a920fd6f21f67e6f70b9530` |
| `msp-podcast` (mirror) | `msp-academic-license` | `academic_only` | `https://lab-msp.com/MSP/MSP-Podcast.html` (mirror: `https://huggingface.co/datasets/AbstractTTS/PODCAST`) | HF `AbstractTTS/PODCAST` rev `main`, commit `e791939e368b6c26595ca7f2c75aa29ebd92e776` | 149,451 | 149,307 | 54,218,867,341 | 142,924 | `f01716b63851efe58bdffde9b8dd4fca424ab952aca4c9e92ccdee0e828645ec` |

### RAVDESS (`ravdess`)

- **Source:** Zenodo record `1188976` (`https://zenodo.org/records/1188976`), archive
  `downloads/Audio_Speech_Actors_01-24.zip` (208,468,073 bytes, SHA-256
  `5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657`).
- **Catalogue licence / policy:** `cc-by-nc-sa-4.0` / `noncommercial`.
- **Files and bytes:** 1,443 files, 798,818,035 bytes total:
  - `Actor_01/` .. `Actor_24/`: 24 directories × 60 `.wav` files = 1,440 audio files.
  - `downloads/Audio_Speech_Actors_01-24.zip`: 1 archive file (208,468,073 bytes).
  - Root metadata files: `source_manifest.json` (441 bytes) and
    `.extract-ok-Audio_Speech_Actors_01-24.zip.json` (191 bytes).
- **Manifest comparison:** `manifests/ravdess.jsonl` has **1,440** rows (528,192 bytes, SHA-256
  `c71bcf21d2fcacba9091ff6b717476cbf79e7036360fcfb3f83fb80cf48cc494`), matching the 1,440 WAV files
  1:1.

### CREMA-D (`crema-d`)

- **Source:** `https://github.com/CheyneyComputerScience/CREMA-D` (Git LFS repository).
- **Pinned commit:** `1658cd342dff90010aa843eaeebd53610a08b1dc` (2025-03-21).
- **Catalogue licence / policy:** `odbl-1.0` / `share_alike`.
- **Counts per top-level folder:**

| Top-level entry | Files | Total bytes |
| --- | ---: | ---: |
| `.git` | 22,340 | 3,690,385,709 |
| `AudioMP3` | 7,442 | 625,064,629 |
| `AudioWAV` | 7,442 | 605,899,936 |
| `VideoFlash` | 7,442 | 2,443,021,141 |
| `docs` | 2 | 11,160 |
| `processedResults` | 2 | 4,507,001 |
| Root files (`.gitattributes`, `LICENSE.txt`, `README.md`, CSVs, R scripts) | 12 | 66,648,220 |
| **Total** | **44,682** | **7,435,537,796** |

- **Manifest comparison:** `manifests/crema-d.jsonl` has **7,442** rows (2,721,046 bytes, SHA-256
  `865c47c197898ccdb5b1d251b7ee8bd60fcfe1749bf5f18c7b3dbce963df9acd`), matching the 7,442 WAV files
  in `AudioWAV/` 1:1.

### MSP-Podcast mirror (`msp-podcast`)

- **Official source:** `https://lab-msp.com/MSP/MSP-Podcast.html`
  (`ser/_internal/data/adapters/msp_podcast.py`).
- **Actual downloaded copy (third-party mirror):** Hugging Face dataset `AbstractTTS/PODCAST`
  (`https://huggingface.co/datasets/AbstractTTS/PODCAST`), revision `main`, commit
  `e791939e368b6c26595ca7f2c75aa29ebd92e776` (70 remote files). This is a **third-party copy**,
  not the official MSP-Podcast release from UT Dallas.
- **SER policy on use:**
  - Licence / policy in SER's catalogue: `msp-academic-license` / `academic_only`
    (`ser/_internal/data/dataset_prepare.py`, `ser/_internal/data/adapters/msp_podcast.py`).
    Training requires explicit persisted consent (`ser/_internal/data/dataset_consents.py`).
  - Under SER's strict dataset recipe and audit policy (`docs/dataset-recipes.md`,
    `ser/_internal/data/dataset_audit.py`), rows without a defensible speaker or session group are
    routed to `ssl_only` and cannot enter supervised validation or test splits. Because the
    third-party mirror omits speaker and official split columns (`labels.csv` contains only
    `FileName` and `emotion`), its utterances are not eligible for supervised validation or test
    under strict auditing.
- **Files and bytes:** 149,451 files, 54,218,867,341 bytes total:
  - `audio/`: 149,307 extracted `.wav` files (27,187,404,694 bytes).
  - `repo/`: 141 snapshot files including parquet shards (26,863,699,805 bytes).
  - Root files: `labels.csv` (5,192,906 bytes), `manifest.json` (875 bytes), `metadata.jsonl`
    (162,569,061 bytes).
- **Manifest comparison (149,307 audio files vs 142,924 manifest rows):**
  `manifests/msp-podcast.jsonl` has **142,924** rows (50,502,646 bytes, SHA-256
  `02b9fbbb7a2c8c476d2dfcaf5cc9d5916c9c964fe30ee6c73919450ee6f31c24`). `manifest.json` records
  149,307 `rows_seen`, 149,307 `files_written`, and 149,307 `labels_written` in `labels.csv`. The
  difference of **6,383** rows is accounted for by the `contempt` class in `labels.csv`: SER's
  default 8-class canonical ontology (`angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`,
  `sad`, `surprised`) excludes `contempt`, and `remap_label` (`ser/_internal/data/ontology.py`)
  drops unknown labels under its default `unknown_label_policy="drop"`. In `labels.csv`:
  `neutral` 48,748 + `happy` 43,498 + `sad` 20,196 + `angry` 17,572 + `surprised` 7,127 +
  `disgust` 4,313 + `fearful` 1,470 = **142,924** (retained) + `contempt` **6,383** (dropped) =
  **149,307**.

## Trained model artifacts

All files live under `<SER_DATA_DIR>/models/`.

> **Security warning:** `ser_model.pkl` (and its `.bak` backup) is a Python pickle file serialized
> via `pickle.dumps` (`ser/_internal/models/artifact_persistence.py:persist_pickle_artifact`).
> Unpickling executes arbitrary code; **load these files only from a trusted source and after
> verifying their SHA-256 digest**. Pickle artifacts are also sensitive to scikit-learn version
> upgrades (the `.20261110-033919.bak` file is the pre-resave backup of `ser_model.pkl`).

| File | Bytes | SHA-256 | Role |
| --- | ---: | --- | --- |
| `ser_model.pkl` | 1,954,441 | `4e96e1e12b593e82dffe782511f07c36386357007b03203f1b8d43d751b36178` | Active `fast` profile classifier pickle (schema v2) |
| `ser_model.pkl.20261110-033919.bak` | 1,954,332 | `c0b8891ca38c0348999a7598d62324308b27e37bbd2d22fed273330e7ec51114` | Pre-resave backup of `ser_model.pkl` |
| `training_report.json` | 4,302 | `3bb724be632fde420f9e1f00abda356ff00822f612a779b0709208bbd8261c97` | Training metrics and provenance for `ser_model.pkl` |
| `training_report.json.20261110-033919.bak` | 4,302 | `3bb724be632fde420f9e1f00abda356ff00822f612a779b0709208bbd8261c97` | Identical backup copy of `training_report.json` |

### Model metadata and reported metrics (`training_report.json`)

- **Profile / backend:** `fast` / `handcrafted` (193-dim feature vector, 3.0 s frame size, 1.0 s
  stride, `mean` pooling; artifact schema `v2`; backend licence `ISC`).
- **Training data:** RAVDESS (`datasets/ravdess/Actor_*/*.wav`), 1,440 corpus samples (1,440
  effective, 0 skipped): **1,080** train samples, **360** test samples.
- **Label distribution (1,440 total):** 192 each for `angry`, `calm`, `disgust`, `fearful`,
  `happy`, `sad`, `surprised`; 96 for `neutral`.
- **Reported test metrics:**
  - Accuracy: `0.5861` (`58.61%`)
  - UAR (unweighted average recall): `0.5677` (`56.77%`)
  - Macro-F1: `0.5674` (`56.74%`)
  - Per-class recall: `angry` `0.7708`, `disgust` `0.6667`, `calm` `0.6458`, `happy` `0.6250`,
    `surprised` `0.5833`, `fearful` `0.5625`, `sad` `0.3958`, `neutral` `0.2917`.

## How to check a restore

Set `SER_DATA_DIR` to the restored data directory (or use the platform default above), then run the
following checks.

### 1. Verify manifest line counts

```bash
wc -l "$SER_DATA_DIR/manifests/ravdess.jsonl" \
      "$SER_DATA_DIR/manifests/crema-d.jsonl" \
      "$SER_DATA_DIR/manifests/msp-podcast.jsonl"
```

Expected counts: `1440` (`ravdess.jsonl`), `7442` (`crema-d.jsonl`), `142924` (`msp-podcast.jsonl`),
`151806` total.

### 2. Verify model and manifest SHA-256 digests

Use `shasum -a 256` (macOS/Linux) or `sha256sum` (Linux):

```bash
shasum -a 256 \
  "$SER_DATA_DIR/models/ser_model.pkl" \
  "$SER_DATA_DIR/models/ser_model.pkl.20261110-033919.bak" \
  "$SER_DATA_DIR/models/training_report.json" \
  "$SER_DATA_DIR/manifests/ravdess.jsonl" \
  "$SER_DATA_DIR/manifests/crema-d.jsonl" \
  "$SER_DATA_DIR/manifests/msp-podcast.jsonl"
```

Compare each hash against the tables above. Note that manifest JSONL files embed absolute audio
paths for the machine on which they were generated; if manifests are regenerated on a new machine,
their SHA-256 will differ while their line counts must still equal `1440`, `7442`, and `142924`.

### 3. Verify CREMA-D pinned Git commit

```bash
git -C "$SER_DATA_DIR/datasets/crema-d" rev-parse HEAD
```

Expected output: `1658cd342dff90010aa843eaeebd53610a08b1dc`.

### 4. Recompute a dataset's file count, total bytes, and tree digest

This Python one-liner reproduces the exact TSV specification (SHA-256, byte size, relative POSIX
path sorted in UTF-8 byte order, LF-terminated) and prints the file count, total bytes, and tree
digest without creating temporary files:

```bash
python3 - "$SER_DATA_DIR/datasets/<dataset_id>" <<'PY'
import hashlib, os, pathlib, sys

root = pathlib.Path(sys.argv[1])
rels = []
for dirpath, _, filenames in os.walk(root):
    for fn in filenames:
        p = pathlib.Path(dirpath) / fn
        if p.is_file() and not p.is_symlink():
            rels.append(p.relative_to(root).as_posix())
rels.sort(key=lambda s: s.encode("utf-8"))

tree_hash = hashlib.sha256()
total_bytes = 0
for rel in rels:
    p = root / rel
    size = p.stat().st_size
    total_bytes += size
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    tree_hash.update(f"{h.hexdigest()}\t{size}\t{rel}\n".encode("utf-8"))

print(f"files={len(rels)} bytes={total_bytes} tree_sha256={tree_hash.hexdigest()}")
PY
```

Replace `<dataset_id>` with `ravdess`, `crema-d`, or `msp-podcast` and compare `files`, `bytes`,
and `tree_sha256` against the dataset summary table. Note that `source_manifest.json` (RAVDESS) and
`manifest.json` (MSP-Podcast) record machine-specific paths and generation timestamps, so a fresh
download on another machine will match the audio WAV counts (`1,440`, `7,442`, `149,307`) even if
those provenance JSON files or `.git/` packfiles differ from a byte-for-byte restore.

## Deliberately omitted

- **Per-file listings:** Individual audio and repository file hashes (~195,576 files across the
  three datasets) are omitted to keep this document compact; the tree digests above commit to the
  complete file set of each dataset.
- **Consent records:** Two local consent files exist in `<SER_DATA_DIR>` and are intentionally
  excluded from this inventory beyond their purpose and referenced licence/policy identifiers:
  - `<SER_DATA_DIR>/.ser/dataset_consents.json` — persists user acknowledgements of restricted
    dataset policies (`academic_only`, `noncommercial`, `share_alike`) and licences
    (`cc-by-nc-sa-4.0`, `msp-academic-license`, `odbl-1.0`).
  - `<SER_DATA_DIR>/restricted_backend_consents.json` — persists user acknowledgements for
    licence-restricted model backends (`emotion2vec`, licence id `other`).
  - No timestamps, source fields, or other consent-record contents are published here.
