# Spanish and Portuguese Emotional-Speech Dataset Qualification

Qualification record for candidate Spanish and Portuguese emotional-speech datasets (monolingual and multilingual) evaluated for the SER project. Every fact below is labeled **VERIFIED** (read directly at the linked primary source on 2026-11-28) or **UNVERIFIED** (could not be read at the source, with what was tried).

Verdicts are given under each of the three licence options of decision D3 (no option is recommended here):

- **D3(a):** open licences only (CC BY, CC0, ODbL).
- **D3(b):** also non-commercial and research-only data for a model used privately, never published.
- **D3(c):** also academic-only data.

Each verdict is one of `train+test`, `test only`, or `reject`.

### Verdict Rule (Applied Uniformly Across All Sources)

- **`reject`** if any of the following holds:
  - the licence class is not admitted by the D3 option;
  - the dataset licence terms are unverifiable at the source *(conditional exception: Basis Global Emotions is admitted under D3(b)/(c) because its verified official dataset card states permissive commercial and research model rights, pending owner review of the gated `LICENSE` file)*;
  - the annotations are not categorical emotions (e.g., dimensional VAD only, sentiment polarity only, or binary neutral/non-neutral);
  - the speech is synthetic (TTS);
  - raw audio is unobtainable; or
  - the corpus has fewer than 4 speakers (cannot support meaningful speaker-disjoint evaluation).
- **`test only`** if the corpus is admissible and has at least 4 speakers, **and** either (a) it lacks per-clip speaker IDs (so it cannot be split speaker-disjointly internally and must be kept whole on one side of the split as held-out test data against models trained on other corpora — or, if used for training, kept strictly out of test), or (b) it is too small (< ~400 clips or ≤ 8 speakers) to split into both train and test.
- **`train+test`** if the corpus is admissible, has more than 8 speakers and > 400 clips, **and** provides per-clip speaker IDs (or a verified speaker-disjoint split).

---

## Decision Table

| Source | Catalogue ID | Language(s) | Kind | Clips / Hours | Speaker IDs | Licence Class | D3(a) Open only | D3(b) + NC / Research-only (private) | D3(c) + Academic-only |
|---|---|---|---|---|---|---|---|---|---|
| [MESD](#1-mesd-mexican-emotional-speech-database) | `mesd` | Spanish (MX) | Acted (single words) | 862 human (+288 degraded) / UNVERIFIED | No (voice type F/M/C only) | Open (CC BY 4.0) | `test only` — Open licence, but no per-clip speaker IDs prevent internal speaker-disjoint splits; hold whole adult human subset out as test only. | `test only` — Same (no per-clip speaker IDs; single-word acted speech). | `test only` — Same (no per-clip speaker IDs; single-word acted speech). |
| [Spanish MEACorpus 2023](#2-spanish-meacorpus-2023) | `spanish-meacorpus-2023` | Spanish (ES; YouTube) | Natural (YouTube) | 5,129 / 13.16 h (790.08 min) | No (`youtube_key` in v1 only; 436 keys / 551 test rows leak across official split) | Non-commercial (CC BY-NC 4.0 in README; CC BY 4.0 in Zenodo metadata; YouTube ToS) | `reject` — GitHub README specifies CC BY-NC 4.0 and audio is subject to YouTube ToS. | `test only` — No per-clip speaker IDs (video-disjoint re-split is source-disjoint, not speaker-disjoint); hold whole corpus on one side only if YouTube audio is rehydrated. | `test only` — Same as D3(b); academic form may also provide audio if eligible, but speaker IDs remain absent. |
| [ESCorpus-PE](#3-escorpus-pe) | `escorpus-pe` | Spanish (PE) | Natural (YouTube) | 3,749 / UNVERIFIED | UNVERIFIED (80 speakers total) | Research-only (text) vs Open (Zenodo CC BY 4.0 tag); YouTube audio | `reject` — Description restricts use to research purposes and labels are dimensional (VAD) only, not categorical. | `reject` — Dimensional labels only (Valence, Arousal, Dominance); no categorical emotion labels. | `reject` — Dimensional labels only; no categorical emotion labels. |
| [CORAA-SER v1.0](#4-coraa-ser-v10) | `coraa-ser` | Portuguese (BR) | Natural (spontaneous) | 933 (625 train + 308 test) / ~50 min | Partial (recording-session prefix only) | Unspecified in repo (source C-ORAL-BRASIL I is CC BY-NC-SA 4.0) | `reject` — No open licence stated (upstream is CC BY-NC-SA 4.0) and labels are binary affect + gender, not named emotions. | `reject` — Only 3 classes (`neutral`, `non-neutral-female`, `non-neutral-male`); no named emotion categories. | `reject` — No named emotion categories. |
| [emoUERJ](#5-emouerj) | — | Portuguese (BR) | Acted (10 sentences) | 377 / UNVERIFIED | Yes (`m01`–`m04`, `w01`–`w04` in filename) | Open (CC BY 4.0) | `test only` — Open CC BY 4.0 with speaker IDs and 4 named emotions, but 377 clips / 8 actors is below the train+test threshold. | `test only` — Same: best reserved as a speaker-disjoint PT-BR evaluation set. | `test only` — Same: small size suits held-out evaluation. |
| [VERBO](#6-verbo) | — | Portuguese (BR) | Acted (14 sentences) | 1,167 / UNVERIFIED | Yes (`f1`–`f6`, `m1`–`m6` in filename) | On-request (dataset licence UNVERIFIED; form link 404) | `reject` — On-request dataset with broken form link (404) and unverified licence. | `reject` — Dataset licence text could not be verified (request form returns 404). | `reject` — Dataset licence text is UNVERIFIED; re-evaluate only if authors provide written terms. |
| [EmoFilm](#7-emofilm) | — | Multilingual (EN, IT, ES) | Acted (film dubs) | 1,115 total / 342 ES (~20 min ES) | UNVERIFIED (57 ES speakers; archive gated) | Academic-only (EULA) | `reject` — Academic-only EULA requiring permanent university faculty signature. | `reject` — EULA prohibits any non-academic purpose and requires permanent academic signatory. | `test only` — Usable only with permanent academic signatory; 342 ES clips (< 400 threshold, no `neutral`) suits held-out testing once speaker IDs are verified. |
| [INTERFACE Spanish (ELRA-S0329)](#8-interface-spanish-elra-s0329) | — | Spanish (ES, Castilian) | Acted (read text) | ~5,520 / 7 h 52 min (2 speakers) | Yes (2 speakers: 1M, 1F) | Non-commercial (0 EUR academic / 4,000 EUR non-academic ELRA END USER) or Commercial paid (ELRA VAR) | `reject` — Proprietary ELRA agreement (not open) and fewer than 4 speakers (2 only). | `reject` — Fewer than 4 speakers (2 only); requires 4,000 EUR non-academic ELRA contract and bars model publishing. | `reject` — Fewer than 4 speakers (1M, 1F) fails the minimum-speaker threshold for speaker-disjoint evaluation. |
| [EMOVOME](#9-emovome) | — | Spanish (ES) | Natural (WhatsApp voice messages) | 999 (+100 read) / 293 min (~4.9 h) | Yes (`participants_ids.csv`, 100 speakers; 80/20 split) | Academic-only for raw audio (CC BY 4.0 for metadata/features only) | `reject` — Raw audio is excluded from CC BY 4.0 and gated by an institutional agreement. | `reject` — Raw audio agreement is academic-only (requires signature of a legally authorised representative of a research institution and excludes profit-making organisations). | `train+test` — Natural real-life Spanish speech, 100 speakers, official speaker-disjoint 80/20 split, 7 expert emotion categories; usable if an eligible research institution signs. |
| [EmoSpanishDB / EmoMatchSpanishDB](#10-emospanishdb--emomatchspanishdb) | — | Spanish (ES) | Elicited (lab) | 3,550 / 2,020 matched / UNVERIFIED | Yes (50 speakers; LOSO in paper) | Academic-only for raw audio (signed agreement; MIT for posted features) | `reject` — Raw audio requires an academic-only signed agreement (MIT applies only to posted features). | `reject` — Agreement restricts raw audio "exclusively in academic, not-for-profit research" with a university signature block. | `train+test` — Largest elicited Spanish 7-class corpus (50 speakers, per-clip LOSO, 48 kHz 16-bit WAV); usable if an eligible academic researcher signs the agreement. |
| [Basis Global Emotions](#11-basis-global-emotions) | — | Multilingual (108 locales: 9 ES, 2 PT) | Acted / Steered (scripted lines) | 14,320 ES (54.69 h) + 3,771 PT (15.06 h) | Yes (`speaker_id`: 862 ES, 134 PT) | Custom permissive (`basis-data-license-1.0`, no raw audio redistribution, manual HF gate; full text UNVERIFIED) | `reject` — Custom no-redistribution licence with manual access gate (not CC BY/CC0/ODbL) and full `LICENSE` file behind login gate. | `train+test` — Conditional exception: verified card states royalty-free commercial and research model training/publishing; confirm full `LICENSE` upon HF approval before use. | `train+test` — Same as D3(b); confirm full `LICENSE` text after HF approval. |
| [CMU-MOSEAS](#12-cmu-moseas) | — | Multilingual (ES, PT, DE, FR) | Natural (YouTube monologues) | 10,000 ES + 10,000 PT sentences / 68 h 49 m total (4 langs) | Partial (341 ES, 399 PT speakers via face ID; video ID ≠ speaker ID) | On-request EULA for raw audio/labels (EULA text UNVERIFIED); SDK code MIT | `reject` — Raw audio and labels require an unverified EULA via email; YouTube copyright applies. | `reject` — Public SDK distributes only non-invertible features; raw audio EULA is UNVERIFIED. | `reject` — Raw audio EULA is UNVERIFIED and 640 GB raw archive requires author approval. |
| [MOUD](#13-moud-multimodal-opinion-utterances-dataset) | — | Spanish (YouTube) | Natural (YouTube reviews) | ~400–550 / ~59 min | Yes (101–105 speakers) | Open page notice (GNU GPL); YouTube source rights unclear | `reject` — Sentiment polarity only (`pos`/`neg`/`neu`), not categorical emotions; YouTube rights unclear. | `reject` — Sentiment polarity only; no categorical emotion labels. | `reject` — Sentiment polarity only; no categorical emotion labels. |
| [LAION Artificial Voice Roleplay](#14-laion-artificial-voice-roleplay-dataset) | — | Multilingual (DE, EN, ES, FR) | Synthetic (TTS) | 2,106 ES (of 67,491 total) / UNVERIFIED for ES | No (synthetic voices; ~40% non-human creatures) | Open (CC BY 4.0) | `reject` — Fully synthetic TTS (including creature voices), no speaker IDs, and free-text captions rather than categorical emotion labels. | `reject` — Same: synthetic TTS without categorical emotion labels or speaker IDs. | `reject` — Same: synthetic TTS without categorical emotion labels or speaker IDs. |
| [RekEmozio](#15-rekemozio) | — | Multilingual (ES, EU) | Acted (lab) | 2,618 ES / UNVERIFIED | UNVERIFIED (10 speakers total) | On-request / Not free (licence UNVERIFIED; source URLs 404) | `reject` — Not free/open per literature; official pages returned 404 and licence is UNVERIFIED. | `reject` — Official distribution and licence terms are UNVERIFIED (source URLs 404). | `reject` — Official distribution and licence terms are UNVERIFIED (source URLs 404). |

---

## What Applies Now

Owner decision (2026-11-28): **English and Spanish** are the benchmark languages now; **Portuguese** is checked on the owner's own recordings; **option D3(c)** applies (all catalogued and qualified data, including academic-only data, may be used for privately used models).

### Spanish — Ranked Acquisition List (Option D3(c), Best First)

| Rank | Source | Clips / Hours | Speakers | Test-eligible? | Access Steps |
|---|---|---|---|---|---|
| 1 | **Basis Global Emotions** (9 ES locales) | 14,320 / 54.69 h | 862 (per-clip `speaker_id`) | Yes (`train+test`) | **Needs owner action:** Request manual access on [Hugging Face](https://huggingface.co/datasets/basis-ai/basis-global-emotions) (requires HF account and agreeing not to identify/impersonate speakers) and read the full 16.6 kB `LICENSE` file before downloading. Adds multi-accent Spanish (9 countries) and 48 kHz WAV with permissive commercial/research model terms per card. |
| 2 | **EmoMatchSpanishDB / EmoSpanishDB** | 2,020 matched / 3,550 total | 50 (per-clip LOSO) | Yes (`train+test`) | **Needs owner action:** Download `EMOMATCHAgreement.docx` from [figshare 14215850](https://doi.org/10.6084/m9.figshare.14215850), fill and sign (requires Researcher Full Name, Position, University, E-mail contact, Signature; academic not-for-profit research only), and email it to the corresponding author listed in the agreement to receive the raw audio download link. |
| 3 | **EMOVOME** | 999 spontaneous (293 min) | 100 (per-clip ID, 80/20 split) | Yes (`train+test`) | **Needs owner action:** Obtain `EMOVOME_agreement.pdf` from [Zenodo 10694370](https://zenodo.org/records/10694370), have it signed by a legally authorised representative of a research institution (e.g., head of laboratory/department), and email it to the corresponding author listed on Zenodo to receive the `Audios` folder. |
| 4 | **Spanish MEACorpus 2023** | 5,129 / 13.16 h | Unknown (3,006 videos; no speaker IDs) | Only as whole held-out test set (`test only`) | **Direct CSV download + rehydration:** Download v1 CSV ([Zenodo 18246241](https://zenodo.org/records/18246241)) and re-fetch/segment YouTube audio with a patched script (source states audio remains subject to YouTube's Terms of Service and original content creators' rights); **or needs owner action:** submit the academic-purpose Google Form linked on the [GitHub README](https://github.com/NLP-UMUTeam/Spanish-MEACorpus-2023). |
| 5 | **MESD** | 862 human (single words) | 11 (no per-clip speaker IDs) | Only as whole held-out test set (`test only`) | **Direct download:** Open CC BY 4.0 on [Mendeley Data v5](https://data.mendeley.com/datasets/cy34mh68j9/5) — no form, account, or agreement. Exclude the 288 naturalness-reduced clips (`Level1`/`Level2`). |
| 6 | **EmoFilm** (Spanish subset) | 342 / ~20 min | 57 dubbing actors (IDs UNVERIFIED) | Yes (`test only`, once IDs verified; no `neutral`) | **Needs owner action:** Download the EULA linked on [Zenodo 7665999](https://zenodo.org/records/7665999), have it signed by a person holding a permanent position at an academic institute (typically a full professor), and submit it from an institutional email address via Zenodo / email to the contact listed on Zenodo. |

### Portuguese — Ranked List (Not Needed for the Benchmark Yet)

Portuguese is currently validated on the owner's own recordings. If external Portuguese data is added later under D3(c):

| Rank | Source | Clips / Hours | Speakers | Test-eligible? | Access Steps |
|---|---|---|---|---|---|
| 1 | **Basis Global Emotions** (`pt-BR`, `pt-PT`) | 3,771 / 15.06 h | 134 (per-clip `speaker_id`) | Yes (`train+test`) | **Needs owner action:** Same Hugging Face manual access request and `LICENSE` review as Spanish rank 1. |
| 2 | **emoUERJ** (`pt-BR`) | 377 | 8 (speaker IDs in filename) | Yes (`test only`, 4 classes) | **Direct download:** Open CC BY 4.0 on [Zenodo 5427549](https://zenodo.org/records/5427549) — no form, account, or agreement. |

### Needs Owner Action (Summary)

- **Basis Global Emotions** ([Hugging Face](https://huggingface.co/datasets/basis-ai/basis-global-emotions)): submit manual HF access request and verify the full 16.6 kB `basis-data-license-1.0` `LICENSE` file. Would add 54.69 h of Spanish (9 locales, 862 speakers) and 15.06 h of Portuguese (`pt-BR` + `pt-PT`, 134 speakers) with per-clip speaker IDs.
- **EmoMatchSpanishDB** ([figshare 14215850](https://doi.org/10.6084/m9.figshare.14215850)): sign `EMOMATCHAgreement.docx` (academic/university affiliation required) and email to the author in the agreement.
- **EMOVOME** ([Zenodo 10694370](https://zenodo.org/records/10694370)): obtain signature from a legally authorised representative of a research institution on `EMOVOME_agreement.pdf` and email to the corresponding author on Zenodo.
- **EmoFilm** ([Zenodo 7665999](https://zenodo.org/records/7665999)): obtain permanent university faculty signature on the EULA and submit from an institutional email address.
- **Spanish MEACorpus 2023 (optional)** ([GitHub README](https://github.com/NLP-UMUTeam/Spanish-MEACorpus-2023)): submit the academic-purpose Google Form if pre-segmented audio is preferred over YouTube rehydration.

---

## Model-Rights Summary by Licence Class (Engineering Facts)

The table below summarizes what each source's licence explicitly states about a **model trained on the data** (not legal advice; not a gate).

| Licence / Agreement | Publish trained model? | Use model privately? | Use model commercially? | Source basis |
|---|---|---|---|---|
| **CC BY 4.0** (MESD, emoUERJ, LAION Roleplay, EMOVOME metadata only) | Yes (silent on ML weights specifically; permits reproducing and sharing Licensed Material and Adapted Material for any purpose with attribution). | Yes. | Yes. | [CC BY 4.0 Legal Code](https://creativecommons.org/licenses/by/4.0/legalcode.en) Section 2(a)(1). |
| **CC BY-NC 4.0** (Spanish MEACorpus 2023 GitHub README) | Only for NonCommercial purposes (silent on whether trained weights constitute Adapted Material; NC restriction applies to all Licensed Rights). | Yes, if NonCommercial ("not primarily intended for or directed towards commercial advantage or monetary compensation"). | No. | [CC BY-NC 4.0 Legal Code](https://creativecommons.org/licenses/by-nc/4.0/legalcode.en) Sections 1(i), 2(a)(1). Note: underlying YouTube audio is separately subject to YouTube ToS and creator copyrights. |
| **CC BY-NC-SA 4.0** (C-ORAL-BRASIL I, upstream of CORAA-SER) | Only NonCommercial and under the same ShareAlike licence if weights are deemed Adapted Material (silent on ML weights). | Yes, if NonCommercial. | No. | [C-ORAL-BRASIL I](https://www.c-oral-brasil.org/c-oral-brasil-i.php). |
| **EmoFilm EULA** | Not stated explicitly; Article 1 bars "any non-academic purpose" including "training or testing of commercial systems" and Article 6 restricts use "only for academic research". | Only within the named academic institute (signee + up to 4 named researchers). | No. | [EmoFilm EULA PDF](https://drive.google.com/file/d/1pFHfsqk7snF_EVqq0WAC0Dz8FcTD3s9_/view?usp=share_link). |
| **ELRA END USER Agreement** (INTERFACE / ELRA-S0329) | **No.** Clauses 8–9 explicitly forbid distributing, marketing, or making available to the public (including free/open-source) any derived "Language Models" (defined in Clause 2 as output of supervised/unsupervised learning). | Only at the licensed site for "bona fide language engineering research activities" (Clauses 3–4, 11). | No (Clauses 7–8). | [ELRA End-User Template](http://www.elra.info/media/filer_public/2022/01/21/enduser_220121.pdf). |
| **ELRA VAR Agreement** (INTERFACE / ELRA-S0329 commercial) | **No.** Clause 10 explicitly forbids making available to third parties any Language Models based on all or any part of the Language Resources (even when incorporated into a larger dataset). | Yes, at the licensed site. | Commercial products/services allowed only if they do not expose or allow extraction of the Language Models or reconstruction of the data (Clauses 9–10). | [ELRA VAR Template](http://www.elra.info/media/filer_public/2022/01/21/var_220121.pdf). |
| **EMOVOME Audio Agreement** | Not explicitly stated in Usage Notes; prohibits sharing data with profit-making companies/organisations and expects "research purposes only" under an institutional signatory. Full `EMOVOME_agreement.pdf` inside `emovome.zip` (1.3 GB) was not downloaded. | Only within the signing research institution for research purposes. | No. | [Zenodo 10694370](https://zenodo.org/records/10694370). |
| **EmoMatchSpanishDB Audio Agreement** | Not explicitly stated in the agreement; restricts use "exclusively in academic, not-for-profit research" and "only in basic and health research projects", and bars providing materials to profit-making companies or organisations. | Only within the signing academic laboratory for research purposes. | No. | [EMOMATCHAgreement.docx (figshare)](https://ndownloader.figshare.com/files/34319516). |
| **Basis Data License 1.0** (Basis Global Emotions) | **Yes** (per dataset card summary: "Models and results built on it are yours: you may publish, distribute, license and sell them..."). Full `LICENSE` file behind HF gate (UNVERIFIED verbatim). | **Yes** (per card). | **Yes** (per card: "operate commercial products, hosted inference services and APIs with them, with no royalties, no revenue sharing"). | [Hugging Face card](https://huggingface.co/datasets/basis-ai/basis-global-emotions). |

---

## Per-Source Qualification

### 1. MESD (Mexican Emotional Speech Database)

- **SER catalogue ID:** `mesd` (`dataset_policy_id="open"`, `dataset_license_id="cc-by-4.0"`).
- **Official URL & institution:** [Mendeley Data — MESD v5](https://data.mendeley.com/datasets/cy34mh68j9/5), Instituto Tecnológico y de Estudios Superiores de Monterrey (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
- **Version, DOI & checksum:** Version 5 (published 17 February 2022), DOI `10.17632/cy34mh68j9.5` (**VERIFIED**, [DataCite API](https://api.datacite.org/dois/10.17632/cy34mh68j9.5), [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)). Root `README.txt` SHA-256: `58752d4d03aacb735e1d1b32f132873b113b16ff0dd6f31f740373163b08e78b` (**VERIFIED**, Mendeley public API). Individual audio files carry per-file SHA-256 hashes in the Mendeley API; no single archive checksum.
- **Licence (quoted verbatim):**
  > "CC BY 4.0" — [Mendeley Data v5](https://data.mendeley.com/datasets/cy34mh68j9/5)
  > "Creative Commons Attribution 4.0 International" (`https://creativecommons.org/licenses/by/4.0/legalcode`) — [DataCite metadata for 10.17632/cy34mh68j9.5](https://api.datacite.org/dois/10.17632/cy34mh68j9.5)
  - **Licence class:** Open (CC BY 4.0) (**VERIFIED**).
- **Redistribution allowed:** Yes, with attribution under CC BY 4.0 (**VERIFIED**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)).
- **Access requirements:** None — direct public download via Mendeley Data web/API without form, account, or agreement (**VERIFIED**).
- **Total size:** 106.64 MB (`106,642,791` bytes) across 1,150 WAV files (89.74 MB / `89,736,128` B across 862 WAVs + 1 `desktop.ini` in `Mexican Emotional Speech Database (MESD)` + 8.71 MB in `Level1` [144 files] + 8.20 MB in `Level2` [144 files] + 663 B `README.txt`) (**VERIFIED**, Mendeley public API `https://data.mendeley.com/public-api/datasets/cy34mh68j9/files`).
- **Clips & hours:** 862 human single-word WAV files + 288 naturalness-reduced female WAV files (144 Level 1 + 144 Level 2) = 1,150 WAV files total (**VERIFIED**, Mendeley public API). Total audio hours: **UNVERIFIED** (tried Mendeley page, DataCite metadata, and Crossref abstract of [Duville et al., 2021](https://doi.org/10.3390/data6120130); MDPI full text returned HTTP 403).
- **Speakers & per-clip speaker ID:** 11 non-professional actors: 3 adult female, 2 adult male, 6 child voices (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)). **Speaker identity is NOT given per clip:** `README.txt` specifies the filename pattern `<emotion>_<type of voice>_<word corpus>_<word>`, where `<type of voice>` is only `F`, `M`, or `C` (**VERIFIED**, [Mendeley `README.txt`](https://data.mendeley.com/public-files/datasets/cy34mh68j9/files/21fc0848-7df6-4dcb-a6af-1d23b44c1dc8/file_viewed)).
- **Emotion labels (as named by source):** `Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness` (**VERIFIED**, [Mendeley `README.txt`](https://data.mendeley.com/public-files/datasets/cy34mh68j9/files/21fc0848-7df6-4dcb-a6af-1d23b44c1dc8/file_viewed)).
- **Authenticity:** Acted (single-word utterances by non-professional actors in a studio; plus two synthetically modified naturalness-reduced subsets) (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
- **Language variety:** Mexican Spanish (`es-MX`) (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
- **Sample rate & format:** 48,000 Hz, 24-bit WAV, amplitude rescaled to [-1, 1] (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
- **Known problems:**
  - Single-word utterances only (short duration, no sentence-level prosody) (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
  - No individual speaker ID in filenames, preventing speaker-disjoint train/test splitting within MESD (**VERIFIED**, [Mendeley `README.txt`](https://data.mendeley.com/public-files/datasets/cy34mh68j9/files/21fc0848-7df6-4dcb-a6af-1d23b44c1dc8/file_viewed)).
  - 288 of the 1,150 WAV files are acoustically degraded ("naturalness-reduced") derivatives of female corpus B utterances and must be excluded from natural/human evaluation (**VERIFIED**, [Mendeley Data](https://data.mendeley.com/datasets/cy34mh68j9/5)).
  - Expected 864 human files (24 utterances × 6 emotions × 3 voice types × 2 corpora), but folder contains 862 WAV files plus `desktop.ini` (2 clips missing: `Anger_C_A` and `Neutral_C_A` have 23 each) (**VERIFIED**, Mendeley public API).
- **Model rights (engineering fact):** CC BY 4.0 permits private use, commercial use, and publishing derived models, subject to attribution.
- **Verdicts:**
  - **D3(a):** `test only` — Open CC BY 4.0, but missing per-clip speaker IDs prevent internal speaker-disjoint splits; use human adult subset as held-out test only.
  - **D3(b):** `test only` — Same.
  - **D3(c):** `test only` — Same.

---

### 2. Spanish MEACorpus 2023

- **SER catalogue ID:** `spanish-meacorpus-2023` (`dataset_policy_id="noncommercial"`, `dataset_license_id="cc-by-nc-4.0"`).
- **Official URL & institution:** [Zenodo 18606423 (v2)](https://zenodo.org/records/18606423), [Zenodo 18246241 (v1)](https://zenodo.org/records/18246241), [GitHub NLP-UMUTeam/Spanish-MEACorpus-2023](https://github.com/NLP-UMUTeam/Spanish-MEACorpus-2023); Universidad de Murcia & UNED (**VERIFIED**).
- **Version, DOI & checksum:** Concept DOI `10.5281/zenodo.18246240`. Version v2: DOI `10.5281/zenodo.18606423`, file `spanish-meacorpus-2023-dataset.csv` (1,010,439 B, `md5:c5ac5faac626c9cb71b1d16f01957e4d`). Version v1: DOI `10.5281/zenodo.18246241`, file `dataset.csv` (1,189,620 B, `md5:7a31c1259f6601ca21015caf39c2858d`) (**VERIFIED**, [Zenodo API v2](https://zenodo.org/api/records/18606423), [Zenodo API v1](https://zenodo.org/api/records/18246241)).
- **Licence (quoted verbatim):**
  > "The dataset annotations, transcriptions, and metadata are distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) licence.
  > Audio and video content remain subject to YouTube’s Terms of Service and the original content creators’ rights." — [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md)
  > Zenodo metadata conflict: both [Zenodo API v2](https://zenodo.org/api/records/18606423) and [Zenodo API v1](https://zenodo.org/api/records/18246241) record `"license": {"id": "cc-by-4.0"}`.
  - **Licence class:** Non-commercial (treating the explicit GitHub README CC BY-NC 4.0 statement as governing over the Zenodo metadata tag; audio itself is governed by YouTube ToS and original creator copyrights) (**VERIFIED**).
- **Redistribution allowed:** Annotations/metadata/transcriptions only (under CC BY-NC 4.0). Audio and video files are **not** redistributed due to copyright restrictions (**VERIFIED**, [Zenodo 18606423](https://zenodo.org/records/18606423), [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md)).
- **Access requirements:** CSV metadata is openly downloadable without login. Audio is not redistributed; rehydration via `code/download-dataset.py` requires patching the script or joining v1 and v2 CSVs (the script reads `filename` from v2 alongside `youtube_key`, `start_time`, and `end_time` from v1, so it runs on neither CSV unmodified, and it relies on `pytube`). Alternatively, academic users can request the dataset via the Google Form (`https://forms.gle/M668aLqHEuA4Qp1k7`) linked in the README (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md), [download-dataset.py](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/code/download-dataset.py)).
- **Total size:** Metadata CSV is ~1.0–1.2 MB (**VERIFIED**, Zenodo API). Full rehydrated audio size: **UNVERIFIED** (audio not redistributed; 13.16 h of MP3 segments).
- **Clips & hours:** 5,129 segments (4,102 train, 1,027 test), 790.08 min (13.16 h), average 9.24 s per segment (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md) and direct row count of v1/v2 CSVs).
- **Speakers & per-clip speaker ID:** Total distinct speaker count: **UNVERIFIED** (tried GitHub README, Zenodo records, and v1/v2 CSVs; Elsevier paper is behind paywall). **No explicit speaker ID column** exists in either CSV; v1 provides `youtube_key` (3,006 unique YouTube video IDs) and `gender`, while v2 drops `youtube_key`, `start_time`, `end_time`, and `gender`, keeping only `filename`, `label`, `transcription`, `split` (**VERIFIED**, [Zenodo v1 CSV](https://zenodo.org/api/records/18246241/files/dataset.csv/content), [Zenodo v2 CSV](https://zenodo.org/api/records/18606423/files/spanish-meacorpus-2023-dataset.csv/content)).
- **Emotion labels (as named by source):** `anger` (668), `disgust` (1,200), `fear` (37), `joy` (643), `neutral` (1,999), `sadness` (582) (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md) and CSV counts).
- **Authenticity:** Natural / spontaneous (YouTube videos from natural environments, heavily featuring Spanish political speeches/debates visible in the CSV transcriptions) (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md) and CSV sample).
- **Language variety:** Spanish (predominantly Peninsular Spanish political/media content in inspected CSV rows; exact dialect breakdown **UNVERIFIED**).
- **Sample rate & format:** Rehydration script (`code/download-dataset.py`) writes `.mp3` clips via `moviepy` from YouTube MP4 streams; exact sample rate depends on YouTube stream (**VERIFIED**, [download-dataset.py](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/code/download-dataset.py)).
- **Known problems:**
  - **Licence contradiction:** Zenodo metadata says `cc-by-4.0`, GitHub README says `CC BY-NC 4.0`, and README also links an academic-only request form (**VERIFIED**).
  - **v2 schema regression & broken download script:** Zenodo v2 (`18606423`) and GitHub `output/spanish-meacorpus-2023-dataset.csv` omit `youtube_key`, `start_time`, `end_time`, and `gender`. Meanwhile `code/download-dataset.py` reads both `row['filename']` (v2 only) and `row['youtube_key']`, `row['start_time']`, `row['end_time']` (v1 only), so it runs on neither CSV unmodified (needs a v1+v2 join or script patch) and depends on `pytube` (**VERIFIED**).
  - **Official split leakage & missing speaker IDs:** Although the README claims "speaker-independent criteria", inspecting v1 `dataset.csv` shows **436 of the 877 unique `youtube_key`s in `test` also appear in `train`**, covering **551 of the 1,027 test rows** (**VERIFIED**, [Zenodo v1 CSV](https://zenodo.org/api/records/18246241/files/dataset.csv/content)). Because neither CSV has speaker IDs and recurring political speakers appear across different videos, a re-split by `youtube_key` is only video-disjoint (source-disjoint), not speaker-disjoint.
  - **Extreme class imbalance:** `fear` has only 37 segments (29 train, 8 test) (**VERIFIED**).
  - **YouTube re-fetching terms & link rot:** The source's own terms state: `"Due to copyright restrictions, audio and video files are not redistributed."` and `"Audio and video content remain subject to YouTube’s Terms of Service and the original content creators’ rights."` ([GitHub README](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/README.md)) (**VERIFIED**). Re-fetching from YouTube also risks deleted or private videos.
- **Model rights (engineering fact):** CC BY-NC 4.0 restricts use to non-commercial purposes; audio is additionally bound by YouTube ToS and original creators' copyrights (no commercial use or public weight distribution without separate clearance).
- **Verdicts:**
  - **D3(a):** `reject` — GitHub README applies CC BY-NC 4.0 and audio is subject to YouTube ToS.
  - **D3(b):** `test only` — No per-clip speaker IDs (a `youtube_key` re-split is video-disjoint, not speaker-disjoint); hold whole corpus on one side only (held-out test set, or train-only with no internal test split) if YouTube audio is rehydrated.
  - **D3(c):** `test only` — Same as D3(b); academic form may provide audio if eligible, but speaker IDs remain absent.

---

### 3. ESCorpus-PE

- **SER catalogue ID:** `escorpus-pe` (`dataset_policy_id="open"`, `dataset_license_id="cc-by-4.0"`).
- **Official URL & institution:** [Zenodo 5793223](https://zenodo.org/records/5793223), [GitHub Alessandra-UNSA/Peruvian_Spanish_Corpus](https://github.com/Alessandra-UNSA/Peruvian_Spanish_Corpus) (also older repo [Alessandra-UNSA/ESCorpus-PE](https://github.com/Alessandra-UNSA/ESCorpus-PE)); Universidad Nacional de San Agustín (Peru) (**VERIFIED**).
- **Version, DOI & checksum:** Version 1.0.0 (published 23 December 2021), DOI `10.5281/zenodo.5793223`, file `Corpus_Globalv1.zip` (1,878,886,206 B / 1.9 GB, `md5:3fbd44f666a9938e22a9d12325d69ecd`) (**VERIFIED**, [Zenodo API 5793223](https://zenodo.org/api/records/5793223)).
- **Licence (quoted verbatim):**
  > Zenodo metadata: `"license": {"id": "cc-by-4.0"}` — [Zenodo API 5793223](https://zenodo.org/api/records/5793223)
  > Zenodo description & GitHub README: `"Anyone can use this data only for research purposes."` — [Zenodo 5793223](https://zenodo.org/records/5793223), [GitHub README](https://raw.githubusercontent.com/Alessandra-UNSA/Peruvian_Spanish_Corpus/main/README.md)
  - **Licence class:** Research-only (explicit restriction in description and README contradicts Zenodo's `cc-by-4.0` metadata tag; audio was also extracted from YouTube videos) (**VERIFIED**).
- **Redistribution allowed:** Unclear — Zenodo `cc-by-4.0` allows redistribution, but the authors' text restricts to "research purposes" and the audio was extracted from 15 YouTube videos without stated third-party copyright clearance (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/Alessandra-UNSA/Peruvian_Spanish_Corpus/main/README.md)).
- **Access requirements:** None — direct public download from Zenodo without form or account (**VERIFIED**).
- **Total size:** 1.88 GB (`1,878,886,206` bytes) (**VERIFIED**, [Zenodo API 5793223](https://zenodo.org/api/records/5793223)).
- **Clips & hours:** 3,749 utterances (**VERIFIED**, [Zenodo 5793223](https://zenodo.org/records/5793223)). Total hours: **UNVERIFIED** (tried Zenodo record, both GitHub repos, and Crossref search; original paper not openly accessible).
- **Speakers & per-clip speaker ID:** 80 speakers (44 male, 36 female) (**VERIFIED**, [Zenodo 5793223](https://zenodo.org/records/5793223)). Per-clip speaker ID availability: **UNVERIFIED** (Zenodo preview returned "Too many files inside the ZIP file", and downloading the 1.9 GB archive is prohibited).
- **Emotion labels (as named by source):** Three continuous dimensions only: `Valence`, `Arousal`, `Dominance` (**VERIFIED**, [Zenodo 5793223](https://zenodo.org/records/5793223)). **No categorical emotion labels.** *(Note: SER's catalogue entry `ESCORPUS_PE_MANIFEST_SPEC` currently defines a categorical Spanish label map that does not match the source's dimensional annotations.)*
- **Authenticity:** Natural (interviews, TV reports, political debate, and testimonials from YouTube) (**VERIFIED**, [Zenodo 5793223](https://zenodo.org/records/5793223)).
- **Language variety:** Peruvian Spanish (`es-PE`) (**VERIFIED**, [Zenodo 5793223](https://zenodo.org/records/5793223)).
- **Sample rate & format:** WAV format extracted via Praat scripts (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/Alessandra-UNSA/Peruvian_Spanish_Corpus/main/README.md)); exact sample rate and bit depth **UNVERIFIED** (archive not downloaded).
- **Known problems:**
  - Licence contradiction (`cc-by-4.0` tag vs `"only for research purposes"` text) and redistributes YouTube-derived audio (**VERIFIED**).
  - Dimensional labels only (VAD); incompatible with SER's categorical emotion head (**VERIFIED**).
- **Model rights (engineering fact):** Source text limits use to "research purposes" (incompatible with commercial deployment; publishing models unclear due to YouTube-derived audio and research-only text).
- **Verdicts:**
  - **D3(a):** `reject` — Research-only text conflicts with CC BY 4.0 tag, YouTube copyright unclear, and no categorical emotion labels.
  - **D3(b):** `reject` — Dimensional (VAD) labels only; no categorical emotion labels.
  - **D3(c):** `reject` — Dimensional (VAD) labels only; no categorical emotion labels.

---

### 4. CORAA-SER v1.0

- **SER catalogue ID:** `coraa-ser` (`dataset_policy_id="research_only"`, `dataset_license_id="custom-research-only"`).
- **Official URL & institution:** [GitHub rmarcacini/ser-coraa-pt-br](https://github.com/rmarcacini/ser-coraa-pt-br), [SE&R 2022 Workshop](https://sites.google.com/view/ser2022/shared-tasks), [CEUR-WS Vol-3175 Overview Paper](https://ceur-ws.org/Vol-3175/paper01.pdf); University of São Paulo (C4AI-USP / TaRSila project) & UTFPR (**VERIFIED**).
- **Version, DOI & checksum:** Version 1.0 (2022); no DOI or archive checksum published on GitHub or Google Drive (**VERIFIED**, [GitHub repo](https://api.github.com/repos/rmarcacini/ser-coraa-pt-br)).
- **Licence (quoted verbatim):**
  > No licence file or licence statement is present in `rmarcacini/ser-coraa-pt-br` (`"license": null` in GitHub API) or in the SE&R 2022 overview paper. The corpus is derived from C-ORAL-BRASIL I, whose official site states:
  > `"O corpus C-Oral-Brasil-I está licenciado com uma licença Creative Commons Atribuição-NãoComercial-CompartilhaIgual 4.0 Internacional (CC BY-NC-SA 4.0)"` — [C-ORAL-BRASIL I](https://www.c-oral-brasil.org/c-oral-brasil-i.php)
  > *(For comparison, sibling corpus CORAA ASR v1.1 at [nilc-nlp/CORAA](https://raw.githubusercontent.com/nilc-nlp/CORAA/main/LICENSE) uses `Attribution-NonCommercial-NoDerivatives 4.0 International`.)*
  - **Licence class:** Non-commercial (inherited from C-ORAL-BRASIL I CC BY-NC-SA 4.0; explicit CORAA-SER v1.0 licence is **UNVERIFIED** because none is published in the repository or workshop page) (**VERIFIED**).
- **Redistribution allowed:** Under upstream C-ORAL-BRASIL I (CC BY-NC-SA 4.0), non-commercial ShareAlike redistribution is allowed; CORAA-SER's own redistribution terms are **UNVERIFIED**.
- **Access requirements:** None — `data_train.zip`, `test_ser.zip`, and `test_ser_metadata.csv` are linked publicly on Google Drive without a form or agreement (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/rmarcacini/ser-coraa-pt-br/main/README.md)).
- **Total size:** **UNVERIFIED** (Google Drive ZIPs not downloaded).
- **Clips & hours:** 933 clips total — 625 train (491 `neutral`, 89 `non-neutral-female`, 45 `non-neutral-male`) + 308 test (248 `neutral`, 37 `non-neutral-female`, 23 `non-neutral-male`); approximately 50 minutes total (**VERIFIED**, [SE&R 2022 paper01.pdf](https://ceur-ws.org/Vol-3175/paper01.pdf) and `test_ser_metadata.csv`).
- **Speakers & per-clip speaker ID:** Total speaker count: **UNVERIFIED**. `test_ser_metadata.csv` filenames carry the C-ORAL-BRASIL recording-session prefix (e.g. `bfamcv32_segment244_neutral.wav`, 82 unique session prefixes in test), not individual speaker IDs (and multiple speakers overlap in some clips) (**VERIFIED**, `test_ser_metadata.csv` and [SE&R 2022 paper04.pdf](https://ceur-ws.org/Vol-3175/paper04.pdf)).
- **Emotion labels (as named by source):** `neutral`, `non-neutral-female`, `non-neutral-male` (3 classes; non-neutral combines all primary emotions and paralinguistic cues like laughing/crying, split by gender) (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/rmarcacini/ser-coraa-pt-br/main/README.md), [SE&R 2022 paper01.pdf](https://ceur-ws.org/Vol-3175/paper01.pdf)).
- **Authenticity:** Natural (informal spontaneous Brazilian Portuguese speech from C-ORAL-BRASIL I) (**VERIFIED**, [GitHub README](https://raw.githubusercontent.com/rmarcacini/ser-coraa-pt-br/main/README.md)).
- **Language variety:** Brazilian Portuguese (`pt-BR`) (**VERIFIED**).
- **Sample rate & format:** Raw `.wav` files (**VERIFIED**, [SE&R 2022 paper01.pdf](https://ceur-ws.org/Vol-3175/paper01.pdf)); exact sample rate and bit depth **UNVERIFIED** (audio not downloaded).
- **Known problems:**
  - No named emotion categories (only neutral vs gender-split non-neutral) (**VERIFIED**).
  - Qualitative audit by SE&R 2022 participants ([Alves et al., 2022](https://ceur-ws.org/Vol-3175/paper04.pdf)) found substantial background noise, overlapping voices, and mixed-gender speakers within single segments (e.g. 56 neutral training clips contain voices of different genders, 53 with overlap) (**VERIFIED**).
  - Severe class imbalance (~79% neutral, ~7% non-neutral-male) and tiny size (~50 min) (**VERIFIED**).
- **Model rights (engineering fact):** No explicit CORAA-SER licence; under upstream C-ORAL-BRASIL I (CC BY-NC-SA 4.0), commercial use is prohibited and non-commercial use/sharing requires attribution (and ShareAlike if models are deemed Adapted Material).
- **Verdicts:**
  - **D3(a):** `reject` — No open licence (upstream CC BY-NC-SA 4.0) and no named emotion categories.
  - **D3(b):** `reject` — Binary affect + gender labels only (`neutral`, `non-neutral-female`, `non-neutral-male`); incompatible with categorical emotion recognition.
  - **D3(c):** `reject` — Same: no named emotion categories.

---

### 5. emoUERJ

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Zenodo 5427549](https://zenodo.org/records/5427549); State University of Rio de Janeiro (UERJ), with co-authors at CEFET/RJ and COPPE/UFRJ (**VERIFIED**, [Zenodo API 5427549](https://zenodo.org/api/records/5427549)).
- **Version, DOI & checksum:** Version 1.0.0 (published 3 September 2021), DOI `10.5281/zenodo.5427549` (concept DOI `10.5281/zenodo.5427548`), file `emoUERJ.zip` (105,084,861 B / 105.1 MB, `md5:7817ccc4cfba39ff85eba0b26a174af8`) (**VERIFIED**, [Zenodo API 5427549](https://zenodo.org/api/records/5427549)).
- **Licence (quoted verbatim):**
  > `"license": {"id": "cc-by-4.0"}` — [Zenodo API 5427549](https://zenodo.org/api/records/5427549)
  - **Licence class:** Open (CC BY 4.0) (**VERIFIED**).
- **Redistribution allowed:** Yes, with attribution under CC BY 4.0 (**VERIFIED**, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en)).
- **Access requirements:** None — direct public download from Zenodo without form, account, or agreement (**VERIFIED**).
- **Total size:** 105.1 MB (`105,084,861` bytes) (**VERIFIED**, [Zenodo API 5427549](https://zenodo.org/api/records/5427549)).
- **Clips & hours:** 377 WAV files: `happiness` (91), `anger` (94), `sadness` (100), `neutral` (92) (**VERIFIED**, [Zenodo 5427549](https://zenodo.org/records/5427549)). Total hours: **UNVERIFIED** (not stated on Zenodo page; archive not downloaded).
- **Speakers & per-clip speaker ID:** 8 actors (4 men `m01`–`m04`, 4 women `w01`–`w04`). **Yes, speaker identity is encoded in every filename** (`<gender><id><emotion><rec>`, e.g. `w04a11.wav`) (**VERIFIED**, [Zenodo 5427549](https://zenodo.org/records/5427549) and [Zenodo ZIP preview](https://zenodo.org/records/5427549/preview/emoUERJ.zip?include_deleted=0)).
- **Emotion labels (as named by source):** `happiness` (`h`), `anger` (`a`), `sadness` (`s`), `neutral` (`n`) (**VERIFIED**, [Zenodo 5427549](https://zenodo.org/records/5427549)).
- **Authenticity:** Acted (8 actors reading 10 predefined Portuguese sentences) (**VERIFIED**, [Zenodo 5427549](https://zenodo.org/records/5427549)).
- **Language variety:** Brazilian Portuguese (`pt-BR`, Rio de Janeiro / UERJ) (**VERIFIED**, [Zenodo 5427549](https://zenodo.org/records/5427549)).
- **Sample rate & format:** `.wav` files (**VERIFIED**, [Zenodo ZIP preview](https://zenodo.org/records/5427549/preview/emoUERJ.zip?include_deleted=0)); exact sample rate and bit depth **UNVERIFIED** (not stated on Zenodo record; archive not downloaded).
- **Known problems:**
  - Very small (377 clips, 8 actors) and covers only 4 emotions (missing `fear`, `disgust`, `surprise`, `calm`) (**VERIFIED**).
  - No official train/dev/test split provided (**VERIFIED**).
  - No perceptual validation study reported on the Zenodo page (**VERIFIED**).
- **Model rights (engineering fact):** CC BY 4.0 permits private use, commercial use, and publishing derived models, subject to attribution.
- **Verdicts:**
  - **D3(a):** `test only` — Only openly licensed (CC BY 4.0) Brazilian Portuguese dataset with named emotions and speaker IDs, but 377 clips / 8 actors is best used as a held-out test set.
  - **D3(b):** `test only` — Same.
  - **D3(c):** `test only` — Same.

---

### 6. VERBO (Voice Emotion Recognition dataBase in pOrtuguese language)

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Google Sites verbodatabase](https://sites.google.com/view/verbodatabase/), [GitHub jrtorresneto/VERBO-emotional-speech-dataset](https://github.com/jrtorresneto/VERBO-emotional-speech-dataset), paper [Torres Neto et al., 2018](https://thescipub.com/jcs/jcssp.2018.1420.1430) (DOI `10.3844/jcssp.2018.1420.1430`); Institute of Mathematical and Computer Sciences, University of São Paulo (ICMC-USP) (**VERIFIED**).
- **Version, DOI & checksum:** Paper DOI `10.3844/jcssp.2018.1420.1430` (accepted 2 November 2018; version of record 2 November 2018 on publisher page); dataset version and checksum **UNVERIFIED** (dataset not publicly hosted).
- **Licence (quoted verbatim):**
  > Paper licence only: `"© 2018 José R. Torres Neto, Geraldo P.R. Filho, Leandro Y. Mano and Jó Ueyama. This open access article is distributed under a Creative Commons Attribution (CC-BY) 3.0 license."` — [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)
  > Dataset access page: `"Request the database by filling out this form and sending it to the email [contact on download page]"` — [VERBO Download Page](https://sites.google.com/view/verbodatabase/home/download)
  > **Dataset licence text: UNVERIFIED** (tried GitHub repo — `"license": null` — Google Sites home and download pages, and the linked Google Drive form `https://drive.google.com/file/d/1X9zT9p2lF3m1dqjfJjnf0QXQ8GvRtzjm/view?usp=sharing`, which returned HTTP 404).
  - **Licence class:** On-request (exact terms UNVERIFIED).
- **Redistribution allowed:** **UNVERIFIED** (request form 404).
- **Access requirements:** Form + email request to the contact listed on the [VERBO Download Page](https://sites.google.com/view/verbodatabase/home/download) (**VERIFIED**); linked form is currently broken (404).
- **Total size:** **UNVERIFIED** (not stated in paper or website).
- **Clips & hours:** 1,167 recordings (note: paper states 14 sentences × 12 actors × 7 emotions = 1,176 on page 1425, but reports 1,167 recordings in Stage 5 and Table 2) (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)). Total hours: **UNVERIFIED**.
- **Speakers & per-clip speaker ID:** 12 professional actors (6 female `f1`–`f6`, 6 male `m1`–`m6`). **Yes, speaker ID is encoded in every filename** (`<emotion>-<speaker>-<phrase>.wav`, e.g. `ale-f1-s1.wav`) (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)).
- **Emotion labels (as named by source):** `happiness` (`ale`), `disgust`, `fear`, `neutral`, `anger`, `surprise`, `sadness` (7 categories) (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)).
- **Authenticity:** Acted (12 professional actors reading 14 semantically neutral sentences, modelled on Italian EMOVO) (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)).
- **Language variety:** Brazilian Portuguese (`pt-BR`) (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)).
- **Sample rate & format:** `.wav` files recorded on portable digital recorders (**VERIFIED**, [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)); exact sample rate and bit depth **UNVERIFIED**.
- **Known problems:**
  - Request form link on the official site is dead (HTTP 404) and GitHub repo has been untouched since January 2019 (**VERIFIED**).
  - In the authors' 3-psychologist validation, 283 of 1,167 clips (24.3%) were not recognized as the target emotion by any of the three judges; only 440 (37.7%) had unanimous agreement (**VERIFIED**, Table 2 of [Torres Neto et al., 2018 PDF](https://thescipub.com/pdf/jcssp.2018.1420.1430.pdf)).
  - Discrepancy between 1,176 (14 × 12 × 7) and 1,167 reported recordings (9 missing clips) (**VERIFIED**).
- **Model rights (engineering fact):** **UNVERIFIED** (dataset licence text unavailable).
- **Verdicts:**
  - **D3(a):** `reject` — On-request dataset with broken form link and unverified licence.
  - **D3(b):** `reject` — Dataset licence terms are UNVERIFIED (form 404).
  - **D3(c):** `reject` — Dataset licence terms are UNVERIFIED; revisit only if authors reply with written licence terms.

---

### 7. EmoFilm

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Zenodo 7665999 (v3)](https://zenodo.org/records/7665999), [Interspeech 2018 paper](https://www.isca-archive.org/interspeech_2018/paradacabaleiro18_interspeech.html); University of Augsburg & University of Rome Tor Vergata (**VERIFIED**).
- **Version, DOI & checksum:** Version v3 (updated 12 July 2024), DOI `10.5281/zenodo.7665999` (concept DOI `10.5281/zenodo.1326427`); file checksum **UNVERIFIED** (files are restricted on Zenodo and hidden until access is granted).
- **Licence (quoted verbatim):**
  > "1. Commercial use
  > The user may not use the corpus for any non-academic purpose. Non-academic purposes include, but are not limited to:
  > • proving the efficiency of commercial systems
  > • training or testing of commercial systems
  > • selling data from the dataset
  > • creating military applications
  > • developing governmental systems used in public spaces
  > 2. Responsibility
  > This document must be signed by a person with a permanent position at an academic institute (the signee). Up to four other researchers affiliated with the same institute for whom the signee is responsible may be named at the end of this document which will allow them to work with EmoFilm.
  > 3. Distribution
  > The user may not distribute the corpus or portions of it in any way, with the exception of using small portions of data for the exclusive purpose of clarifying academic publications or presentations. […]
  > […]
  > 6. Academic research
  > The user may use EmoFilm only for academic research." — [EmoFilm EULA PDF](https://drive.google.com/file/d/1pFHfsqk7snF_EVqq0WAC0Dz8FcTD3s9_/view?usp=share_link) (linked from [Zenodo 7665999](https://zenodo.org/records/7665999))
  - **Licence class:** Academic-only (**VERIFIED**).
- **Redistribution allowed:** No (except small excerpts for clarifying academic publications/presentations) (**VERIFIED**, EULA Article 3).
- **Access requirements:** Signed EULA by a person with a permanent position at an academic institute (typically a full professor), submitted from an institutional email address via Zenodo / email to the contact listed on [Zenodo 7665999](https://zenodo.org/records/7665999); non-institutional emails are automatically rejected (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999)).
- **Total size:** **UNVERIFIED** (restricted Zenodo record hides file sizes; ~1,115 clips × ~3.5 s at 48 kHz 16-bit mono ≈ ~375 MB raw audio).
- **Clips & hours:** 1,115 clips total (mean 3.5 s, std 1.2 s; ~65 min total): 360 English, 413 Italian, **342 Spanish** (165 female, ~20 min Spanish) (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999), [Parada-Cabaleiro et al., 2018 PDF](https://www.isca-archive.org/interspeech_2018/paradacabaleiro18_interspeech.pdf)).
- **Speakers & per-clip speaker ID:** 207 speakers total; **57 Spanish dubbing speakers** (24 female, 33 male) (**VERIFIED**, [Parada-Cabaleiro et al., 2018 PDF](https://www.isca-archive.org/interspeech_2018/paradacabaleiro18_interspeech.pdf)). Per-clip speaker ID availability in the distributed metadata: **UNVERIFIED** (archive is gated; paper notes the same dubbing actor can dub multiple original actors across films, and EMOVOME Table II cites SI experiments on EmoFilm).
- **Emotion labels (as named by source):** `anger`, `contempt`, `happiness`, `fear`, `sadness` (5 categories; **no `neutral`, `disgust`, or `surprise`**) (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999)).
- **Authenticity:** Acted (over-dubbed film dialogue from 43 English-original films) (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999)).
- **Language variety:** Multilingual — English, Italian, Spanish (dubbed Spanish; exact regional dubbing variety not specified in paper, **UNVERIFIED**).
- **Sample rate & format:** Uncompressed mono WAV, 48 kHz, 16-bit (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999)).
- **Known problems:**
  - Strict institutional gate (non-academic users automatically rejected) (**VERIFIED**).
  - Pre-test filtering was performed by 10 Italian listeners only; no `neutral` class (**VERIFIED**, [Parada-Cabaleiro et al., 2018 PDF](https://www.isca-archive.org/interspeech_2018/paradacabaleiro18_interspeech.pdf)).
  - Version 3 notes a typo fix in the README concerning emotion labels (**VERIFIED**, [Zenodo 7665999](https://zenodo.org/records/7665999)).
- **Model rights (engineering fact):** Strictly academic research only; training or testing commercial systems is explicitly prohibited; private non-academic use is prohibited.
- **Verdicts:**
  - **D3(a):** `reject` — Restricted academic-only EULA.
  - **D3(b):** `reject` — EULA explicitly prohibits any non-academic purpose and requires a permanent academic signatory.
  - **D3(c):** `test only` — Permitted only under an eligible academic signatory; 342 Spanish clips (missing `neutral`) suits held-out testing once per-clip speaker IDs are confirmed.

---

### 8. INTERFACE Spanish (ELRA-S0329)

- **SER catalogue ID:** Not in catalogue. *(This is the ELRA-distributed Spanish emotional speech database created under the EU INTERFACE project.)*
- **Official URL & institution:** [ELRA Catalogue ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/) (ISLRN `477-238-467-792-9`); distributed by ELDA / ELRA, created within the EU-funded INTERFACE project (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)).
- **Version, DOI & checksum:** Version 1.0 (last updated 2012-05-01; catalogue displays dd/mm `01/05/2012`); no DOI or checksum listed (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)).
- **Licence (quoted verbatim):**
  > Catalogue licensing table: `"Licence: Non Commercial Use - ELRA END USER"` (Academic: 0.00 €; Commercial: 2000.00 € Member / 4000.00 € Non-Member) and `"Licence: Commercial Use - ELRA VAR"` (2000.00 € Member / 4000.00 € Non-Member) — [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)
  > ELRA End-User Agreement clauses 7–9: `"7. END-USER is not permitted to distribute and market any derivative product, based on all or a substantial part of the Language Resources. 8. END-USER is also not permitted to distribute and market all or any part of Language Models and/or services or any product derived from such Language Model. 9. END-USER is not permitted to make available to the public all or any substantial part of the contents of the Language Resources, or any derived Language Models, evaluated quantitatively and/or qualitatively, by the distribution of copies, by renting, leasing or any other form of distribution, including free or open-source ones."` — [ELRA End-User Template](http://www.elra.info/media/filer_public/2022/01/21/enduser_220121.pdf)
  > ELRA VAR Agreement clause 10: `"10. VAR is not permitted to make available to third parties Language Models based on all or any part of the Language Resources. VAR is not permitted to make available to third parties Language Models created by incorporation of the Language Resources in a larger dataset. […]"` — [ELRA VAR Template](http://www.elra.info/media/filer_public/2022/01/21/var_220121.pdf)
  - **Licence class:** Non-commercial (0 EUR for academic institutions; 2,000–4,000 EUR for non-academic/commercial organisations under ELRA END USER) or paid Commercial (ELRA VAR) (**VERIFIED**).
- **Redistribution allowed:** No (explicitly prohibited in both ELRA End-User and VAR agreements) (**VERIFIED**).
- **Access requirements:** ELRA account, signed bilateral agreement (ELRA End-User or VAR), and fee payment for non-academic entities (**VERIFIED**, [ELRA Pricing](https://catalogue.elra.info/en-us/pricing-info/)).
- **Total size:** 2 GB listed under language variety (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)).
- **Clips & hours:** 3 h 59 min (male) + 3 h 53 min (female) = 7 h 52 min total (catalogue summary field says "4 Hours", likely per speaker); 184 text items across neutral + 6 MPEG expressions × styles (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)). Secondary survey ([EMOVOME paper Table II](https://arxiv.org/html/2403.02167v3)) reports 5,520 clips; exact clip count at ELRA source is **UNVERIFIED**.
- **Speakers & per-clip speaker ID:** **2 professional speakers only** (1 male, 1 female) (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)). Per-clip speaker ID is trivially separable by male/female voice.
- **Emotion labels (as named by source):** `"neutral style plus six MPEG expressions, all in fast, slow, soft and loud speech styles"` (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)). Exact label strings for the six MPEG-4 expressions in the archive metadata: **UNVERIFIED** (archive not downloaded).
- **Authenticity:** Acted (read text in a noise-reduced room for speech synthesis) (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)).
- **Language variety:** Castilian Spanish (`es-ES`) (**VERIFIED**, [ELRA-S0329](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0329/)).
- **Sample rate & format:** **UNVERIFIED** (not stated on the ELRA-S0329 catalogue page).
- **Known problems:**
  - Only 2 speakers (1 male, 1 female) — cannot form speaker-disjoint train/dev/test splits and strongly overfits to speaker timbre (**VERIFIED**).
  - Designed for expressive TTS, not SER; includes artificial speed/loudness variants (**VERIFIED**).
  - ELRA contracts explicitly prohibit publishing or sharing trained models (even free/open-source) (**VERIFIED**).
- **Model rights (engineering fact):** ELRA END USER prohibits publishing or commercializing any derived Language Model; ELRA VAR prohibits sharing Language Models with third parties even when mixed into a larger dataset.
- **Verdicts:**
  - **D3(a):** `reject` — Proprietary ELRA licence; fewer than 4 speakers (2 only).
  - **D3(b):** `reject` — Fewer than 4 speakers (2 only); requires 4,000 EUR non-academic ELRA contract and bars model publishing.
  - **D3(c):** `reject` — Fewer than 4 speakers (1M, 1F) fails the minimum-speaker threshold for speaker-disjoint evaluation.

---

### 9. EMOVOME (Emotional Voice Messages)

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Zenodo 10694370 (v2)](https://zenodo.org/records/10694370), paper [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3); HUMAN-tech Institute, Universitat Politècnica de València (UPV, Spain) (**VERIFIED**).
- **Version, DOI & checksum:** Version v2 (published 22 February 2024), DOI `10.5281/zenodo.10694370` (concept DOI `10.5281/zenodo.6453063`), file `emovome.zip` (1,329,352,216 B / 1.3 GB, `md5:f5d9a21ed4852cc5e352a711a144699e`) (**VERIFIED**, [Zenodo API 10694370](https://zenodo.org/api/records/10694370)).
- **Licence (quoted verbatim):**
  > Zenodo metadata: `"license": {"id": "cc-by-4.0"}` — [Zenodo API 10694370](https://zenodo.org/api/records/10694370)
  > Usage Notes (governing raw audio): `"All the data included in the EMOVOME database is publicly available under the Creative Commons Attribution 4.0 International license. The only exception is the original raw audio files, for which an additional step is required as a security measure to safeguard the speakers' privacy. To request access, interested authors should first complete and sign the agreement file EMOVOME_agreement.pdf and send it to the corresponding author (…)." […] "The data included in the EMOVOME database is expected to be used for research purposes only. Therefore, the agreement file states that the authors are not allowed to share the data with profit-making companies or organisations. They are also not expected to distribute the data to other research institutions; instead, they are suggested to kindly refer interested colleagues to the corresponding author of this article. By agreeing to the terms of the agreement, the authors also commit to refraining from publishing the audio content on the media (such as television and radio), in scientific journals (or any other publications), as well as on other platforms on the internet. The agreement must bear the signature of the legally authorised representative of the research institution (e.g., head of laboratory/department)."` — [Zenodo 10694370](https://zenodo.org/records/10694370)
  > *(Full text of `EMOVOME_agreement.pdf` inside `emovome.zip` is UNVERIFIED because downloading the 1.3 GB archive is outside limits.)*
  - **Licence class:** **Academic-only** for raw audio (requires institutional research-head signatory and bars profit-making organisations); Open (CC BY 4.0) for metadata, labels, transcriptions, and eGeMAPS features only (**VERIFIED**).
- **Redistribution allowed:** Metadata/features: Yes (CC BY 4.0). Raw audio: **No** (explicitly prohibited from sharing with profit-making organisations, other research institutions, or publishing online) (**VERIFIED**, [Zenodo 10694370](https://zenodo.org/records/10694370)).
- **Access requirements:** Signed `EMOVOME_agreement.pdf` bearing the signature of a legally authorised representative of a research institution (e.g., head of laboratory/department), emailed to the corresponding author listed on [Zenodo 10694370](https://zenodo.org/records/10694370) (**VERIFIED**).
- **Total size:** `emovome.zip` (metadata, features, baseline models) = 1.3 GB; raw `Audios` folder (delivered separately) ≈ 213 MB (**VERIFIED**, [Zenodo 10694370](https://zenodo.org/records/10694370)).
- **Clips & hours:** 999 spontaneous voice messages (293 min ≈ 4.88 h) + 100 text-reading recordings (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
- **Speakers & per-clip speaker ID:** 100 Spanish speakers (50 female, 50 male, balanced across 4 age groups 18–55). **Yes, speaker ID per clip is provided** in `participants_ids.csv`, and an official speaker-independent 80/20 split (`partitions.csv`: 80 dev / 20 test speakers) is included (**VERIFIED**, [Zenodo 10694370](https://zenodo.org/records/10694370)).
- **Emotion labels (as named by source):** Valence (3-class: `negative`, `neutral`, `positive`; 5-point SAM) and Arousal (3-class: `low`, `neutral`, `high`; 5-point SAM) from 3 non-experts + 2 clinical psychologists; **plus 7 discrete emotion categories** (`happiness`, `disgust`, `anger`, `surprise`, `fear`, `sadness`, `neutral`) labeled by the expert clinical psychologists (each rating half of the audios) (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
- **Authenticity:** Natural / spontaneous (real pre-existing WhatsApp voice messages recorded in-the-wild before recruitment) (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
- **Language variety:** Spanish (Spain; recruited via UPV Valencia study) (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
- **Sample rate & format:** Ogg Vorbis, 16-bit, 44.1 kHz or 48 kHz (**VERIFIED**, [Zenodo 10694370](https://zenodo.org/records/10694370)).
- **Known problems:**
  - Low inter-annotator agreement on arousal (Cohen's κ = 0.075–0.407; valence κ = 0.614–0.754); each clip's discrete category was rated by a single expert psychologist (each rated half) (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
  - Discrete emotion recognition is challenging (best model UniSpeech-SAT-Large reached 42.58% UA) (**VERIFIED**, [arXiv:2403.02167v3](https://arxiv.org/html/2403.02167v3)).
  - Mixed sample rates (44.1 kHz and 48 kHz) in lossy Ogg Vorbis format (**VERIFIED**).
- **Model rights (engineering fact):** Usage Notes restrict raw audio to research purposes only under a research-institution signatory and bar sharing with profit-making organisations; exact clause on trained model weights inside `EMOVOME_agreement.pdf` is **UNVERIFIED**.
- **Verdicts:**
  - **D3(a):** `reject` — Raw audio is excluded from CC BY 4.0 and gated behind a research-only institutional agreement.
  - **D3(b):** `reject` — Requires signature of a legally authorised representative of a research institution and excludes profit-making organisations.
  - **D3(c):** `train+test` — High-value spontaneous real-life Spanish corpus (999 clips, 100 speakers, official speaker-disjoint 80/20 split, 7 discrete categories), usable if an eligible research institution signs the agreement.

---

### 10. EmoSpanishDB / EmoMatchSpanishDB

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [figshare 14215850 (v3)](https://doi.org/10.6084/m9.figshare.14215850) ([figshare API](https://api.figshare.com/v2/articles/14215850)), paper [Garcia-Cuesta et al., 2023/2024](https://link.springer.com/article/10.1007/s11042-023-15959-w) (*Multimedia Tools and Applications*, 83:13093–13112); Universidad Europea de Madrid & Universidad Politécnica de Madrid (UPM) (**VERIFIED**).
- **Version, DOI & checksum:** Figshare Version 3 (published 2023-06-07), DOI `10.6084/m9.figshare.14215850.v3`, paper DOI `10.1007/s11042-023-15959-w`. Posted files on figshare: `EMOMATCHAgreement.docx` (13,862 B, `md5:3ea06889fddafe38a2cc99d748aa0b4e`), `EmoMatchSpanishDB-Features.zip` (21.35 MB, `md5:8da13b3624ee7665fb5ffb0ff23c0b89`), `EmoSpanishDB-features.zip` (37.32 MB, `md5:c0763d5aeec3e493e9b41329193cdd8b`), `EMOMATCHSpanishDB-COMPARE-features.zip` (1.25 GB, `md5:8aaae5a0f4d5ee3df99fd31776cd9b44`). Raw audio archive checksum: **UNVERIFIED** (delivered via private download link after signing the agreement) (**VERIFIED**, [figshare API 14215850](https://api.figshare.com/v2/articles/14215850)).
- **Licence (quoted verbatim):**
  > Posted feature files licence: `"MIT"` (`https://opensource.org/licenses/MIT`) — [figshare API 14215850](https://api.figshare.com/v2/articles/14215850)
  > Figshare description: `"The features are available under MIT license and if you want to get access to the original raw audio files for creating your own features and research purposes you can get them under CC-BY-NC completing and signing the agreement file (EMOMATCHAgreement.docx) and sending it via email to [author contact in agreement]"` — [figshare API 14215850](https://api.figshare.com/v2/articles/14215850)
  > Raw audio agreement (`EMOMATCHAgreement.docx`):
  > `"The EMOMATCH is to be used exclusively in academic, not-for-profit research. It is expected that the audios are used only in basic and health research projects. In accepting these materials, researchers are expected not to provide them to profit making companies or organizations. Furthermore, they are expected not to make them available to the media (television, magazines, etc.), to publish the audios (including in scientific journals or any other publication) or to place them on the internet."`
  > `"Finally, the EMOMATCH is given specifically for use in your laboratory with research purposes. We appreciate you’re not distributing it to other individuals or laboratories, but instead, please refer interested colleagues to us."`
  > `"Please send us back this agreement signed to [author contact in agreement] and we will send you a link to download the raw original audios."` — [EMOMATCHAgreement.docx (figshare)](https://ndownloader.figshare.com/files/34319516)
  - **Licence class:** **Academic-only** for raw audio (signed agreement requiring university affiliation; MIT for posted feature files only; figshare description also mentions CC-BY-NC alongside the academic-only agreement) (**VERIFIED**).
- **Redistribution allowed:** Features: Yes (MIT). Raw audio: **No** (agreement explicitly prohibits providing materials to profit-making companies/organisations, distributing to other individuals/laboratories, or publishing/placing audios on the internet) (**VERIFIED**, [EMOMATCHAgreement.docx](https://ndownloader.figshare.com/files/34319516)).
- **Access requirements:** Complete and sign `EMOMATCHAgreement.docx` (fields: Researcher Full Name, Position, University, E-mail contact, Signature) and email it to the corresponding author listed in the agreement to receive the raw audio download link (**VERIFIED**, [EMOMATCHAgreement.docx](https://ndownloader.figshare.com/files/34319516)).
- **Total size:** Posted feature archives total 1.31 GB on figshare; raw audio archive size **UNVERIFIED** (delivered by link after signing).
- **Clips & hours:** EmoSpanishDB: 3,550 clips; EmoMatchSpanishDB (consensus-matched subset): 2,020 clips in the Springer paper (figshare description writes 2,050; thesis repo [andsanjuan](https://raw.githubusercontent.com/andsanjuan/spanish-speech-emotion-recognition/main/README.md) reports 2,005); sentences are approximately 2 s each (~1.1–2.0 h estimated, exact total hours **UNVERIFIED**) (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w), [figshare API 14215850](https://api.figshare.com/v2/articles/14215850)).
- **Speakers & per-clip speaker ID:** 50 non-actors (30 male, 20 female) reproducing 12 semantically neutral sentences. **Yes, per-clip speaker ID is provided** (paper and downstream benchmarks run Leave-One-Speaker-Out evaluation across all 50 speakers) (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w), [EMOVOME Table II](https://arxiv.org/html/2403.02167v3)).
- **Emotion labels (as named by source):** `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral` (7 categories) (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w)).
- **Authenticity:** Elicited in a professional radio studio (non-actors induced via image + short text, validated by 194 crowdsourced native Spanish listeners) (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w)).
- **Language variety:** Spanish (Spain; Universidad Europea de Madrid / UPM) (**VERIFIED**).
- **Sample rate & format:** Uncompressed mono PCM WAV, 48 kHz, 16-bit (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w)).
- **Known problems:**
  - Raw audio is gated behind an academic-only signed agreement requiring university affiliation (**VERIFIED**).
  - Minor clip-count discrepancy across sources for EmoMatchSpanishDB (2,020 in Springer paper vs 2,050 in figshare description vs 2,005 in thesis repo) (**VERIFIED**).
  - EmoSpanishDB contains noisy/mismatched labels (e.g. only 28.3% match rate for `disgust` vs 86% for `neutral`), which is why the authors curated EmoMatchSpanishDB (**VERIFIED**, [Springer full text](https://link.springer.com/article/10.1007/s11042-023-15959-w)).
- **Model rights (engineering fact):** Agreement restricts raw audio "exclusively in academic, not-for-profit research" and "only in basic and health research projects", and bars sharing with profit-making companies/organisations; silent on publishing trained weights.
- **Verdicts:**
  - **D3(a):** `reject` — Raw audio requires an academic-only signed agreement (MIT covers posted features only).
  - **D3(b):** `reject` — Raw audio agreement is academic-only (requires university affiliation and excludes profit-making organisations).
  - **D3(c):** `train+test` — Largest elicited Spanish 7-class corpus (50 speakers, per-clip LOSO, 48 kHz 16-bit PCM WAV), usable if an eligible academic researcher signs `EMOMATCHAgreement.docx`.

---

### 11. Basis Global Emotions

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Hugging Face basis-ai/basis-global-emotions](https://huggingface.co/datasets/basis-ai/basis-global-emotions), [Basis landing page](https://withbasis.co/open/global-emotions); Basis (**VERIFIED**).
- **Version, DOI & checksum:** Published October 2026 (`sha` `57f26ff90a6240e54df94a2fabaf0f744df02575` on HF API); `CHECKSUMS.sha256` and per-shard SHA-256 listed in the repo (contents behind HF gate) (**VERIFIED**, [HF API](https://huggingface.co/api/datasets?search=emo&filter=modality:audio&limit=20) & [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Licence (quoted verbatim):**
  > Licence identifier: `"basis-data-license-1.0"` — [HF LICENSE page metadata](https://huggingface.co/datasets/basis-ai/basis-global-emotions/blob/main/LICENSE)
  > Dataset card terms summary: `"The Basis Data License grants a worldwide, royalty-free, perpetual license to download, store, process, annotate, modify and combine the data, and to pretrain, train, fine-tune, distill, evaluate and benchmark models on it, for research and commercial purposes. Models and results built on it are yours: you may publish, distribute, license and sell them, and operate commercial products, hosted inference services and APIs with them, with no royalties, no revenue sharing and no further permission from Basis, and no obligation to publish code, weights or methods. Two things are not allowed: reselling or redistributing the recordings themselves (or copies and derivatives that contain them), and identifying, locating or contacting speakers, impersonating a specific person, or reconstructing removed information."` — [HF dataset card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)
  > **Full `LICENSE` file text (16.6 kB): UNVERIFIED** (tried `https://huggingface.co/datasets/basis-ai/basis-global-emotions/raw/main/LICENSE`, which returned HTTP 401 because the repo is manually gated on Hugging Face).
  - **Licence class:** Custom permissive (commercial + research model use allowed; raw audio redistribution prohibited; manual gate) (**VERIFIED** from card summary; full legal text **UNVERIFIED**).
- **Redistribution allowed:** **No** for raw recordings or derivatives containing them; **Yes** for trained models and results (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Access requirements:** Hugging Face account + manual access-request review by a person + agreement to terms in `LICENSE` (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Total size:** 168 GB for all 108 locales; per-locale parquet shards can be loaded individually (`es-*` and `pt-*` subset size **UNVERIFIED**, ~13% of total hours) (**VERIFIED** total size from [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Clips & hours:** Full release: 525.17 h, 137,696 recordings, 10,420 speakers, 108 locales.
  - **Spanish (9 locales, 54.69 h, 14,320 recordings, 862 speakers):** `es-CL` (8.06 h / 2,199 / 81 spk), `es-MX` (7.73 h / 2,092 / 86 spk), `es-PE` (7.31 h / 1,940 / 125 spk), `es-ES` (6.54 h / 1,671 / 128 spk), `es-GT` (6.39 h / 1,687 / 117 spk), `es-EC` (5.73 h / 1,381 / 68 spk), `es-CO` (5.16 h / 1,333 / 82 spk), `es-AR` (4.27 h / 1,106 / 70 spk), `es-DO` (3.50 h / 911 / 105 spk).
  - **Portuguese (2 locales, 15.06 h, 3,771 recordings, 134 speakers):** `pt-BR` (9.97 h / 2,382 / 80 spk), `pt-PT` (5.09 h / 1,389 / 54 spk) (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Speakers & per-clip speaker ID:** **Yes** — `speaker_id` per row, joined to `speakers.jsonl` with salted stable anonymous IDs explicitly designed for speaker-disjoint splits (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Emotion labels (as named by source):** 38 `steer` values (23 emotions + 15 delivery styles) plus `neutral` intensity in every series: emotions include `angry`, `afraid`, `disgusted`, `happy`, `sad`, `surprised`, `calm`, `bored`, `amused`, `frustrated`, `excited`, `affectionate`, `apologetic`, `confused`, `curious`, `disappointed`, `embarrassed`, `envious`, `grateful`, `guilty`, `nostalgic`, `proud`, `relieved`; delivery styles include `assertive`, `crying`, `dramatic`, `encouraging`, `flirtatious`, `hesitant`, `laughing`, `mumbling`, `playful`, `sarcastic`, `screaming`, `shouting`, `singing`, `soothing`, `whispering` (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Authenticity:** Acted / steered (paid native speakers reading scripted lines in described situations at neutral/medium/full intensities) (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Language variety:** 9 Spanish locales (Chile, Mexico, Peru, Spain, Guatemala, Ecuador, Colombia, Argentina, Dominican Republic) and 2 Portuguese locales (Brazil, Portugal) (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Sample rate & format:** WAV, 16-bit PCM, 48 kHz, mono (raw unenhanced recordings stored in parquet) (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
- **Known problems:**
  - Full `LICENSE` text cannot be inspected prior to HF account login / access approval (**VERIFIED**).
  - Scripted acted speech (not spontaneous); `given_line` is the prompt, not a human-validated transcript (**VERIFIED**, [HF card](https://huggingface.co/datasets/basis-ai/basis-global-emotions)).
  - Mixes 23 emotions and 15 delivery styles — requires filtering/mapping to SER's 8-class ontology (neutral + 7 basic emotions + calm) (**VERIFIED**).
- **Model rights (engineering fact):** Per the dataset card, models trained on the data may be used privately, used commercially, and published/sold with no royalties or obligation to publish code/weights (must be confirmed against the full `LICENSE` text upon access).
- **Verdicts:**
  - **D3(a):** `reject` — Custom no-redistribution licence with manual gate (not CC BY/CC0/ODbL) and full `LICENSE` text is behind the HF login gate.
  - **D3(b):** `train+test` — Conditional exception to the unverified-licence reject rule: verified official card states permissive commercial and research model rights for 54.69 h Spanish (9 locales, 862 speakers) and 15.06 h Portuguese (`pt-BR` + `pt-PT`, 134 speakers); confirm full `LICENSE` upon HF approval before downloading.
  - **D3(c):** `train+test` — Same as D3(b).

---

### 12. CMU-MOSEAS

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [ACL Anthology 2020.emnlp-main.141](https://aclanthology.org/2020.emnlp-main.141/), [GitHub CMU-MultiComp-Lab/CMU-MultimodalSDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/blob/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEAS/Europe/readme.md); Carnegie Mellon University (LTI / SCS) (**VERIFIED**).
- **Version, DOI & checksum:** Paper DOI `10.18653/v1/2020.emnlp-main.141` (November 2020); dataset checksum **UNVERIFIED**.
- **Licence (quoted verbatim):**
  > SDK code licence only:
  > ```text
  > MIT License
  >
  > Copyright (c) 2023 CMU-MultiComp-Lab
  > ```
  > — [CMU-MultimodalSDK LICENSE](https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/LICENSE)
  > Paper Section 3.2: `"Even though videos are publicly available on YouTube, a specific EULA (End User License Agreement) is required to download the labels (to see the EULA, please refer to supplementary). Non-invertible high-level computational features are provided publicly online."` — [CMU-MOSEAS EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)
  > **EULA text: UNVERIFIED** (not present in the GitHub repo; supplementary archive not linked on ACL Anthology page).
  - **Licence class:** On-request (EULA text UNVERIFIED).
- **Redistribution allowed:** **UNVERIFIED** (EULA not publicly readable).
- **Access requirements:** Raw files (~640 GB) require emailing the contact listed in the [CMU-MOSEAS Europe README](https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEAS/Europe/readme.md); labels require signing an EULA (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)).
- **Total size:** ~640 GB for raw multimodal files across all 4 languages (**VERIFIED**, [CMU-MOSEAS Europe README](https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/main/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEAS/Europe/readme.md)).
- **Clips & hours:** 40,000 annotated sentences total (68 h 49 min across 4 languages): **10,000 Spanish** (avg 6.70 s/sentence) and **10,000 Portuguese** (avg 5.77 s/sentence) (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf) Tables 1–2).
- **Speakers & per-clip speaker ID:** 341 distinct Spanish speakers, 399 distinct Portuguese speakers (max 5 videos per YouTube channel; distinct speakers identified via face identification in §3.6, so a video ID is not a unique speaker ID) (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)). Gender distribution is not provided (**VERIFIED**, [EMOVOME paper](https://arxiv.org/html/2403.02167v3)).
- **Emotion labels (as named by source):** 6 Ekman emotions on Likert intensity scales (`Happiness`, `Sadness`, `Anger`, `Fear`, `Disgust`, `Surprise`), plus Sentiment (`[-3, 3]`), Subjectivity (binary), and 12 personality/delivery attributes (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)).
- **Authenticity:** Natural / spontaneous (YouTube monologue videos) (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)).
- **Language variety:** European Spanish (`es-ES`, region Spain) and European Portuguese (`pt-PT`, region Portugal) (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf) Section 3.1).
- **Sample rate & format:** Public SDK provides non-invertible COVAREP acoustic features only (no raw audio); raw audio format **UNVERIFIED**.
- **Known problems:**
  - Public download contains only non-invertible features, not raw audio required by SER's encoders (**VERIFIED**, [EMNLP 2020 PDF](https://aclanthology.org/2020.emnlp-main.141.pdf)).
  - EULA text is not publicly accessible; YouTube copyright applies (**VERIFIED**).
- **Model rights (engineering fact):** **UNVERIFIED** (EULA unavailable).
- **Verdicts:**
  - **D3(a):** `reject` — Requires unverified EULA and email request for raw audio.
  - **D3(b):** `reject` — Public release lacks raw audio; EULA text is UNVERIFIED.
  - **D3(c):** `reject` — EULA text is UNVERIFIED and 640 GB raw archive requires author approval.

---

### 13. MOUD (Multimodal Opinion Utterances Dataset)

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Rada Mihalcea Downloads](https://web.eecs.umich.edu/~mihalcea/downloads.html#MOUD), [ACL 2013 paper](https://aclanthology.org/P13-1096.pdf); University of North Texas / University of Michigan & USC ICT (**VERIFIED**).
- **Version, DOI & checksum:** Released 15 August 2013 (`MOUD.tar.gz`); no version number or checksum published (**VERIFIED**, [Downloads page](https://web.eecs.umich.edu/~mihalcea/downloads.html)).
- **Licence (quoted verbatim):**
  > Page header: `"Various software modules and data sets that are/were used in my research. They are made available under the terms of GNU General Public License. Both data and software are distributed without any warranty."` — [Rada Mihalcea Downloads](https://web.eecs.umich.edu/~mihalcea/downloads.html)
  - **Licence class:** Open (GNU GPL stated on page, though GPL is a software copyleft licence and underlying YouTube video copyrights are not addressed) (**VERIFIED**).
- **Redistribution allowed:** Under GPL with copyleft, subject to unresolved YouTube video creator copyrights (**VERIFIED**).
- **Access requirements:** None — direct download link (`MOUD.tar.gz`) (**VERIFIED**).
- **Total size:** "over 500M" (**VERIFIED**, [Downloads page](https://web.eecs.umich.edu/~mihalcea/downloads.html)).
- **Clips & hours:** ~400–550 utterances, ~59 min total (**VERIFIED** via [CMU-MOSEAS Table 1](https://aclanthology.org/2020.emnlp-main.141.pdf) and [EMOVOME Table II](https://arxiv.org/html/2403.02167v3)).
- **Speakers & per-clip speaker ID:** 101–105 speakers (heavily female-skewed: 84F / 21M per [EMOVOME Table II](https://arxiv.org/html/2403.02167v3)); per-clip speaker ID via video ID (**VERIFIED**).
- **Emotion labels (as named by source):** Sentiment polarity only: `pos` (positive), `neg` (negative), `neu` (neutral) (**VERIFIED**, [Pérez-Rosas et al., 2013 PDF](https://aclanthology.org/P13-1096.pdf)). **No categorical emotion labels.**
- **Authenticity:** Natural (YouTube product review monologues) (**VERIFIED**, [Pérez-Rosas et al., 2013 PDF](https://aclanthology.org/P13-1096.pdf)).
- **Language variety:** Spanish (YouTube; mixed varieties) (**VERIFIED**).
- **Sample rate & format:** **UNVERIFIED** (archive >500 MB, not downloaded).
- **Known problems:** Sentiment polarity only (not categorical emotions); 80% female speaker skew; YouTube copyright status unclear (**VERIFIED**).
- **Model rights (engineering fact):** Page-level GNU GPL notice is ambiguous for datasets and does not clear underlying YouTube creator rights.
- **Verdicts:**
  - **D3(a):** `reject` — Sentiment polarity only (`pos`/`neg`/`neu`), no categorical emotion labels, unclear YouTube rights.
  - **D3(b):** `reject` — Sentiment polarity only.
  - **D3(c):** `reject` — Sentiment polarity only.

---

### 14. LAION Artificial Voice Roleplay Dataset

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** [Hugging Face laion/emotional-roleplay-finetuning-dataset](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset); LAION (**VERIFIED**).
- **Version, DOI & checksum:** HF commit `68acdac74d30c12140fe3ebb9a5db30c7247cf1c` (July 2026); no DOI (**VERIFIED**, [HF API](https://huggingface.co/api/datasets?search=emo&filter=modality:audio&limit=20)).
- **Licence (quoted verbatim):**
  > `"CC-BY-4.0. All audio is newly synthesized (no recorded audio from any source corpus is redistributed); captions/transcripts are short automatic text. The generating model is an Apache-2.0 base model fine-tune."` — [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)
  - **Licence class:** Open (CC BY 4.0) (**VERIFIED**).
- **Redistribution allowed:** Yes, with attribution under CC BY 4.0 (**VERIFIED**).
- **Access requirements:** None — public Hugging Face download (**VERIFIED**).
- **Total size:** 3.13 GB (WebDataset, all 4 languages) (**VERIFIED**, [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)).
- **Clips & hours:** 67,491 clips (~184 h) total across DE/EN/ES/FR; **2,106 Spanish clips** (hours for ES subset **UNVERIFIED**) (**VERIFIED**, [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)).
- **Speakers & per-clip speaker ID:** No human speakers — 100% synthetic TTS (`MOSS-TTS-Local v1.5`), ~40% non-human fantasy/creature voices; no speaker IDs (**VERIFIED**, [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)).
- **Emotion labels (as named by source):** Free-text `voice_description` generated automatically by `gemini-3.5-flash` (plus mostly-null `req_emotion`); no fixed human-verified categorical emotion taxonomy (**VERIFIED**, [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)).
- **Authenticity:** Synthetic (machine-generated TTS) (**VERIFIED**).
- **Language variety:** Multilingual (DE, EN, ES, FR; Spanish variety unspecified) (**VERIFIED**).
- **Sample rate & format:** Mono MP3, 24 kHz (**VERIFIED**, [HF README](https://huggingface.co/datasets/laion/emotional-roleplay-finetuning-dataset/raw/main/README.md)).
- **Known problems:** Fully synthetic TTS with exaggerated creature/roleplay voices; automatic LLM captions; no speaker IDs (**VERIFIED**).
- **Model rights (engineering fact):** CC BY 4.0 permits private, commercial, and public model use with attribution.
- **Verdicts:**
  - **D3(a):** `reject` — Synthetic TTS (including creature voices), no speaker IDs, and no human-verified categorical emotion labels.
  - **D3(b):** `reject` — Same.
  - **D3(c):** `reject` — Same.

---

### 15. RekEmozio

- **SER catalogue ID:** Not in catalogue.
- **Official URL & institution:** University of the Basque Country (UPV/EHU, Aholab); official pages (`https://aholab.ehu.eus/aholab/rekemozio.html`, `https://aholab.ehu.eus/rekemozio/`) returned HTTP 404 (**UNVERIFIED** at source; referenced in [EMOVOME paper Table II](https://arxiv.org/html/2403.02167v3)).
- **Version, DOI & checksum:** **UNVERIFIED** (official pages 404).
- **Licence (quoted verbatim):** **UNVERIFIED** (tried Aholab URLs [404], Zenodo search [0 hits], GitHub search [0 hits]; EMOVOME Table II lists `Free: No`).
  - **Licence class:** On-request / Not free (**UNVERIFIED**).
- **Redistribution allowed:** **UNVERIFIED**.
- **Access requirements:** **UNVERIFIED**.
- **Total size:** **UNVERIFIED**.
- **Clips & hours:** 2,618 Spanish clips (secondary source: [EMOVOME paper Table II](https://arxiv.org/html/2403.02167v3)); total hours **UNVERIFIED**.
- **Speakers & per-clip speaker ID:** 10 speakers (5M, 5F) per [EMOVOME Table II](https://arxiv.org/html/2403.02167v3); per-clip speaker ID **UNVERIFIED** (SI results reported in literature).
- **Emotion labels (as named by source):** 7 discrete emotions per [EMOVOME Table II](https://arxiv.org/html/2403.02167v3); exact label names **UNVERIFIED**.
- **Authenticity:** Acted (lab) per [EMOVOME Table II](https://arxiv.org/html/2403.02167v3).
- **Language variety:** Multilingual — Spanish and Basque (Spain).
- **Sample rate & format:** **UNVERIFIED**.
- **Known problems:** Official web pages are offline (404); not freely available (**VERIFIED**).
- **Model rights (engineering fact):** **UNVERIFIED**.
- **Verdicts:**
  - **D3(a):** `reject` — Not free/open; official source offline (404) and licence UNVERIFIED.
  - **D3(b):** `reject` — Official source and licence UNVERIFIED.
  - **D3(c):** `reject` — Official source and licence UNVERIFIED.

---

## Multilingual Datasets with Spanish or Portuguese Subsets

Summary of multilingual emotional-speech corpora checked for Spanish (`es`) and Portuguese (`pt`) coverage:

| Dataset | Languages Included | Spanish Subset | Portuguese Subset | Licence Class | Qualified Above? |
|---|---|---|---|---|---|
| **Basis Global Emotions** | 108 locales (83 languages) | Yes — 9 locales (`es-AR`, `es-CL`, `es-CO`, `es-DO`, `es-EC`, `es-ES`, `es-GT`, `es-MX`, `es-PE`): 54.69 h, 14,320 clips, 862 speakers | Yes — 2 locales (`pt-BR`, `pt-PT`): 15.06 h, 3,771 clips, 134 speakers | Custom permissive (`basis-data-license-1.0`, manual HF gate, full text UNVERIFIED) | [Section 11](#11-basis-global-emotions) |
| **CMU-MOSEAS** | ES, PT, DE, FR | Yes — `es-ES`: 10,000 sentences, 341 speakers | Yes — `pt-PT`: 10,000 sentences, 399 speakers | On-request EULA for raw audio/labels (UNVERIFIED); public SDK has features only | [Section 12](#12-cmu-moseas) |
| **EmoFilm** | EN, IT, ES | Yes — 342 clips, 57 dubbing speakers, 5 emotions (no `neutral`) | No | Academic-only EULA | [Section 7](#7-emofilm) |
| **LAION Artificial Voice Roleplay** | DE, EN, ES, FR | Yes — 2,106 synthetic TTS clips (free-text captions) | No | Open (CC BY 4.0) | [Section 14](#14-laion-artificial-voice-roleplay-dataset) |
| **RekEmozio** | ES, EU (Basque) | Yes — 2,618 clips, 10 speakers (per secondary survey) | No | On-request / Not free (UNVERIFIED; source 404) | [Section 15](#15-rekemozio) |
| **ASVP-ESD** (`asvp-esd`, in SER catalogue) | Chinese, English, French, Russian & Others | **No dedicated Spanish code** (`Language(01=Chinese , 02=English ,04 = French , others:Russian and Others )` per [Zenodo 7132783](https://zenodo.org/records/7132783)) | **No dedicated Portuguese code** | Open (CC BY 4.0 on Zenodo; audio scraped from movies/YouTube/sound-effect sites) | Checked & excluded from ES/PT table |

---

## Additional Literature Corpora Checked and Excluded

Six additional Spanish acted corpora listed in the [EMOVOME paper Table II](https://arxiv.org/html/2403.02167v3) were checked and excluded from the qualification table (no official public download URLs found; licence texts **UNVERIFIED**):

- **Spanish Expressive Voices (SEV, UPM):** 3,890 clips, 2 speakers (1M/1F), 7 discrete emotions — excluded (`Free: No` in EMOVOME Table II; fewer than 4 speakers; no public source found).
- **Spanish Emotional Speech (SES):** 1,288 clips, 1 speaker (1M/0F), 4 discrete emotions — excluded (`Free: No`; single speaker; no public source found).
- **Iriondo et al.:** 336 clips, 8 speakers (4M/4F), 7 discrete emotions — excluded (`Free: No`; no public source found).
- **Emotional Mexican Spanish speech:** 240 clips, 6 speakers (3M/3F), 4 discrete emotions — excluded (`Free: No`; no public source found).
- **Martínez & Cruz:** 380 clips, >15 speakers, 5 discrete emotions — excluded (`Free: No`; no public source found).
- **"Mexican Emotional Speech Database" [ref. 30 in EMOVOME Table II]:** 3,456 clips, 8 speakers (4M/4F), 6 discrete emotions (`Free: Yes` in EMOVOME Table II; distinct from Mendeley MESD v5 which has 11 voices / 862 human clips) — excluded (no official download URL found; licence **UNVERIFIED**).

---

## Catalogue Corrections for the Acquisition Brief

No code was edited per brief limits. The following discrepancies in `ser/_internal/data/catalog/public_datasets.py` were verified against primary sources and should be addressed in the acquisition brief (D4-3):

1. **`escorpus-pe` (`ser/_internal/data/catalog/public_datasets.py:32-35`, `105-125`):**
   - **What the catalogue says:** Sets `ESCORPUS_PE_DATASET_POLICY_ID = "open"`, `ESCORPUS_PE_DATASET_LICENSE_ID = "cc-by-4.0"`, and maps 11 categorical Spanish emotion strings (`alegria`, `feliz`, `enojado`, `enojo`, `ira`, `miedo`, `triste`, `tristeza`, `neutral`, `asco`, `sorpresa`) in `ESCORPUS_PE_MANIFEST_SPEC`.
   - **What the source says:** [Zenodo 5793223](https://zenodo.org/records/5793223) and [GitHub README](https://raw.githubusercontent.com/Alessandra-UNSA/Peruvian_Spanish_Corpus/main/README.md) state `"It contains 3749 utterances of three emotional dimensions: Valence, Arousal and Dominance"` (no categorical emotion labels) and `"Anyone can use this data only for research purposes."` (contradicting the Zenodo `cc-by-4.0` tag).
2. **`spanish-meacorpus-2023` (`ser/_internal/data/catalog/public_datasets.py:82-85`, `275-293`):**
   - **What the catalogue says:** Sets `SPANISH_MEACORPUS_2023_SOURCE_URL = "https://zenodo.org/records/18606423"` (Zenodo v2) with `noncommercial` / `cc-by-nc-4.0`.
   - **What the source says:** Zenodo v2 ([18606423](https://zenodo.org/records/18606423)) distributes only `filename, label, transcription, split` (dropping `youtube_key`, `start_time`, `end_time`, and `gender`), whereas Zenodo v1 ([18246241](https://zenodo.org/records/18246241)) keeps `youtube_key`, `segment_id`, `label`, `gender`, `start_time`, `end_time`, `transcription`, `split`, `source`. The upstream `code/download-dataset.py` ([GitHub](https://raw.githubusercontent.com/NLP-UMUTeam/Spanish-MEACorpus-2023/main/code/download-dataset.py)) reads columns from both CSVs and runs on neither unmodified. Moreover, v1's official split leaks 436 of 877 test `youtube_key`s (551 of 1,027 test rows) into `train`, and neither CSV contains per-clip speaker IDs.
3. **`coraa-ser` (`ser/_internal/data/catalog/public_datasets.py:77-80`, `261-273`):**
   - **What the catalogue says:** Sets `CORAA_SER_DATASET_POLICY_ID = "research_only"` and `CORAA_SER_DATASET_LICENSE_ID = "custom-research-only"` pointing to `https://github.com/rmarcacini/ser-coraa-pt-br`.
   - **What the source says:** [rmarcacini/ser-coraa-pt-br](https://github.com/rmarcacini/ser-coraa-pt-br) contains no licence file (`license: null` in GitHub API); its upstream source [C-ORAL-BRASIL I](https://www.c-oral-brasil.org/c-oral-brasil-i.php) is licensed under `CC BY-NC-SA 4.0`. Also note the catalogue's label map uses underscores (`non_neutral_female`, `non_neutral_male`) while the source CSV/filenames use hyphens (`non-neutral-female`, `non-neutral-male`).
