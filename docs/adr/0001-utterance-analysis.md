# ADR 0001: Utterance analysis interface

- Status: proposed. Interface, stubs and tests only; the implementation is brief A1-2.
- Contract source: SER plan section 2.2, "The integration contract SER must meet", and checklist U1 to U14.
- Revised after the independent review of 95b5eef and the re-check of ecbebaf. Rulings marked "lead" and
  decisions D-a to D-g were settled by the lead of this work.

## Context

`ser.api.infer()` takes a file path, reloads the model on every call, prints a timeline and transcribes
by default. A consuming application needs a quiet, load-once call that analyses one utterance and returns
one JSON-serialisable record, `ser.utterance/v1`, within a hard deadline.

The stubs in `ser/_internal/analysis/` raise `NotImplementedError`. The tests in
`tests/suites/unit/analysis/` pin each decision below. They are marked
`xfail(strict=True, raises=NotImplementedError)`, one marker per test, until the implementation lands.
Every test uses generated audio and generated or synthetic models. All test paths below are relative to
that directory.

```python
from ser.api import UtteranceAnalyzer

manifest = UtteranceAnalyzer.inspect_model(profile="fast", model_path=path)
# the caller checks manifest.provenance against its own licence allow-list, then:
with UtteranceAnalyzer.load(profile="fast", model_path=path,
                            expected_sha256=manifest.identity.model_sha256) as analyzer:
    record = analyzer.analyze("clip.wav")
    record = analyzer.analyze(samples, sample_rate_hz=48_000, language="es")
```

In a script, put the worker-mode `load` call under `if __name__ == "__main__":`
so Python's spawn process can import the main module safely.

## Decisions

Each decision gives the rule, then the reason, then the tests that pin it.

### D1. The call is public in `ser.api` now (lead ruling 1)

**Rule**

`ser.api` re-exports the analysis names:
- `UtteranceAnalyzer`, the `Analysis*Error` family, and `utterance_record_json_schema`;
- the record `TypedDict`s and `UTTERANCE_SCHEMA_VERSION`;
- `AnalysisConfig`, `AnalysisTargets` and `AnalysisIsolation`;
- `LanguageHint`;
- `ModelIdentity`, `ModelManifest`, `ModelFormat`, `TrainingAudioForm`, `ModelProvenance`, `LicenceRecord` and `LicenceClass`.

Data types live in the contract leaves `ser.runtime.contracts` and `ser.runtime.schema`. Behaviour lives
in `ser._internal.analysis`.

This change updates the `ser.api` literal in
`tests/suites/integration/architecture/test_api_import_boundary.py`. It regenerates
`public_api_snapshot.json` with `scripts/dump_public_api.py --write`.

**Reason**

The brief requires public names and tests that use them. The lead approved the API review in this change.

**Tests**

Every contract test imports public names from `ser.api`. The architecture suite pins the export list,
the snapshot, and the rule that importing `ser.api` loads no PyTorch.

### D2. One construction route

**Rule**

The analyser is built only with `UtteranceAnalyzer.load(*, profile, ...)`. `profile` is required.
`__init__` takes a private session and is not part of the contract.

The `load` arguments:
- `model_path` overrides the configured model;
- `expected_sha256` pins the bytes;
- `allow_legacy_baseline` opts into a non-conforming model;
- `config` and `settings` override the defaults.

The analyser is a context manager, and leaving the block is `close()`. The `close()` rules are in D3
(lead decision D-a).

**Reason**

A required profile means a later change of the default can never silently change a caller's results.
A single route keeps the implementation seam out of the contract.

**Tests**

- `test_analyzer_runtime.py::test_close_is_idempotent_and_ends_the_analyzer`
- `test_analyzer_runtime.py::test_close_waits_for_the_call_in_flight`

### D3. Hard deadline through a persistent worker process (lead ruling 2; decisions D-a, D-b, D-d)

**Rule: the deadline (D-b)**

`analyze` returns no later than its deadline, whatever the backend is doing. The deadline is measured
from entry to `analyze` and includes queueing behind other calls. Its length is
`max(timeout_floor_s, timeout_per_audio_second × duration_s)`, where `duration_s` is the source
duration:
- **Array input.** The duration is known at entry, so the formula applies from the start.
- **A path.** The deadline starts at `timeout_floor_s`. When the worker reports the source duration
  (from the header, or after decoding), the deadline is extended once to the formula's value. If the
  floor passes first, the call is `timeout`.

**Rule: supervision (D-b)**

- The caller's wait is a bounded wait on "result or deadline", never a blocking receive
  (`AnalysisWorker.next_event(until_monotonic_s=...)`).
- Killing, reaping and restarting the worker run on the supervisor's own thread, never on a caller's
  path.
- Decoding a path happens in the worker, so a blocked decode is bounded like scoring.
- **Tie between result and deadline.** A result fully received before the deadline is returned.
  Otherwise the call returns `timeout`.

**Rule: the worker**

`load` reads the model bytes once, then spawns one child process, the worker. The worker loads the
backend and keeps it loaded. Calls on one analyser are served one at a time, in arrival order.

**Rule: states (D-a)**

`WorkerSupervisor` in `worker.py` holds the states `starting`, `ready`, `busy`, `restarting`,
`failed`, `closing` and `closed`.

| From | Event | To |
|---|---|---|
| `starting` | the child is ready, and `load` returns | `ready` |
| `starting` | the start fails, or exceeds `worker_start_timeout_s` | none: `load` raises `AnalysisModelLoadError` |
| `ready` | a call is admitted and the child is alive | `busy` |
| `ready` | a call is admitted and the child died while idle | `restarting`; the call waits |
| `busy` | the outcome is fully received before the deadline | `ready` |
| `busy` | the deadline passes; the call returns `timeout`, and the child is killed | `restarting` |
| `busy` | the child dies; the call returns `model_error` | `restarting` |
| `restarting` | the child is ready | `ready` |
| `restarting` | the start fails, or exceeds `worker_start_timeout_s` | `failed` |
| `failed` | a call is admitted; one new start attempt is made | `restarting` |
| any open state | `close()` begins | `closing` |
| `closing` | the call in flight has ended, and the child is stopped | `closed` |

- **During a restart.** A call that arrives while the worker restarts waits for it, inside its own
  deadline. It gets its result if the worker comes back, `timeout` if its own deadline passes first,
  and `model_error` if the restart fails. A call that times out while queued or waiting neither evicts
  the worker nor cancels the restart.
- **Idle death.** A worker that dies while idle is restarted at the next call, in the same way. That
  costs one reload, and there is no error unless the restart fails.
- **A failed restart with no caller waiting.** It leaves the analyser `failed`. The next call triggers
  one new restart attempt and reports `model_error` if that attempt fails too.
- **Reload.** A restart starts from the bytes that `load` kept, so the file is not read again.
  Load-once holds while the worker is healthy.
- **Close.** These raise `AnalysisClosedError`, a subclass of `AnalysisUsageError`:
  - a call that arrives after `close()` began;
  - a queued call that had not started when `close()` began.

  `close()` lets the call in flight finish or reach its deadline. It is idempotent.
- **What a caller observes.** A caller never waits past its own deadline. After an eviction, within
  `worker_start_timeout_s` calls either succeed again or return `model_error`.

**Rule: process settings (D-d)**

- **The worker.** It runs at `worker_nice` (default 19, the lowest priority). Its native thread pools
  are capped at `max_threads` (default 2), and every GPU is hidden from it. A nice value can only be
  raised relative to the caller's, so the worker runs at `max(worker_nice, caller nice)`.
- **Inline mode.** `isolation="inline"` is an explicit opt-in for callers that isolate SER themselves.
  It runs the backend in the caller's process and checks the deadline only between stages, so it is
  **not** a hard bound.
- **Inline never changes process-wide settings** (nice value, thread pools, environment): the caller
  owns its process. `BackendLoadRequest.max_threads` is `None` inline, and a loader then leaves thread
  settings alone.

**Reason**

- Section 2.2 makes the deadline part of the contract, and U9 says nothing hangs past it.
- Only another process can stop a native call that is stuck, and only a wait that never blocks can
  return on time while that process is killed.
- Naming each state settles what a caller observes in every race; the alternative is an implementation
  choice the consumer cannot see.
- The owner has decided that SER runs in its own process at the lowest priority, beside its consumer.
  That makes the worker's priority and thread cap part of the configuration. A process-wide cap applied
  inline would change the caller's other work.

**Reuse of what the repository has**

- Reused:
  - the spawn start method, as `fast_public_boundary.py` uses through `mp.get_context` (line 260);
  - `worker_lifecycle.terminate_worker_process` (line 136), called on the supervisor thread only.
- Not reused:
  - `worker_lifecycle.recv_worker_message` (line 34), because it blocks;
  - `run_process_setup_compute_handshake` (`worker_lifecycle.py:150`), and the medium and accurate
    `run_with_process_timeout`: each starts one process per call, which breaks load-once;
  - `run_with_timeout` (`worker_lifecycle.py:98`), which is thread-based and cannot stop native code;
  - the module-global `_SINGLE_FLIGHT_REGISTRY` (`fast_public_boundary.py:65`), because instances are
    independent (D15).
- Nothing in the repository sets a nice value or caps thread pools. The worker adds both.

**Tests (`test_analyzer_deadline.py` unless named)**

Real-clock tests allow 50 ms (`_SCHEDULING_SLACK_S`) for thread scheduling only. The rule allows no
overrun. Event-driven tests with an injected clock are acceptance items (below).

| Pins | Test |
|---|---|
| The formula table | `test_deadline_is_the_floor_or_two_seconds_per_audio_second` |
| Both branches of the formula | `test_both_deadline_branches_allow_work_within_them` |
| A hung backend returns `timeout` on time; one restart and one file read | `test_a_hung_backend_times_out_on_time_and_the_worker_comes_back` |
| A queued call times out without evicting | `test_a_call_that_times_out_in_the_queue_does_not_evict_the_running_one` |
| A dead worker | `test_a_worker_that_dies_is_a_model_error_and_is_replaced` |
| A failed restart, then the retry | `test_a_failed_restart_is_a_model_error_and_the_next_call_retries` |
| Inline is not a hard bound | `test_inline_isolation_checks_the_deadline_between_stages_only` |
| Worker priority and thread cap | `test_analyzer_config.py::test_worker_runs_at_low_priority_with_capped_threads_and_no_gpu` |
| Inline leaves the caller alone | `test_analyzer_config.py::test_inline_mode_leaves_the_caller_process_alone` |
| Close and the closed error | `test_analyzer_runtime.py::test_close_is_idempotent_and_ends_the_analyzer`, `test_analyzer_runtime.py::test_close_waits_for_the_call_in_flight` |

### D4. Models load through one artifact protocol with digest-bound provenance (lead ruling 3; decision D-c)

**Rule: the protocol**

The model is loaded only through `ser._internal.analysis.artifact.open_model`. That function receives the
bytes that `read_model_file` read once. The `ArtifactFormat` protocol (`name`, `claims`, `open`) yields an
`OpenedArtifact`. Its `ModelManifest` carries:
- `identity`: the SHA-256 of the verified bytes, the model id, the label set id and the labels;
- `training_audio`: the audio form the model was trained on;
- `provenance`: the payload SHA-256, the encoder licence and the dataset licences. Each licence has a
  name, a licence, a licence class and a source.

The safe format is the seam that brief P2-4 fills.

**Rule: verification order**

1. Match `expected_sha256` if given.
2. Find the format that claims the bytes.
3. Require provenance whose payload digest matches the payload.
4. Require a complete licence record (decision D-c):
   - `encoder` is the encoder's licence, or the explicit `"none"`, which means no pretrained encoder
     was used;
   - at least one training dataset is listed;
   - every record has a non-empty `name` (the id an allow-list matches on), `licence` and `source`,
     and a known `licence_class`.

   An incomplete record is `AnalysisModelLoadError`, not a warning, even with the opt-in.
5. Require `model_id` and `label_set_id`.
6. Require the artifact's profile to match.
7. Require canonical training audio (D6).

**Rule: the licence allow-list hand-off**

`UtteranceAnalyzer.inspect_model(...)` returns the manifest without starting a backend. The caller checks
the licences against its own allow-list. It then calls `load(expected_sha256=...)`, so that the bytes that
load are the bytes it inspected. SER keeps no allow-list.

**Rule: pickle**

Pickle is trusted legacy plumbing only. Without `allow_legacy_baseline=True`, a pickle is refused, with
`AnalysisModelNotConformingError`, before any unpickling. With the opt-in it loads as a baseline (D6).
It never satisfies provenance.

**Reason**

Section 2.2 rules out pickle and requires a licence gate before analysis. Binding the provenance to the
digest means it describes the bytes that run. An allow-list can only judge what is recorded, so a
missing entry must stop the load rather than pass as "no restriction".

**Tests (`test_analyzer_load.py`, with synthetic artifacts)**

- `test_safe_artifact_manifest_is_its_verified_identity_and_licence`
- `test_inspection_reads_the_licence_without_loading_the_model`
- `test_allow_list_hand_off_pins_the_inspected_bytes`
- `test_unverifiable_safe_artifacts_never_load`: tampered, missing or incomplete provenance (encoder
  unrecorded, no datasets, a dataset without id, licence or class), with and without the opt-in
- `test_a_model_without_a_pretrained_encoder_says_so`
- `test_safe_artifacts_must_name_model_and_label_set`
- `test_an_artifact_for_another_profile_is_a_load_error`
- `test_a_pickle_is_refused_before_it_is_unpickled`: a booby-trapped pickle

### D5. `model_sha256` is the identity; `model_id` is a label (decision D-c)

**Rule**

- **Identity.** `model_sha256` is the SHA-256 of the bytes that were loaded, in full lower-case hex. It
  is the identity of a model.
- **`model_id`.** It is a label chosen by whoever built the model and is not promised unique. A safe
  artifact's `model_id` is copied verbatim, so two different files may share it. A consumer keys stored
  results on the pair (`model_id`, `model_sha256`).
- **Safe artifacts.** These must carry `model_id` and `label_set_id`.
- **Baselines.** The id is `baseline:<name>@<full sha256>`. `<name>` is the artifact's `model_id`, else
  the legacy envelope's `model_revision` (`ArtifactMetadata.model_revision`), else
  `<profile>-<backend_id>`.
- **Legacy label sets.** `label_set_id` is `"labels:" + sha256(json.dumps(sorted(set(labels)), separators=(",", ":")))`.

**Reason**

SER cannot make a builder's label unique; a digest is unique without coordination. A 12-hex prefix is
not guaranteed unique. The canonical JSON form cannot collide the way a delimiter-joined list can.

**Tests (`test_analyzer_load.py`)**

- `test_legacy_pickle_loads_only_as_a_baseline_with_exact_ids`
- `test_legacy_model_revision_names_the_baseline`
- `test_baseline_ids_carry_the_digest_and_label_set_ids_follow_the_labels`

### D6. Canonical audio only; non-canonical models are baselines on opt-in (lead ruling 6)

**Rule**

The analyser always converts input to canonical audio, and there is no model-native path. A model whose
recorded training audio is not canonical raises `AnalysisModelNotConformingError`. A model whose training
audio is unknown, as with today's pickles, raises the same error. Not canonical means any of:
- a rate other than 16 kHz;
- more than one channel;
- no record that the model went through the canonical conversion.

Either kind of model loads only with `allow_legacy_baseline=True`. It is then marked by the `baseline:`
prefix of its `model_id`. The schema stays closed: no new field is added.

The existing `fast` model is a plumbing baseline only. It is uncalibrated and is not a conforming model.

**Reason**

The existing reader keeps the source rate (`audio_utils.py:104`), and the training feature path accepts
it (`data_loader.py:497`). Today's model may therefore have been trained on audio that inference would
not produce. A silent fallback would hide that difference.

**Tests (`test_analyzer_load.py`)**

- `test_non_canonical_training_audio_is_a_baseline_on_opt_in_only`
- `test_legacy_pickle_loads_only_as_a_baseline_with_exact_ids`

### D7. One canonical conversion seam (lead ruling 5)

**Rule**

Both input forms pass, in native channels and at the native rate, through
`ser._internal.utils.audio_utils.to_canonical_audio(samples, sample_rate, *, channel_axis=1)`. That is
brief M5-1's seam. It is frames-first, never guesses the layout, and raises `ValueError` on invalid input.

`ser._internal.analysis.audio_input` calls it as `audio_utils.to_canonical_audio`, and re-exports
`CanonicalAudio` and `CANONICAL_SAMPLE_RATE` from the same module. Nothing mixes down or resamples
before it. Training uses the same function.

It is called exactly once for each call that passes step 2 of the input order in D11, and not at all
otherwise (decision D-e).

**Reason**

If training and inference both use one conversion, they cannot drift apart. Mixing down before the seam
would make it something other than the shared train and inference conversion.

**Tests (`test_analyzer_inputs.py`)**

| Pins | Test |
|---|---|
| Native channels and rate reach the seam; mono 16 kHz comes out | `test_native_channels_and_rate_reach_the_shared_seam` (a spy at the seam) |
| A signal-sensitive backend agrees within 0.05 across 8, 16, 44.1 and 48 kHz, for arrays and files | `test_every_input_rate_is_resampled_to_the_same_scores` |
| Unequal stereo channels give the mix, not one channel | `test_stereo_is_mixed_by_the_canonical_conversion_not_by_dropping_a_channel` |
| Empty and out-of-range sources never reach the seam | `test_out_of_range_sources_are_classified_before_conversion` |

### D8. Inputs and misuse

**Rule**

`analyze(path)` takes a `str` or an `os.PathLike`. `analyze(samples, *, sample_rate_hz=...)` takes a
one-dimensional `float32` array and a positive `int` rate.

The following raise `AnalysisUsageError`:
- other dtypes (float64 included), other shapes and lists;
- a missing, non-positive, float or `bool` rate;
- a path given together with a rate.

`language` is `None`, `"en"`, `"es"` or `"pt"`, and any other value raises (lead ruling 8). The
normalised hint reaches the backend's `score(..., language=...)`.

**Reason**

Rejecting float64 leaves the precision loss to the caller. Rejecting an unknown hint keeps caller mistakes
visible; a caller that wants leniency can map unknown hints to `None`.

**Tests (`test_analyzer_inputs.py`)**

- `test_malformed_arguments_are_usage_errors`
- `test_unknown_language_hint_is_a_usage_error`
- `test_the_language_hint_reaches_the_backend`
- `test_wav_path_may_be_str_or_pathlike`
- `test_samples_and_float_wav_of_the_same_audio_give_the_same_record`
- `test_integer_pcm_wav_is_accepted`

### D9. The record `ser.utterance/v1`

**Rule**

`UtteranceRecord` has exactly the 13 fields of the plan's example, and `schema_version` is typed
`Literal["ser.utterance/v1"]`. `utterance_record_json_schema(version)` returns the closed Draft 2020-12
schema (`additionalProperties: false`). An unknown version raises `ValueError`.

- **Changes.** Adding, removing or retyping a field requires a new `schema_version`.
- **Reading old records.** The function keeps the schema of every published version, so a record stays
  readable through the version it carries.
- **`ser_version`.** This is `ser.__version__`. Its SemVer promise is that of `docs/api-stability.md`.
- **`dimensions`.** This is `null` in v1; real valence and arousal come with v2 (lead ruling 8).
- **Failure statuses.** Any status other than `ok` carries `null` for `labels`, `top_label`, `confidence`
  and `frames`. It still carries every identity field, `audio` as far as known, and `timing_ms`.
- **Timing (decision D-f).** The clock is `time.monotonic_ns`, floored to whole milliseconds.
  - `timing_ms.features` covers decode, conversion and extraction. `timing_ms.predict` covers scoring.
  - A stage that never started reports 0.
  - When the worker is killed or dies, a stage the worker reported finished keeps its time. The
    interrupted stage runs until the parent stopped waiting.

**Reason**

A closed, versioned schema lets consumers rely on the exact shape. Timing is the only field excluded from
determinism.

**Tests**

- `test_record_schema.py`, all of it. This includes:
  - a fixed v1 fixture;
  - every identity field;
  - every failure status, with each label field set wrongly;
  - integral floats as JSON integers but not as record ints.
- `test_analyzer_record.py::test_ok_record_meets_every_record_clause`
- `test_analyzer_record.py::test_timing_reports_features_and_predict_separately`
- `test_analyzer_record.py::test_records_of_two_models_on_the_same_audio_each_name_their_model`
- `test_analyzer_record.py::test_record_names_the_schema_and_ser_versions`

The in-repository schema validator is a tested subset of Draft 2020-12, and it adds no dependency (lead
ruling 10). Schema checks and Python-type checks are separate assertions.

### D10. Scores have one authority; frames are descriptive (lead ruling 9)

**Rule: scores (decision D-f)**

The backend's utterance scores are the one authority for `labels`, `top_label` and `confidence`. Ties go
to the alphabetically first label.

A backend's scores, for the utterance and for each frame, must meet all of these:
- the keys are exactly the label set;
- every value is finite and in `[0, 1]`;
- the values sum to 1 within 1e-6.

The frames must not leave a gap from 0 to the duration, and must not lie outside it.

Anything else is `model_error`. Valid scores are divided by their sum before they are written, so every
record satisfies the schema.

**Rule: frames**

Frames are descriptive and together cover `[0, duration_s]`. A backend that scores the whole utterance
returns one frame.

For `fast`, frame `k` starts at `k × 1 s` and ends at `min(start + 3 s, duration_s)`. Frames are added
until one reaches the duration, so the tail frame is truncated. The 3 s size and 1 s stride come from
`artifact_envelope.py:27-28`.

**Rule: pooling for `fast`**

`fast` pools its frames with the coverage-weighted mean. Every instant of the utterance weighs the same,
so overlapping windows do not overweight the middle.

**Rule: calibration**

Calibration is required of a conforming model; it belongs to briefs M5-2 and U11. The legacy `fast`
scores are uncalibrated.

**Reason**

Pooled frame scores and separately calibrated utterance scores must not compete as authorities. A plain
duration-weighted mean would overweight audio that overlapping windows cover several times.
Renormalising only within 1e-6 removes rounding noise without hiding a broken backend.

**Tests (`test_analyzer_record.py`, unless named)**

| Pins | Test |
|---|---|
| The `fast` frame layout | `test_fast_frame_layout`, `test_fast_records_use_the_fast_frame_layout` |
| The pooling rule | `test_coverage_weighted_mean_weighs_every_instant_once`, `test_fast_utterance_scores_are_the_coverage_weighted_mean_of_its_frames` |
| The tie-break | `test_ties_go_to_the_alphabetically_first_label` |
| The backend is the authority | `test_backend_utterance_scores_are_the_authority_for_top_label` |
| Invalid scores are `model_error`, including a value above 1 within the sum tolerance | `test_analyzer_statuses.py::test_invalid_backend_scores_are_a_model_error` |
| Valid scores are renormalised | `test_valid_scores_are_renormalised_before_they_are_written` |

### D11. Statuses, exceptions and their order

**Rule: input order (decision D-e)**

Each call goes through these steps in order. The first step that decides the outcome ends the call.

1. **Validation.** Type, dtype, shape and sample rate are checked. A violation raises
   `AnalysisUsageError` (D8).
2. **Duration.** A path is first decoded, in the worker. A file that is missing, empty or a directory,
   whose header or payload is truncated, or that is not audio, is `unreadable`. A valid-shape array with zero samples is `too_short`, without conversion. So is any
   source whose duration is below `min_duration_s`. A source above `max_duration_s` is `too_long`,
   without conversion. Both limits are inclusive.
3. **Conversion.** The source is converted exactly once through the seam (D7). A `ValueError` from the
   seam, as for NaN or infinite samples, is `unreadable`.
4. **Silence.** It is decided on the canonical mono signal, by strict less-than (D12).
5. **Scoring.** The backend scores the audio. If it raises, if the worker dies, or if the scores are
   invalid (D10), the status is `model_error`. Otherwise it is `ok`.

`timeout` can end any step (D3).

`audio.duration_s` and `audio.input_sample_rate_hz` are the source's values whenever they are known,
including on `too_short` and `too_long`. Otherwise they are `null`.

**Rule: exceptions**

Programming errors and unusable models raise. All exceptions derive from `AnalysisError`:

| Exception | When |
|---|---|
| `AnalysisUsageError` | a misuse (D8); an invalid config; a profile without a backend |
| `AnalysisClosedError`, a subclass of `AnalysisUsageError` | a call made, or still queued, when `close()` began (D3) |
| `AnalysisModelUnavailableError` | no model is configured, or the file is missing. SER never downloads one |
| `AnalysisModelLoadError` | the model cannot be read or started, or does not match its profile |
| `AnalysisModelNotConformingError`, a subclass of `AnalysisModelLoadError` | the cases in D4 and D6 |

Errors are raised `from None`, and their messages contain no caller paths. A traceback therefore
reveals nothing through `__cause__`.

`KeyboardInterrupt` and `MemoryError` propagate unchanged in inline mode. In worker mode they happen in
the child and are reported as `model_error`.

**Tests**

- `test_analyzer_statuses.py`, all of it
- `test_analyzer_config.py::test_limits_apply_exactly_at_their_boundaries`
- `test_analyzer_load.py`, from `test_missing_model_file_fails_at_load` to `test_a_second_backend_serves_its_own_profile`
- `test_analyzer_runtime.py::test_interrupts_and_memory_errors_propagate_inline`

### D12. Silence (lead ruling 7, provisional)

**Rule**

A record is `silent` when the RMS of the canonical mono samples is strictly less than `silence_rms_dbfs`.
The RMS is taken before any feature transform, with full scale 1.0, as `20·log10(rms)` dBFS. The default
is -60 dBFS, and it is configurable.

**Reason**

A threshold is needed so that silence does not produce confident labels. The number is provisional. It
must be validated on real quiet recordings before anything relies on it; this is not an owner decision.

**Tests**

`test_analyzer_config.py::test_limits_apply_exactly_at_their_boundaries` checks levels just above and just
below the configured threshold. Exact equality is an acceptance item (below).

### D13. Configuration: enforced limits apart from measured targets (lead ruling 4)

**Rule: limits**

`AnalysisConfig` holds the enforced limits. Their defaults are:
- `min_duration_s=0.3` and `max_duration_s=120.0`, both inclusive;
- `timeout_floor_s=10.0` and `timeout_per_audio_second=2.0`;
- `silence_rms_dbfs=-60.0`;
- `isolation="worker"`;
- `max_threads=2`;
- `worker_nice=19`;
- `worker_start_timeout_s=60.0`.

**Rule: targets**

`AnalysisConfig.targets` (`AnalysisTargets`) holds targets that are measured by the performance harness
and never enforced:
- `p95_real_time_factor=0.5`;
- `cold_start_s=20.0`;
- `steady_memory_bytes=1_000_000_000`, which is 1.0 GB in decimal bytes.

**Rule: device and validation**

Analysis is CPU-only, and no field selects a device. An invalid value raises `AnalysisUsageError` at
`load`.

**Reason**

The plan's numbers stand as defaults until measured; that is the owner's decision of 2026-11-28. Keeping
targets apart from limits prevents a measurement goal from being read as a guarantee. Byte units remove
the MB/MiB ambiguity. `worker_start_timeout_s` is new here and proposed.

**Tests (`test_analyzer_config.py`)**

- `test_invalid_configuration_is_a_usage_error`: 23 cases
- `test_limits_apply_exactly_at_their_boundaries`
- `test_performance_targets_are_not_enforced`

### D14. Load once, quiet and offline

**Rule**

- **Loading.** The file is read once, at `load`. In worker mode the backend is deserialised once per
  worker start; the parent also unpickles a legacy baseline once, for its metadata. After that, `analyze`
  does not need the file.
- **Network.** No network access, during `load` or `analyze`.
- **Output.** Nothing goes to stdout or stderr, and no file is created or deleted, including temporary
  files.
- **Logging (decision D-g).**
  - Analysis logs under `ser._internal.analysis`. SER sets no level and adds no handler, so the default
    effective level is WARNING.
  - A record at WARNING is written only for a worker eviction, death or failed restart. Every other
    record is DEBUG; a healthy call writes nothing at WARNING or above.
  - No record at any level, no exception message and no metric carries audio, file contents,
    caller-supplied paths or backend content (backend exception text, features, scores). This holds in
    the parent and in the worker.
  - No warnings filter changes, and no warning is emitted.
- **Determinism.** The same audio and the same model give the same record apart from `timing_ms`.

**Reason**

The consumer embeds SER beside a voice-reply path and must see neither side effects nor reloads.
Section 2.2 sets the default level to warning and bans content from logs, errors and metrics. A
restart is the one event an operator needs to see.

**Tests (`test_analyzer_runtime.py`)**

- `test_the_model_is_read_and_started_once_for_many_analyses`: 100 calls
- `test_inline_loads_the_backend_once`
- `test_the_model_file_is_not_needed_after_load`
- `test_load_and_analysis_use_no_network`: an audit hook
- `test_nothing_is_printed_or_written`: read-only working directory and `TMPDIR`
- `test_logging_and_warnings_are_left_alone`
- `test_results_are_deterministic`

**Gaps**

Memory growth of under 5% across many calls is measured by the C7 harness, not here. The audit hook sees
only the parent. The worker's load count, I/O, network and logs are acceptance items (below).

### D15. Synchronous and thread-safe; instances are independent

**Rule**

`analyze` is synchronous; async callers use `asyncio.to_thread`. Calls on one instance run one at a time,
and their results equal sequential results.

Separate instances share no lock or registry. Each has its own worker, and memory doubles.

**Reason**

The work is CPU-bound, and SER cannot size a hidden pool well for its caller.

**Tests (`test_analyzer_runtime.py`)**

- `test_one_analyzer_can_be_shared_by_threads`
- `test_two_analyzers_in_two_threads_are_independent`
- `test_inline_calls_on_one_instance_never_overlap_in_the_backend`
- `test_inline_instances_do_not_serialise_each_other`

### D16. A second backend

**Rule**

`AnalysisBackend` is a protocol with three methods:
- `extract(audio: CanonicalAudio, *, deadline_monotonic_s)`;
- `score(features, *, language, deadline_monotonic_s) -> BackendScores`;
- `close()`.

`backend_loaders()` maps each profile to a module-level loader, which pickles by reference into the
worker. Adding `medium` means registering one loader. `accurate` and `accurate-research` raise
`AnalysisUsageError`.

**Tests (`test_analyzer_load.py`)**

- `test_a_second_backend_serves_its_own_profile`
- `test_profile_without_an_analysis_backend_is_a_usage_error`

### D17. `infer()` is unchanged

`ser.api.infer()` keeps its behaviour, and this change does not deprecate it.

## Acceptance items for the implementation

The tests below cannot be proven tight while the analyser is a stub, because an expected failure hides
any other error. Brief A1-2 makes each one real against the actual analyser, and the adversarial review
checks each one.

1. **Gated events (D3).**
   - Inject the supervisor's clock.
   - Add controllable gates for a blocked decode, a queued call, blocked scoring and a blocked restart.
   - Assert each D3 transition, including the result-versus-deadline tie, idle death, a failed restart
     with nobody waiting, and arrival order.
   - Assert that the caller's return never waits on kill, reap or restart.
2. **One real-process smoke test (D3).** A hung child returns `timeout` within the deadline plus
   scheduling slack, and the next call succeeds.
3. **Load count over 100 calls (D14).** The child's own model-load count, counted inside the worker, is
   1 over 100 healthy calls. A wrapper count does not satisfy this.
4. **Child isolation (D14).**
   - The child makes no network connection.
   - The child creates and deletes no files, with I/O observed in the child.
   - `TMPDIR` is redirected to an empty directory that is still empty afterwards.
5. **Content sentinels (D14, decision D-g).**
   - Put a sentinel in the path, the backend's exception text and the backend's scores.
   - Capture logs at every level, DEBUG included, in the parent and in the worker, plus exception
     messages.
   - Assert the sentinel appears in none of them.
6. **Samples at the seam and at the backend (D7, D11).**
   - Do this for array input and for an integer PCM file.
   - The seam receives the source samples and rate.
   - The backend receives exactly the seam's output.
7. **Stereo cancellation (D7).** Opposite-phase stereo channels mix to silence and give `silent`, which
   proves that the channels were mixed and not one dropped.
8. **Exact silence equality (D12).** A signal whose canonical RMS equals the threshold exactly (for
   example, full scale at 0 dBFS) is not `silent`.
9. **Scores (D10).**
   - Asymmetric frame windows, where the coverage-weighted mean differs from the plain mean.
   - A tie, observed through a public record.
   - Utterance and frame scores that disagree, so that the record follows the utterance scores.
   - An invalid frame probability alongside valid utterance scores.
10. **Context exit (D2).** Leaving a `with` block closes the analyser before any explicit `close()`; a
    later redundant `close()` is a no-op.

## Hand-offs to other work

- **The caller.** It schedules analysis off its reply path. A failed start is a typed exception from
  `load`; after that, every call yields exactly one status.
- **The consumer's adapter.** It translates any failure status or exception to "no emotion". It adds no
  fields to the record. This is pinned by the adapter's acceptance tests, which are external.
- **A1-3 (batch).** It owns atomic rows and restart keys, and relies on the bounded lifecycle of D3. U10
  is tested there.
- **External tests.** U12 (consumer integration) and U13 (licence records, on top of D4's hand-off) are
  tested outside this repository's unit suite.
- **Performance and memory (U7, U8).** The C7 harness measures these against the targets in D13.
- **Calibration (U11).** M5-2 provides it.

## Consequences

- Consumers get one call, one closed versioned record, and a hard time bound.
- Each analyser costs one extra process and its resident model.
- A conforming model needs the P2-4 safe artifact and M5-1 canonical training. Until both land, only a
  baseline loads, and only on explicit opt-in.

## Open questions

1. `worker_start_timeout_s=60.0` is proposed here and is not in the plan.
2. -60 dBFS awaits validation on real quiet recordings (D12).
3. Network and file isolation of the worker are pinned only through acceptance item 4.
