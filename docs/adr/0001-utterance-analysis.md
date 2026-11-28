# ADR 0001: Utterance analysis interface

- Status: proposed. Interface, stubs and tests only; the implementation is brief A1-2.
- Contract source: SER plan section 2.2, "The integration contract SER must meet", and checklist U1 to U14.
- Revised after the independent review of 95b5eef. Rulings marked "lead" were settled by the lead of this work.

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

The analyser is a context manager. `close()` waits for the call in flight and is idempotent. `analyze`
after `close` raises `AnalysisUsageError`.

**Reason**

A required profile means a later change of the default can never silently change a caller's results.
A single route keeps the implementation seam out of the contract.

**Tests**

- `test_analyzer_runtime.py::test_close_is_idempotent_and_ends_the_analyzer`
- `test_analyzer_runtime.py::test_close_waits_for_the_call_in_flight`

### D3. Hard deadline through a persistent worker process (lead ruling 2)

**Rule: the deadline**

`analyze` returns no later than its deadline, whatever the backend is doing. The deadline is
`max(timeout_floor_s, timeout_per_audio_second × duration_s)`. It is measured from entry to `analyze` and
includes queueing behind other calls.

Until the duration is known, the floor applies. The default `isolation="worker"` meets the deadline as
follows:

- **The worker.** `load` reads the model bytes once, then spawns one child process, the worker. The
  worker loads the backend and keeps it loaded. Calls on one analyser are sent to it one at a time.
- **At the deadline.** The parent kills the worker with SIGKILL, so the call returns
  `status: "timeout"` on time.
- **Restart.** The parent starts a new worker from the bytes it kept, so the file is not read again. The
  next call pays one reload. Load-once holds while the worker is healthy.
- **A worker that dies.** If the worker dies during a call, that call returns `model_error`, and the
  worker is replaced in the same way.
- **During a restart.** A caller either waits for the restart or returns `timeout` at its own deadline,
  whichever comes first. A call that times out while queued or waiting neither evicts the worker nor
  cancels the restart.
- **A restart that does not finish.** A restart that exceeds `worker_start_timeout_s` (default 60 s), or
  fails, makes the waiting calls return `model_error`. The next call starts a new worker.
- **The bound.** A caller never waits longer than its own deadline. After an eviction, no more than
  `worker_start_timeout_s` passes before calls either succeed again or return `model_error`.
- **Process settings.** The worker runs with `worker_nice` (default 19, lowest priority) and with native
  thread pools capped at `max_threads` (default 2). All GPUs are hidden from it. A nice value can only be
  raised relative to the caller's, so the worker runs at `max(worker_nice, caller nice)`.

**Rule: inline mode**

`isolation="inline"` is an explicit opt-in for callers that isolate SER themselves. It runs the backend in
the caller's process and checks the deadline only between stages, so it is **not** a hard bound.

Inline mode leaves the caller's nice value alone. It applies the thread cap, which is process-wide in the
native libraries.

**Reason**

Section 2.2 makes the deadline part of the contract, and U9 says nothing hangs past it. Only another
process can stop a native call that is stuck. The owner has decided that SER runs in its own process at
the lowest priority, beside its consumer. That decision does not remove the deadline; it makes the
priority and the thread cap part of the configuration.

**Reuse of what the repository has**

- Reused:
  - the spawn start method, as `fast_public_boundary.py` uses through `mp.get_context` (line 260);
  - `worker_lifecycle.terminate_worker_process` (line 136);
  - `worker_lifecycle.recv_worker_message` (line 34).
- Not reused:
  - `run_process_setup_compute_handshake` (`worker_lifecycle.py:150`), and the medium and accurate
    `run_with_process_timeout`: each starts one process per call, which breaks load-once;
  - `run_with_timeout` (`worker_lifecycle.py:98`), which is thread-based and cannot stop native code;
  - the module-global `_SINGLE_FLIGHT_REGISTRY` (`fast_public_boundary.py:65`), because instances are
    independent (D11).
- Nothing in the repository sets a nice value or caps thread pools. The worker adds both.

**Tests (`test_analyzer_deadline.py` unless named)**

| Pins | Test |
|---|---|
| The formula table | `test_deadline_is_the_floor_or_two_seconds_per_audio_second` |
| Both branches of the formula | `test_both_deadline_branches_allow_work_within_them` |
| A hung backend returns `timeout` on time; one restart and one file read | `test_a_hung_backend_times_out_on_time_and_the_worker_comes_back` |
| A queued call times out without evicting | `test_a_call_that_times_out_in_the_queue_does_not_evict_the_running_one` |
| A dead worker | `test_a_worker_that_dies_is_a_model_error_and_is_replaced` |
| A failed restart | `test_a_failed_restart_is_a_model_error_and_the_next_call_retries` |
| Inline is not a hard bound | `test_inline_isolation_checks_the_deadline_between_stages_only` |
| Worker priority and thread cap | `test_analyzer_config.py::test_worker_runs_at_low_priority_with_capped_threads_and_no_gpu` |
| Inline leaves the caller alone | `test_analyzer_config.py::test_inline_mode_leaves_the_caller_process_alone` |

### D4. Models load through one artifact protocol with digest-bound provenance (lead ruling 3)

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
4. Require `model_id` and `label_set_id`.
5. Require the artifact's profile to match.
6. Require canonical training audio (D6).

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
digest means it describes the bytes that run.

**Tests (`test_analyzer_load.py`, with synthetic artifacts)**

- `test_safe_artifact_manifest_is_its_verified_identity_and_licence`
- `test_inspection_reads_the_licence_without_loading_the_model`
- `test_allow_list_hand_off_pins_the_inspected_bytes`
- `test_unverifiable_safe_artifacts_never_load`: tampered or missing provenance, with and without the opt-in
- `test_safe_artifacts_must_name_model_and_label_set`
- `test_an_artifact_for_another_profile_is_a_load_error`
- `test_a_pickle_is_refused_before_it_is_unpickled`: a booby-trapped pickle

### D5. Identity is one-to-one with the loaded bytes

**Rule**

`model_sha256` is the SHA-256 of the bytes that were loaded, in full lower-case hex.

- **Safe artifacts.** These must carry `model_id` and `label_set_id`.
- **Baselines.** The id is `baseline:<name>@<full sha256>`. `<name>` is the artifact's `model_id`, or
  else the legacy `model_revision`, or else `<profile>-<backend_id>`.
- **Legacy label sets.** `label_set_id` is `"labels:" + sha256(json.dumps(sorted(set(labels)), separators=(",", ":")))`.

**Reason**

A 12-hex prefix is not guaranteed unique, and equal explicit ids could name unequal bytes. The canonical
JSON form cannot collide the way a delimiter-joined list can.

**Tests (`test_analyzer_load.py`)**

- `test_legacy_pickle_loads_only_as_a_baseline_with_exact_ids`
- `test_legacy_model_revision_names_the_baseline`
- `test_model_files_differing_in_weights_or_labels_have_distinct_ids`

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

`ser._internal.analysis.audio_input` calls it as `audio_utils.to_canonical_audio`, exactly once per call,
and re-exports `CanonicalAudio` and `CANONICAL_SAMPLE_RATE` from the same module. Nothing mixes down or
resamples before it. Training uses the same function.

**Reason**

If training and inference both use one conversion, they cannot drift apart. Mixing down before the seam
would make it something other than the shared train and inference conversion.

**Tests (`test_analyzer_inputs.py`)**

| Pins | Test |
|---|---|
| Native channels and rate reach the seam; mono 16 kHz comes out | `test_native_channels_and_rate_reach_the_shared_seam` (a spy at the seam) |
| A signal-sensitive backend agrees within 0.05 across 8, 16, 44.1 and 48 kHz, for arrays and files | `test_every_input_rate_is_resampled_to_the_same_scores` |
| Unequal stereo channels give the mix, not one channel | `test_stereo_is_mixed_by_the_canonical_conversion_not_by_dropping_a_channel` |

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
- **Timing.** `timing_ms.features` covers decode, conversion and extraction. `timing_ms.predict` covers
  scoring. Both are whole milliseconds, and a stage that never started reports 0.

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

**Rule: scores**

The backend's utterance scores are the one authority for `labels`, `top_label` and `confidence`. Ties go
to the alphabetically first label.

Scores that the record may not carry make the status `model_error`. They are not renormalised. These are:
- keys other than the label set;
- values that are not finite, or are negative;
- utterance or frame scores whose sum is off by more than 1e-6;
- frames that leave a gap from 0 to the duration, or lie outside it.

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

**Tests (`test_analyzer_record.py`, unless named)**

| Pins | Test |
|---|---|
| The `fast` frame layout | `test_fast_frame_layout`, `test_fast_records_use_the_fast_frame_layout` |
| The pooling rule | `test_coverage_weighted_mean_weighs_every_instant_once`, `test_fast_utterance_scores_are_the_coverage_weighted_mean_of_its_frames` |
| The tie-break | `test_ties_go_to_the_alphabetically_first_label` |
| The backend is the authority | `test_backend_utterance_scores_are_the_authority_for_top_label` |
| Invalid scores are `model_error` | `test_analyzer_statuses.py::test_invalid_backend_scores_are_a_model_error_not_renormalised` |

### D11. Statuses, exceptions and their order

**Rule: statuses**

Bad audio gives a status. The checks run in order, and the first that applies wins:

| Status | When |
|---|---|
| `unreadable` | the file is missing, empty or a directory; its header or payload is truncated; it is not audio; or the samples include NaN or infinity |
| `too_short` | the duration is below `min_duration_s` |
| `too_long` | the duration is above `max_duration_s` |
| `silent` | see D12 |
| `timeout` | the deadline passed (D3) |
| `model_error` | the backend raised, the worker died, or the scores are invalid (D10) |
| `ok` | the analysis finished |

**Rule: exceptions**

Programming errors and unusable models raise. All exceptions derive from `AnalysisError`:

| Exception | When |
|---|---|
| `AnalysisUsageError` | a misuse (D8); an invalid config; a profile without a backend; a call after `close` |
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
below the configured threshold.

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
- **Logging and warnings.** No logger is reconfigured, and no handler is added. No warnings filter
  changes, and no warning is emitted. Nothing is logged at WARNING or above on any path, and no log line
  contains caller paths.
- **Determinism.** The same audio and the same model give the same record apart from `timing_ms`.

**Reason**

The consumer embeds SER beside a voice-reply path and must see neither side effects nor reloads.

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
only the parent. In the worker, network isolation is pinned only through inline mode and the read-only
`TMPDIR`.

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
3. Network isolation of the worker is only partly pinned (D14).
