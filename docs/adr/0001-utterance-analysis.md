# ADR 0001: Utterance analysis interface

- Status: proposed (interface and tests only; implementation is brief A1-2)
- Contract source: SER plan section 2.2 "The integration contract SER must meet" and checklist U1-U14

## Context

`ser.api.infer()` is the only entry point today. It takes a file path, reloads the model on every call,
prints a timeline to standard output and transcribes by default. A consuming application needs a quiet,
load-once call that analyses one utterance and returns one JSON-serialisable record. This ADR fixes the
names, types, statuses and rules of that call. The stubs in `ser/_internal/analysis/` raise
`NotImplementedError`; the tests in `tests/suites/unit/analysis/` pin every clause and are marked
`xfail(strict=True, raises=NotImplementedError)` until the implementation lands.

## Decision

### Public names and their module

| Name | Kind | Defined in |
|---|---|---|
| `UtteranceAnalyzer` | class: `load(...)`, `analyze(...)`, `model_identity`, `close()`, context manager | `ser._internal.analysis.analyzer` |
| `AnalysisError` | base exception | `ser._internal.analysis.errors` |
| `AnalysisUsageError` | `AnalysisError`, `ValueError` | `ser._internal.analysis.errors` |
| `AnalysisModelUnavailableError` | `AnalysisError` | `ser._internal.analysis.errors` |
| `AnalysisModelLoadError` | `AnalysisError` | `ser._internal.analysis.errors` |
| `utterance_record_json_schema()` | returns the JSON Schema (draft 2020-12) of the record | `ser._internal.analysis.record_schema` |
| `UtteranceRecord`, `UtteranceFrame`, `UtteranceAudio`, `UtteranceTiming` | `TypedDict` | `ser.runtime.schema` |
| `UtteranceStatus` | `Literal` of the seven statuses | `ser.runtime.schema` |
| `UTTERANCE_SCHEMA_VERSION` | `"ser.utterance/v1"` | `ser.runtime.schema` |
| `AnalysisConfig` | frozen dataclass of every numeric limit | `ser.runtime.contracts` |
| `ModelIdentity` | frozen dataclass: profile, model id, SHA-256, label set id, labels | `ser.runtime.contracts` |
| `LanguageHint` | `Literal["en", "es", "pt"]` | `ser.runtime.contracts` |

The public module for all of them is `ser.api`. Data types live in the contract leaves
`ser.runtime.schema` and `ser.runtime.contracts`, as every other public record type does, so that
consumers and internal code share one definition without importing behaviour. Behaviour lives in
`ser._internal.analysis`, behind the facade, like the rest of the runtime.

The `ser.api` exports are not added in this change. `tests/suites/integration/architecture/test_api_import_boundary.py`
pins `ser.api.__all__` and says growth requires explicit API review, and
`tests/suites/integration/architecture/public_api_snapshot.json`
records the public surface. Brief A1-2 adds the exports together with those two updates, so the review
happens once, with a working implementation behind the names. Until then the tests import from the
modules above; only the import lines change when the exports land.

```python
from ser.api import UtteranceAnalyzer  # after A1-2

with UtteranceAnalyzer.load(profile="fast") as analyzer:
    record = analyzer.analyze("clip.wav")
    record = analyzer.analyze(samples, sample_rate_hz=48_000, language="es")
```

`load` takes `profile` as a required keyword. A later change of the default profile can then never
change a caller's results silently. `model_path` overrides the configured model file; `settings`
overrides the ambient `AppConfig`; `config` overrides the limits.

### Inputs

- `analyze(path)`: a `str` or `os.PathLike` to an audio file. Any rate and channel count that
  `soundfile` decodes is accepted; channels are mixed down.
- `analyze(samples, *, sample_rate_hz=...)`: a one-dimensional `numpy.float32` array and a positive
  `int` rate. Other dtypes, shapes, lists, a missing or non-positive rate, and a path given with a rate
  raise `AnalysisUsageError`. Float64 is rejected instead of being converted so that the caller, not SER,
  decides on the precision loss.
- `language`: `None`, `"en"`, `"es"` or `"pt"`. Any other value raises `AnalysisUsageError`. The fast
  backend ignores the hint; it is accepted so that the call does not change when a language-aware
  backend arrives.

Both forms are converted to the canonical 16 kHz mono form in exactly one place,
`ser._internal.analysis.audio_input.to_canonical`. That function is the seam where the analyser calls the
canonical-audio function owned by brief M5-1; this design contains no resampler. Training and inference
must use the same function (plan section 2.2, "Input").

### The record `ser.utterance/v1`

`analyze` returns `UtteranceRecord`, a `TypedDict` with exactly the 13 fields of the plan's example.
`utterance_record_json_schema()` is the machine-checkable definition; it is closed
(`additionalProperties: false`), so adding, removing or retyping a field requires a new
`schema_version`. Old records stay readable because the version is in every record.

- `labels`: probability per label of the model's label set; keys are exactly the model's labels; values
  sum to 1 within 1e-6. Probabilities are not rounded.
- `top_label`: the label with the highest probability; ties go to the label name that sorts first.
- `confidence`: `labels[top_label]`. Calibration is a property of the model (benchmark workstream), not
  of this interface.
- `frames`: one entry per backend analysis window, sorted, inside `[0, duration_s]`, with start and end
  rounded to milliseconds and per-frame `labels` that sum to 1. For the fast backend the windows are the
  model metadata's frame size and stride. The utterance `labels` are the duration-weighted mean of the
  frame probabilities, renormalised. A backend that scores the whole utterance returns one frame.
- `dimensions`: always `null` in v1. The plan reserves it for valence and arousal; a model that
  provides them changes what the field means, so it comes with `ser.utterance/v2`.
- `audio.duration_s`: input samples divided by the input rate, rounded to milliseconds;
  `audio.input_sample_rate_hz`: the caller's or the file's rate. Both are `null` only when the input
  could not be decoded.
- `timing_ms.features` and `timing_ms.predict`: whole milliseconds; a stage that did not run reports 0.
  Timing is the only field excluded from determinism.
- `schema_version`: `"ser.utterance/v1"`; `ser_version`: `ser.__version__`.

### Identity of today's model

`load` reads the model file's bytes once, computes their SHA-256 and deserialises from the same bytes, so
the hash names exactly what was loaded and later changes to the file have no effect.

- `model_sha256`: SHA-256 of those bytes, lower-case hex.
- `model_id`: the artifact metadata's `model_id` if present; otherwise
  `<profile>-<backend_id>-<first 12 hex of model_sha256>`. Today's artifacts have no `model_id`, and the
  derived form gives every distinct file a distinct id.
- `label_set_id`: the metadata's `label_set_id` if present; otherwise `labels-<first 12 hex of the
  SHA-256 of the sorted labels joined by newlines>`. The same labels give the same id across retrains; a
  changed label set gives a new id.
- `UtteranceAnalyzer.model_identity` returns these values without analysing anything.

The model artifact is still a pickle (`ser_model.pkl`), which the plan rules out. Its replacement is not
part of this interface: the identity rules above do not depend on the file format.

### Statuses and exceptions

Bad audio gives a `status`; programming errors and unusable models raise. Statuses are checked in this
order, and the first that applies wins:

| Status | When |
|---|---|
| `unreadable` | the file is missing, empty, a directory, truncated or not audio; or the array contains NaN or infinity |
| `too_short` | duration below `min_duration_s` (an empty array included) |
| `too_long` | duration above `max_duration_s`; bounds are inclusive and compared on the unrounded duration |
| `silent` | whole-utterance RMS below `silence_rms_dbfs` |
| `timeout` | the deadline passed during analysis |
| `model_error` | the backend raised during feature extraction or prediction |
| `ok` | the analysis finished |

Any status but `ok` has `labels`, `top_label`, `confidence` and `frames` set to `null` and still carries
every identity field, `audio` (as far as known) and `timing_ms`.

Raised, all subclasses of `AnalysisError`:

| Exception | When |
|---|---|
| `AnalysisUsageError` | wrong argument types or shapes, unknown language, invalid `AnalysisConfig`, a profile without an analysis backend, `analyze` after `close` |
| `AnalysisModelUnavailableError` | no model file at load; SER never downloads one |
| `AnalysisModelLoadError` | the model file cannot be deserialised or does not match the profile, backend or feature size |

Missing audio is a status because it describes the input; a missing model is an exception because no
input can succeed. Messages never contain caller-supplied paths; `__cause__` is not part of the contract.
`MemoryError`, `KeyboardInterrupt` and other `BaseException`s propagate unchanged.

### Limits

Every numeric limit is a field of `AnalysisConfig`, with the plan's proposed values as defaults:
`min_duration_s=0.3`, `max_duration_s=120.0`, `timeout_floor_s=10.0`, `timeout_per_audio_second=2.0`,
`silence_rms_dbfs=-60.0`, `max_threads=2`, `memory_budget_mb=1024`. They are proposals awaiting the
owner's decision (D5); no other module repeats them. `silence_rms_dbfs` is not in the plan and is proposed
here.

The deadline is `max(timeout_floor_s, timeout_per_audio_second × duration_s)`, measured from the start of
`analyze` and checked between stages and frames. It is cooperative: one long native call can overrun it.
A hard limit that can stop a stuck native call needs a separate process, which the batch runner (A1-3) and
the consuming application own; the in-process deadline guarantees `analyze` returns `timeout` instead of
results that arrive too late.

`max_threads` caps the native thread pools during `analyze`. `memory_budget_mb` is not enforced at run
time; the performance harness measures against it.

### Sync, not async

`analyze` is synchronous. The work is CPU-bound, so an `async` method would either block the event loop
or hide a thread pool whose size SER cannot choose well. Async callers use `asyncio.to_thread` or their
own executor.

### Thread-safety

- One analyser shared by several threads is safe. Calls on one instance run one at a time under a
  per-instance lock, so results equal sequential results.
- Two analysers in two threads are independent. Each holds its own model in memory (memory doubles) and
  there is no module-level cache or registry; the single-flight registry of the existing fast runtime
  path is not used.
- The thread cap is process-wide in the native libraries, so concurrent analysers share it.

### Load once, quiet and offline

- The model file is read during `load` and never again; deleting it afterwards does not affect `analyze`.
- No network access during `load` or `analyze`.
- Nothing is written to standard output or standard error and no files are created.
- Logging uses the logger `ser.analysis` only. The analyser adds no handlers, does not touch the root
  logger and does not use `ser._internal.utils.logger.get_logger`, which configures the root logger.
  Log lines never contain audio, file contents or caller-supplied paths (audio or model). Levels follow
  the caller's logging configuration; nothing is logged above `DEBUG` on the success path.
- Same audio and same model on the same machine and environment give the same record apart from
  `timing_ms`. Bitwise equality across different numeric library builds is not promised.

### A second backend

`ser._internal.analysis.backend.AnalysisBackend` is the seam: a `Protocol` with `identity`,
`predict(samples, *, deadline_monotonic_s)` returning per-frame scores and stage timings, and `close()`.
`backend_loaders()` maps each supported profile to a loader. Adding `medium` registers one loader; the
public call stays `UtteranceAnalyzer.load(profile="medium")`. Status checks, aggregation, identity and
the record are shared by every backend. Today only `fast` is supported; `accurate` and
`accurate-research` raise `AnalysisUsageError`, and `medium` does so until its backend lands.

### What `infer()` keeps doing

`ser.api.infer()` is unchanged: it takes a file path, loads the model on every call, returns the existing
timeline result, prints as before and transcribes by default. It is not deprecated by this change.

## Clause-to-test map

All paths are under `tests/suites/unit/analysis/`.

| Clause | Test |
|---|---|
| Record schema and its examples (U3) | `test_record_schema.py` |
| Path, `PathLike` and array inputs agree (U2) | `test_analyzer_inputs.py::test_wav_path_may_be_str_or_pathlike`, `::test_samples_and_float_wav_of_the_same_audio_give_the_same_record` |
| 8, 22.05, 44.1, 48 kHz, stereo and PCM input (U2) | `test_analyzer_inputs.py::test_any_input_rate_is_converted_and_reported`, `::test_multichannel_wav_is_mixed_down`, `::test_integer_pcm_wav_is_accepted` |
| Optional language hint | `test_analyzer_inputs.py::test_language_hint_is_optional`, `::test_unknown_language_hint_is_a_usage_error` |
| Malformed arguments raise | `test_analyzer_inputs.py::test_malformed_arguments_are_usage_errors` |
| Every field of an `ok` record, frames (U3) | `test_analyzer_record.py::test_ok_record_meets_every_record_clause`, `::test_frames_cover_the_whole_utterance` |
| Versioning and identity fields | `test_analyzer_record.py::test_record_names_versions_and_the_exact_model_file`, `::test_another_model_file_has_another_model_id_and_the_same_label_set`, `::test_a_changed_label_set_has_another_label_set_id` |
| `too_short`, `too_long`, `silent` (U9) | `test_analyzer_statuses.py::test_duration_and_level_limits`, `::test_limits_come_from_the_configuration` |
| `unreadable` (U9) | `test_analyzer_statuses.py::test_undecodable_files_are_unreadable`, `::test_non_finite_samples_are_unreadable` |
| `timeout`, `model_error` | `test_analyzer_statuses.py::test_exceeding_the_deadline_is_a_timeout`, `::test_a_failing_model_is_a_model_error` |
| Exception family | `test_analyzer_errors.py` |
| Load once (U6) | `test_analyzer_runtime.py::test_model_is_read_once_at_load` |
| No network, no output, no files, quiet logs (U5) | `test_analyzer_runtime.py::test_load_and_analysis_use_no_network`, `::test_nothing_is_printed_or_written`, `::test_root_logging_is_left_alone`, `::test_logs_never_contain_caller_paths` |
| Determinism (U4) | `test_analyzer_runtime.py::test_results_are_deterministic` |
| Thread-safety | `test_analyzer_runtime.py::test_one_analyzer_can_be_shared_by_threads`, `::test_two_analyzers_in_two_threads_are_independent` |

The tests use generated tones, noise and silence and small generated stand-in models; their
probabilities mean nothing, so numeric agreement across rates (U2, within 0.05) is left to the
canonical-audio work and the consumer suite. Performance and memory (U7, U8), batch resume (U10) and
licence records (U13) are outside this interface.

## Consequences

- Consumers get one call whose record is stable under a closed, versioned schema.
- The stubs and tests let the implementation and a consumer contract suite proceed in parallel against
  fixed names.
- Making the call public requires one API review in A1-2: the `ser.api` exports, the tier-1 export list
  and the public API snapshot.

## Open questions

1. Silence threshold: -60 dBFS RMS is proposed here; the plan has no number.
2. "Hard timeout": the in-process deadline is cooperative. Recommended reading: the analyser returns
   `timeout` at the next stage boundary, and a hard kill is the caller's worker process.
3. Model format: today's model is a pickle, which the plan rules out. Recommended: the identity rules
   stay; the format change belongs to the model-packaging work.
4. Canonical rate: today's fast model was trained on audio not converted to 16 kHz. Recommended: until
   it is retrained on the canonical form, the implementation converts to 16 kHz and records the mismatch
   as a known accuracy risk.
5. Unknown language hints raise rather than being ignored, and float64 arrays are rejected rather than
   converted. Both keep caller mistakes visible; either can be relaxed later without a schema change.
6. `dimensions` stays `null` in v1; values require v2.
