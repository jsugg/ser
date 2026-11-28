# ADR 0002: Resumable batch analysis

## Input and output

`ser analyze --manifest in.jsonl --out out.jsonl` reads UTF-8 JSON Lines. Each line
must be an object with a unique, opaque string `id` and a nonempty string `path`.
Relative paths are relative to the manifest's directory. Other fields are ignored.
`--dir` instead selects direct `.wav` files (case-insensitive extension), sorted
by filename, using the full filename including extension as the ID.

Each output line is `{"id": <input ID>, "record": <unchanged ser.utterance/v1 record>}`.
Rows are written in input order. Missing, corrupt and other per-file analysis
failures get their normal analysis status and count as completed. Malformed input
rows and duplicate IDs cannot be mapped unambiguously to results: the entire input
list is validated before writing anything.

## Resume and publication

Resume from a checkpoint produced by this command. Existing output must be a
complete, newline-terminated prefix of the current input IDs in order. Those IDs
are done, regardless of record status. Resume with unchanged inputs and options
and one writer per output. SER does not hash the audio or verify the source path
for an existing ID. Use a fresh output when files change or to retry a terminal
status. If there is unfinished work, the first record's model digest pins the
resume load; a completed output needs no model load.

Publish each chunk of up to 32 rows by writing the prior output plus new rows to
a private temporary file in the output directory, flushing it with `fsync`, and
atomically replacing the output there. A reader sees either the prior snapshot
or the complete new one, never a partial line. After SIGKILL, at most one chunk is
recomputed; abandoned temporary files are not results and can be removed once no
writer is running. Copying the prefix at each checkpoint trades write volume for
an ordinary, directly readable JSONL output. This is a process-crash guarantee,
not a power-loss guarantee. Put the output in an existing directory other than
the input audio directories; the batch never changes an input.

## Execution and exit

One public `UtteranceAnalyzer` handles pending files sequentially. `--model`
overrides the configured model for the `fast` profile; `--allow-legacy-baseline`
is required for trusted legacy pickles. `--threads` sets the worker's native
thread cap (default 2). The worker keeps its idle priority (nice 19) and its
analysis deadline (`max(10 s, 2 × audio duration)`); the caller is not reniced.

Exit 0 means every input has a committed row, even if some statuses are not
`ok`. Exit 2 means invalid CLI arguments, input, or existing output. Exit 1 means
an operational failure (including input/output I/O or model load failure).
Diagnostics never include paths, IDs, exception text, or audio. `--progress`
reports count-only progress, which can include rows not yet committed.
`--summary` reports rows read, analysed this run, skipped, and status totals
across both groups. Both write to stderr, not stdout.
