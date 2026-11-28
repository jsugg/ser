# ADR 0002: Resumable batch analysis

## Input and output

`ser analyze --manifest in.jsonl --out out.jsonl` reads UTF-8 JSON Lines. Each line
must be an object with a unique, opaque string `id` and a nonempty string `path`.
Relative paths are relative to the manifest's directory. Other fields are ignored.
`--dir` instead selects direct `.wav` files (case-insensitive extension), sorted
by filename, using the full filename including extension as the ID. It follows
symlinked `.wav` files and skips symlink loops.

Each output line is `{"id": <input ID>, "record": <unchanged ser.utterance/v1 record>}`.
Rows are written in input order. Missing, corrupt and other per-file analysis
failures get their normal analysis status and count as completed. Malformed input
rows and duplicate IDs cannot be mapped unambiguously to results: the entire input
list is validated before writing anything.

## Resume and publication

Resume from a checkpoint produced by this command. Existing output must be a
complete, newline-terminated prefix of the current input IDs in order. Those IDs
are done, regardless of record status. Resume with unchanged inputs and options.
SER does not hash the audio or verify the source path for an existing ID. Use a
fresh output when files change or to retry a terminal status. If there is
unfinished work, the first record's model digest pins the
resume load; a completed output needs no model load.

On platforms with `fcntl`, SER acquires an exclusive, non-blocking `flock` on
`.<output name>.lock` in the output directory before reading the checkpoint or
loading the model. A competing writer exits 2 without writing output or loading
a model. The lock lasts through the run. The sidecar stays in place so future
writers lock the same file. On platforms without `fcntl`, the caller must ensure
one writer per output. SER skips stale staging cleanup on those platforms.

Publish each chunk of up to 32 rows by writing the prior output plus new rows to
a staging file named `.<output name>.ser-batch-XXXX` in the output directory,
where `XXXX` is a random suffix. SER flushes it with `fsync` and atomically
replaces the output. On POSIX a new output has mode `0600`, and a replacement
inherits the existing output's mode. A reader sees a complete snapshot, and
after SIGKILL at most one chunk is recomputed. At startup with the lock held,
SER removes stale staging files belonging to that output. The sidecar and
staging filenames are reserved for SER. The durability guarantee covers process
crashes. Put the output in an existing directory separate from the input audio
directories. SER refuses an `--out` symlink.

Copying the prefix at every checkpoint makes total bytes written grow with the
square of the row count. A write-only probe used 32-row chunks and 1,134-byte
JSONL lines.

| Rows | Final size (MB) | Total written (MB) | Seconds |
| ---: | ---: | ---: | ---: |
| 1,000 | 1.1 | 19.1 | 0.33 |
| 5,000 | 5.7 | 450.1 | 3.46 |
| 20,000 | 22.7 | 7,098.8 | 46.80 |

Use at most 5,000 rows per manifest. Split larger jobs into separate manifests
with separate outputs.

## Execution and exit

One public `UtteranceAnalyzer` handles pending files sequentially. `--model`
overrides the configured model for the `fast` profile; `--allow-legacy-baseline`
is required for trusted legacy pickles. `--threads` sets the worker's native
thread cap (default 2). The worker keeps its idle priority (nice 19) and its
analysis deadline (`max(10 s, 2 × audio duration)`); the caller is not reniced.

Exit 0 means every input has a committed row, regardless of row status. Exit 2
means invalid arguments, input, or existing output, or a held output lock. Exit 1
means an operational failure such as I/O after path validation or model load
failure. Before loading a model, SER creates and removes a staging file to check
that the output directory is writable. It holds the lock during this check on
platforms with `fcntl`.

| Path condition | Exit |
| --- | --- |
| `--manifest` missing or other than a regular file | 2 |
| `--dir` missing or other than a directory | 2 |
| `--out` parent missing or other than a directory | 2 |
| Existing `--out` symlink or other than a regular file | 2 |
| Output directory cannot be written | 1 |

An absent `--out` file is created on the first successful publish.
Diagnostics never include paths, IDs, exception text, or audio. `--progress`
reports count-only progress, which can include rows not yet committed.
`--summary` reports rows read, analysed this run, skipped, and status totals
across both groups. Both write to stderr, not stdout.
