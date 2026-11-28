"""Batch CLI over the public utterance analyzer."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Never, cast

from ser.api import (
    UTTERANCE_SCHEMA_VERSION,
    AnalysisConfig,
    AnalysisError,
    AnalysisModelNotConformingError,
    UtteranceAnalyzer,
    UtteranceRecord,
)
from ser.config import AppConfig

_CHUNK_ROWS = 32
_STATUSES = ("ok", "too_short", "too_long", "silent", "unreadable", "timeout", "model_error")


class _InvalidBatch(ValueError):
    """A batch contract violation with a content-free diagnostic."""


class _BatchParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        # argparse can embed unrecognized arguments and invalid values in message.
        super().error("invalid arguments (see ser analyze --help)")


class _BatchArgs(argparse.Namespace):
    manifest: Path | None
    directory: Path | None
    out: Path
    model: Path | None
    allow_legacy_baseline: bool
    threads: int
    progress: bool
    summary: bool


@dataclass(frozen=True)
class _InputRow:
    id: str
    path: Path


@dataclass(frozen=True)
class _Checkpoint:
    statuses: Counter[str]
    model_sha256: str | None


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _parser() -> _BatchParser:
    parser = _BatchParser(
        prog="ser analyze",
        description="Analyze stored recordings and resume from a JSONL checkpoint.",
        allow_abbrev=False,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest", type=Path, help="UTF-8 JSONL manifest with id and path fields"
    )
    source.add_argument("--dir", dest="directory", type=Path, help="Directory of WAV recordings")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output JSONL path in a separate directory"
    )
    parser.add_argument("--model", type=Path, help="Override the configured fast model")
    parser.add_argument(
        "--allow-legacy-baseline", action="store_true", help="Allow a trusted legacy model"
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=AnalysisConfig().max_threads,
        help="Maximum native threads in the analysis worker (default: %(default)s)",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Report count-only progress to stderr"
    )
    parser.add_argument("--summary", action="store_true", help="Report final JSON counts to stderr")
    return parser


def _object(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _reject_json_constant(_: str) -> Never:
    raise ValueError("Non-JSON numeric literal.")


def _json_object(line: bytes) -> dict[str, object] | None:
    try:
        return _object(json.loads(line.decode("utf-8"), parse_constant=_reject_json_constant))
    except (ValueError, RecursionError):
        return None


def _manifest_rows(manifest: Path) -> list[_InputRow]:
    if not stat.S_ISREG(manifest.stat().st_mode):
        raise _InvalidBatch("The manifest must be a regular file.")
    rows: list[_InputRow] = []
    seen: set[str] = set()
    with manifest.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            entry = _json_object(line)
            row_id = entry.get("id") if entry is not None else None
            path = entry.get("path") if entry is not None else None
            if not isinstance(row_id, str) or not isinstance(path, str) or not path or "\0" in path:
                raise _InvalidBatch(f"Invalid manifest row {number}.")
            try:
                os.fsencode(path)
            except UnicodeError:
                raise _InvalidBatch(f"Invalid path in manifest row {number}.") from None
            if row_id in seen:
                raise _InvalidBatch(f"Duplicate ID in manifest row {number}.")
            seen.add(row_id)
            rows.append(_InputRow(row_id, manifest.parent / path))
    return rows


def _inputs(args: _BatchArgs) -> tuple[list[_InputRow], Path]:
    if args.manifest is not None:
        rows = _manifest_rows(args.manifest)
    else:
        assert args.directory is not None
        rows = [
            _InputRow(path.name, path)
            for path in sorted(args.directory.iterdir())
            if path.suffix.lower() == ".wav" and path.is_file()
        ]

    parent = Path(os.path.realpath(args.out.parent, strict=True))
    out = parent / args.out.name
    if out.is_symlink() or any(
        path is not None and Path(os.path.realpath(path)) == out
        for path in (args.manifest, args.model)
    ):
        raise _InvalidBatch("The output must not alias an input or be a symbolic link.")
    if (
        args.directory is not None
        and Path(os.path.realpath(args.directory)) == parent
        or any(
            parent
            in (Path(os.path.realpath(row.path.parent)), Path(os.path.realpath(row.path)).parent)
            for row in rows
        )
    ):
        raise _InvalidBatch("The output must not be in an input audio directory.")
    return rows, out


def _read_checkpoint(out: Path, rows: list[_InputRow]) -> _Checkpoint | None:
    try:
        mode = out.stat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise _InvalidBatch("The existing output must be a regular file.")

    digest: str | None = None
    counts: Counter[str] = Counter()
    with out.open("rb") as stream:
        for index, line in enumerate(stream):
            entry = _json_object(line) if line.endswith(b"\n") else None
            if (
                entry is None
                or entry.keys() != {"id", "record"}
                or index >= len(rows)
                or entry["id"] != rows[index].id
            ):
                raise _InvalidBatch("The existing output is not an ordered input prefix.")
            record = _object(entry["record"])
            if (
                record is None
                or record.keys() != UtteranceRecord.__required_keys__
                or record["schema_version"] != UTTERANCE_SCHEMA_VERSION
            ):
                raise _InvalidBatch("The existing output contains an invalid utterance record.")
            status, sha256 = record["status"], record["model_sha256"]
            if (
                not isinstance(status, str)
                or status not in _STATUSES
                or not isinstance(sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
                or (digest is not None and digest != sha256)
            ):
                raise _InvalidBatch(
                    "The existing output contains an invalid status or model digest."
                )
            digest = sha256
            counts[status] += 1
    return _Checkpoint(counts, digest)


def _publish(out: Path, lines: list[bytes], *, has_checkpoint: bool) -> None:
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out.parent, prefix=".ser-batch-", delete=False
        ) as stream:
            staging = Path(stream.name)
            if has_checkpoint:
                with out.open("rb") as previous:
                    shutil.copyfileobj(previous, stream)
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, out)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def _run(args: _BatchArgs, settings: AppConfig) -> None:
    rows, out = _inputs(args)
    checkpoint = _read_checkpoint(out, rows)
    counts = checkpoint.statuses if checkpoint is not None else Counter()
    skipped = counts.total()
    has_checkpoint = checkpoint is not None
    if skipped < len(rows):
        with UtteranceAnalyzer.load(
            profile="fast",
            model_path=args.model,
            expected_sha256=checkpoint.model_sha256 if checkpoint is not None else None,
            allow_legacy_baseline=args.allow_legacy_baseline,
            config=AnalysisConfig(max_threads=args.threads),
            settings=settings,
        ) as analyzer:
            chunk: list[bytes] = []
            for index in range(skipped, len(rows)):
                row = rows[index]
                record = analyzer.analyze(row.path)
                counts[record["status"]] += 1
                line = json.dumps({"id": row.id, "record": record}, allow_nan=False)
                chunk.append(line.encode("ascii") + b"\n")
                if len(chunk) == _CHUNK_ROWS or index + 1 == len(rows):
                    _publish(out, chunk, has_checkpoint=has_checkpoint)
                    has_checkpoint = True
                    chunk.clear()
                if args.progress:
                    print(f"Analyzed {index + 1}/{len(rows)}", file=sys.stderr, flush=True)
    elif not has_checkpoint:
        _publish(out, [], has_checkpoint=False)

    if args.summary:
        summary = {
            "read": len(rows),
            "analyzed": len(rows) - skipped,
            "skipped": skipped,
            "statuses": counts,
        }
        print(json.dumps(summary, sort_keys=True), file=sys.stderr)


def run_analyze_command(argv: Sequence[str], *, settings: AppConfig) -> int:
    """Runs batch analysis with diagnostics that never echo caller content."""
    args = _parser().parse_args(argv, namespace=_BatchArgs())
    try:
        _run(args, settings)
    except _InvalidBatch as error:
        print(f"ser analyze: {error}", file=sys.stderr)
        return 2
    except AnalysisModelNotConformingError:
        print("ser analyze: model is not conforming; see --allow-legacy-baseline.", file=sys.stderr)
        return 1
    except AnalysisError:
        print(
            "ser analyze: analysis could not complete; check the model and settings.",
            file=sys.stderr,
        )
        return 1
    except (OSError, UnicodeError):
        print("ser analyze: batch input or output I/O failed.", file=sys.stderr)
        return 1
    return 0
