"""Model inspection and explicit trusted migration from legacy pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from ser._internal.analysis.artifact import read_model_file
from ser._internal.analysis.errors import AnalysisError
from ser._internal.models.artifact_safe import encode_model, verify_artifact

_TRUST_WARNING = "Conversion unpickles and can execute code. Only convert files you trust."


def _convert(source: Path, destination: Path) -> None:
    from ser._internal.analysis.legacy import load_legacy
    from ser._internal.models.artifact_persistence import atomic_write_bytes

    if source.resolve() == destination.resolve():
        raise ValueError("The source and destination must be different files.")
    if destination.exists():
        raise ValueError("The destination already exists.")
    loaded, version_mismatch = load_legacy(read_model_file(source))
    metadata = loaded.artifact_metadata
    if metadata is None or metadata["profile"] != "fast" or metadata["backend_id"] != "handcrafted":
        raise ValueError("Only fast handcrafted legacy SER model envelopes can be converted.")
    if version_mismatch:
        print(
            "Legacy model was saved with a different scikit-learn version; "
            "check prediction parity with the original environment.",
            file=sys.stderr,
        )
    revision = metadata.get("model_revision")
    model_id = revision if isinstance(revision, str) else "fast-handcrafted"
    split = metadata.get("split_ledger_digest")
    created = metadata.get("created_at_utc")
    atomic_write_bytes(
        destination,
        encode_model(
            loaded.model,
            model_id=model_id,
            split_ledger_digest=split if isinstance(split, str) else None,
            created_at=created if isinstance(created, str) else None,
        ),
    )


def run_models_command(argv: Sequence[str]) -> int:
    """Runs the model inspection or conversion subcommand."""
    parser = argparse.ArgumentParser(prog="ser models", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="Verify and print a safe model manifest.")
    show.add_argument("file", type=Path)
    convert = commands.add_parser("convert", help=_TRUST_WARNING, description=_TRUST_WARNING)
    convert.add_argument("--trust-pickle", action="store_true", help=_TRUST_WARNING)
    convert.add_argument("pickle", type=Path)
    convert.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    if args.command == "convert" and not args.trust_pickle:
        convert.error(f"{_TRUST_WARNING} Pass --trust-pickle to proceed.")
    try:
        if args.command == "show":
            data = read_model_file(args.file)
            verified = verify_artifact(data)
            print(
                json.dumps(
                    {
                        "verified": True,
                        "model_sha256": hashlib.sha256(data).hexdigest(),
                        "manifest": verified.manifest,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print(_TRUST_WARNING, file=sys.stderr)
            _convert(args.pickle, args.out)
    except (AnalysisError, OSError, ValueError) as err:
        print(f"ser models {args.command}: {err}", file=sys.stderr)
        return 1
    return 0
