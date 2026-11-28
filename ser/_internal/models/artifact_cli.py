"""Model inspection and explicit trusted migration from legacy pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import fields
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ser._internal.analysis.artifact import read_model_file
from ser._internal.analysis.errors import AnalysisError
from ser._internal.models.artifact_safe import (
    encode_model,
    read_conversion_provenance,
    verify_artifact,
)
from ser.runtime.contracts import LicenceRecord, TrainingAudioForm

_TRUST_WARNING = "Conversion unpickles and can execute code. Only convert files you trust."


def _enabled_features(dimension: int | None) -> tuple[str, ...] | None:
    from ser._internal.repr.handcrafted import HandcraftedBackend
    from ser.config import FeatureFlags

    names = tuple(field.name for field in fields(FeatureFlags))
    for combination in product((False, True), repeat=len(names)):
        flags = dict(zip(names, combination, strict=True))
        if HandcraftedBackend(feature_flags=FeatureFlags(**flags)).feature_dim == dimension:
            return tuple(name for name in names if flags[name])
    return None


def _convert(source: Path, destination: Path, provenance: Path | None) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("The source and destination must be different files.")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise ValueError("The destination already exists.")
    audio, datasets = _read_provenance(provenance) if provenance is not None else (None, ())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / "model"
        with staged.open("xb") as stream:
            stream.write(_encode_legacy(read_model_file(source), audio, datasets))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, destination)


def _read_provenance(path: Path) -> tuple[TrainingAudioForm, tuple[LicenceRecord, ...]]:
    try:
        return read_conversion_provenance(read_model_file(path))
    except AnalysisError:
        raise ValueError("The provenance file could not be read.") from None
    except ValueError as err:
        raise ValueError(f"Invalid provenance file. {err}") from None


def _encode_legacy(
    data: bytes, audio: TrainingAudioForm | None, datasets: tuple[LicenceRecord, ...]
) -> bytes:
    from ser._internal.analysis.legacy import load_legacy

    try:
        loaded, version_mismatch = load_legacy(data)
    except MemoryError:
        raise
    except Exception as err:
        raise ValueError("Could not read the trusted legacy model envelope.") from err
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
    if loaded.expected_feature_size != getattr(loaded.model, "n_features_in_", None):
        raise ValueError("Legacy model feature dimensions disagree.")
    return encode_model(
        loaded.model,
        model_id=model_id,
        enabled_features=_enabled_features(loaded.expected_feature_size),
        frame_size_seconds=cast(float, metadata["frame_size_seconds"]),
        frame_stride_seconds=cast(float, metadata["frame_stride_seconds"]),
        pooling_strategy=cast(str, metadata["pooling_strategy"]),
        training_audio=audio,
        datasets=datasets,
        split_ledger_digest=split if isinstance(split, str) else None,
        created_at=created if isinstance(created, str) else None,
    )


def run_models_command(argv: Sequence[str]) -> int:
    """Runs the model inspection or conversion subcommand."""
    parser = argparse.ArgumentParser(prog="ser models", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    show = commands.add_parser("show", help="Verify and print a safe model manifest.")
    show.add_argument("file", type=Path)
    convert = commands.add_parser("convert", help=_TRUST_WARNING, description=_TRUST_WARNING)
    convert.add_argument("--trust-pickle", action="store_true", help=_TRUST_WARNING)
    convert.add_argument(
        "--provenance", type=Path, help="JSON file containing datasets and training_audio."
    )
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
            _convert(args.pickle, args.out, args.provenance)
    except (AnalysisError, OSError, ValueError) as err:
        print(f"ser models {args.command}: {err}", file=sys.stderr)
        return 1
    return 0
