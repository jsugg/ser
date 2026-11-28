"""Manifest and source accounting for the human portion of MESD v5."""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ser._internal.data.catalog.public_datasets import (
    MESD_CORPUS_ID,
    MESD_DATASET_LICENSE_ID,
    MESD_DATASET_POLICY_ID,
    MESD_DATASET_REVISION,
    MESD_DEFAULT_VERSION,
    MESD_EXTRACT_DIR_NAME,
    MESD_MENDELEY_DATASET_ID,
    MESD_SOURCE_INVENTORY_SHA256,
    MESD_SOURCE_URL,
)
from ser._internal.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser._internal.data.manifest_jsonl import write_manifest_jsonl
from ser._internal.data.ontology import LabelOntology
from ser._internal.data.public_dataset_label_inference import MesdFilename, parse_mesd_filename

_REDUCTION_FOLDERS = {"Level1": "L1", "Level2": "L2"}


def _matches_reduction_folder(path: Path, parsed: MesdFilename) -> bool:
    levels = [_REDUCTION_FOLDERS[part] for part in path.parts[:-1] if part in _REDUCTION_FOLDERS]
    return levels == ([parsed.reduction_level] if parsed.reduction_level is not None else [])


def _normalized_pcm_sha256(path: Path) -> str:
    # Match the readiness audit's sample rate + little-endian mono float32 identity.
    frames, rate = sf.read(path, dtype="float32", always_2d=True)
    if not frames.size:
        raise ValueError(f"MESD audio is empty: {path}")
    mono = frames.mean(axis=1, dtype=np.float32).astype("<f4", copy=False)
    digest = hashlib.sha256(str(rate).encode("ascii"))
    digest.update(mono.tobytes())
    return digest.hexdigest()


def build_mesd_manifest_jsonl(
    *, dataset_root: Path, ontology: LabelOntology, output_path: Path
) -> list[Utterance]:
    """Writes MESD's test manifest plus a row ledger and raw file SHA-256 list."""
    source_root = dataset_root / "raw" / MESD_EXTRACT_DIR_NAME
    marker = source_root / (
        f".extract-ok-{MESD_MENDELEY_DATASET_ID}-{MESD_DEFAULT_VERSION}.zip.json"
    )
    utterances: list[Utterance] = []
    dispositions: list[tuple[str, str, str | None, str | None, str | None, str | None]] = []
    checksums: list[str] = []
    human_files: dict[str, list[tuple[Path, MesdFilename]]] = defaultdict(list)
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path == marker:
            continue
        relative = path.relative_to(source_root).as_posix()
        with path.open("rb") as handle:
            source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        checksums.append(f"{source_hash}  {relative}\n")
        parsed = parse_mesd_filename(path)
        if parsed is None or not _matches_reduction_folder(Path(relative), parsed):
            reason = "invalid_mesd_filename" if path.suffix.lower() == ".wav" else "not_audio"
            dispositions.append((relative, reason, None, None, None, None))
            continue
        if parsed.reduction_level is not None:
            dispositions.append(
                (
                    relative,
                    "naturalness_reduced",
                    parsed.emotion,
                    parsed.voice_type,
                    parsed.reduction_level,
                    None,
                )
            )
            continue
        human_files[source_hash].append((path, parsed))

    checksum_text = "".join(checksums)
    actual_inventory = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    if actual_inventory != MESD_SOURCE_INVENTORY_SHA256:
        raise ValueError(
            f"MESD v5 source inventory SHA-256 mismatch under {source_root}: "
            f"expected {MESD_SOURCE_INVENTORY_SHA256}, got {actual_inventory}."
        )

    for copies in human_files.values():
        conflicting = len({(parsed.emotion, parsed.voice_type) for _, parsed in copies}) > 1
        first = copies[0][0].relative_to(source_root).as_posix()
        for path, parsed in copies:
            relative = path.relative_to(source_root).as_posix()
            if conflicting:
                status, duplicate_of = "conflicting_duplicate", None
            elif relative != first:
                status, duplicate_of = "duplicate_source_file", first
            else:
                status, duplicate_of = "used", None
                row = Utterance(
                    schema_version=MANIFEST_SCHEMA_VERSION,
                    sample_id=f"{MESD_CORPUS_ID}:{relative}",
                    corpus=MESD_CORPUS_ID,
                    audio_path=path,
                    label=parsed.label,
                    raw_label=parsed.emotion,
                    voice_type=parsed.voice_type,
                    language="es",
                    split="test",
                    normalized_audio_sha256=_normalized_pcm_sha256(path),
                    dataset_revision=MESD_DATASET_REVISION,
                    dataset_policy_id=MESD_DATASET_POLICY_ID,
                    dataset_license_id=MESD_DATASET_LICENSE_ID,
                    source_url=MESD_SOURCE_URL,
                )
                row.validate(ontology=ontology)
                utterances.append(row)
            dispositions.append(
                (relative, status, parsed.emotion, parsed.voice_type, None, duplicate_of)
            )

    with (dataset_root / "mesd-dispositions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "source_path",
                "disposition",
                "source_emotion",
                "voice_type",
                "reduction_level",
                "duplicate_of",
            )
        )
        dispositions.sort(key=lambda row: row[0])
        writer.writerows(dispositions)
    (dataset_root / "mesd-SHA256SUMS").write_bytes(checksum_text.encode("utf-8"))
    write_manifest_jsonl(output_path, utterances, base_dir=dataset_root)
    return utterances
