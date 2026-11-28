"""Versioned curation ledger for known duplicate clips in staged datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Literal, cast, get_args

import numpy as np
import soundfile as sf

from ser._internal.data.manifest import Utterance
from ser._internal.data.manifest_jsonl import write_manifest_jsonl
from ser._internal.utils.logger import get_logger

logger = get_logger(__name__)

type CurationReason = Literal["conflicting_label_duplicate", "exact_repeat_same_label"]
CURATION_REASONS: frozenset[str] = frozenset(get_args(CurationReason.__value__))
_BUILTIN_LEDGER_PATH = Path(__file__).with_name("curation_ledger_english_v1.json")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LFS_POINTER_TEMPLATE = (
    "version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
)


class CurationLedgerError(ValueError):
    """Raised when a ledger is malformed or disagrees with the audited data."""


@dataclass(frozen=True)
class LedgerEntry:
    """One excluded clip, identified by dataset-relative id and content hash."""

    corpus: str
    dataset_revision: str
    sample_id: str
    normalized_audio_sha256: str
    reason: CurationReason
    kept_sample_id: str | None = None

    def to_record(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible record."""
        record: dict[str, object] = {
            "corpus": self.corpus,
            "dataset_revision": self.dataset_revision,
            "sample_id": self.sample_id,
            "normalized_audio_sha256": self.normalized_audio_sha256,
            "reason": self.reason,
        }
        if self.kept_sample_id is not None:
            record["kept_sample_id"] = self.kept_sample_id
        return record


@dataclass(frozen=True)
class CurationExclusion:
    """An excluded row reported by the audit: id and reason only."""

    sample_id: str
    reason: CurationReason


@dataclass(frozen=True)
class CurationLedger:
    """Exclusion decisions per dataset revision."""

    ledger_id: str
    version: str
    entries: tuple[LedgerEntry, ...]

    def to_record(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible record."""
        return {
            "ledger_id": self.ledger_id,
            "version": self.version,
            "entries": [entry.to_record() for entry in self.entries],
        }


@dataclass(frozen=True)
class CurationResult:
    """Rows that remain after curation and the rows that were excluded."""

    utterances: tuple[Utterance, ...]
    excluded: tuple[CurationExclusion, ...]


def _text(record: dict[str, object], key: str, *, optional: bool = False) -> str | None:
    value = record.get(key)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CurationLedgerError(f"Ledger field {key!r} must be a non-empty string.")
    return value


def _parse_ledger(payload: object) -> CurationLedger:
    if not isinstance(payload, dict):
        raise CurationLedgerError("Curation ledger root must be a JSON object.")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise CurationLedgerError("Curation ledger needs an 'entries' list.")
    entries: list[LedgerEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise CurationLedgerError("Curation ledger entries must be objects.")
        reason = _text(raw, "reason")
        if reason not in CURATION_REASONS:
            raise CurationLedgerError(f"Unknown curation reason {reason!r}.")
        kept = _text(raw, "kept_sample_id", optional=True)
        if (reason == "exact_repeat_same_label") != (kept is not None):
            raise CurationLedgerError(
                "kept_sample_id is required for, and only valid with, exact_repeat_same_label."
            )
        digest = cast(str, _text(raw, "normalized_audio_sha256"))
        if not _SHA256_PATTERN.fullmatch(digest):
            raise CurationLedgerError("Ledger hashes must be 64 lowercase hexadecimal characters.")
        entries.append(
            LedgerEntry(
                corpus=cast(str, _text(raw, "corpus")),
                dataset_revision=cast(str, _text(raw, "dataset_revision")),
                sample_id=cast(str, _text(raw, "sample_id")),
                normalized_audio_sha256=digest,
                reason=cast(CurationReason, reason),
                kept_sample_id=kept,
            )
        )
    return CurationLedger(
        ledger_id=cast(str, _text(payload, "ledger_id")),
        version=cast(str, _text(payload, "version")),
        entries=tuple(entries),
    )


def load_curation_ledger(path: Path) -> CurationLedger:
    """Loads and validates a curation ledger file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise CurationLedgerError(f"Unable to load curation ledger {path.name}: {err}") from err
    return _parse_ledger(payload)


@cache
def load_builtin_curation_ledger() -> CurationLedger:
    """Returns the ledger shipped with the package."""
    return load_curation_ledger(_BUILTIN_LEDGER_PATH)


def normalized_audio_sha256(path: Path) -> str:
    """Streams the sample-rate plus mono float32 identity used by the readiness audit."""
    digest = hashlib.sha256()
    with sf.SoundFile(path) as handle:
        digest.update(str(handle.samplerate).encode("ascii"))
        for block in handle.blocks(blocksize=65_536, dtype="float32", always_2d=True):
            mono = np.asarray(block, dtype="<f4").mean(axis=1, dtype=np.float32)
            digest.update(np.asarray(mono, dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def media_digest(path: Path, cache: dict[Path, str]) -> str:
    """Returns the current content hash of a file, computed once per cache."""
    if path not in cache:
        cache[path] = normalized_audio_sha256(path)
    return cache[path]


def git_blob_sha1(path: Path) -> bytes:
    """Returns the git blob id of a file's bytes."""
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def git_lfs_pointer_sha1(path: Path) -> bytes:
    """Returns the git blob id of the Git LFS pointer that tracks a file."""
    with path.open("rb") as handle:
        oid = hashlib.file_digest(handle, "sha256").hexdigest()
    pointer = _LFS_POINTER_TEMPLATE.format(oid=oid, size=path.stat().st_size).encode("ascii")
    return hashlib.sha1(f"blob {len(pointer)}\0".encode("ascii") + pointer).digest()


def git_tree_sha1(blobs: Mapping[str, bytes]) -> str:
    """Returns the git tree id of regular files keyed by POSIX path relative to the tree."""
    files: dict[str, bytes] = {}
    subtrees: defaultdict[str, dict[str, bytes]] = defaultdict(dict)
    for rel, oid in blobs.items():
        head, _, rest = rel.partition("/")
        if rest:
            subtrees[head][rest] = oid
        else:
            files[head] = oid
    entries = [(name.encode(), b"100644 " + name.encode(), oid) for name, oid in files.items()]
    entries.extend(
        ((name + "/").encode(), b"40000 " + name.encode(), bytes.fromhex(git_tree_sha1(tree)))
        for name, tree in subtrees.items()
    )
    body = b"".join(header + b"\0" + oid for _key, header, oid in sorted(entries))
    return hashlib.sha1(f"tree {len(body)}\0".encode("ascii") + body).hexdigest()


def stamp_manifest_provenance(
    utterances: Sequence[Utterance], *, dataset_revision: str | None
) -> list[Utterance]:
    """Records each row's current content hash and the verified dataset revision."""
    return [
        replace(
            row,
            normalized_audio_sha256=normalized_audio_sha256(row.audio_path),
            dataset_revision=dataset_revision,
        )
        for row in utterances
    ]


def source_manifest_path(manifest_path: Path) -> Path:
    """Returns where the uncurated rows of a curated manifest are written."""
    return manifest_path.with_name(f"{manifest_path.stem}.source{manifest_path.suffix}")


def write_curated_manifest(
    utterances: Sequence[Utterance],
    *,
    output_path: Path,
    base_dir: Path,
    curation_ledger: CurationLedger | None,
) -> list[Utterance]:
    """Writes the ledger-curated rows to ``output_path`` and the source rows beside it.

    Without a ledger the source rows are written to ``output_path`` alone.
    """
    if curation_ledger is None:
        write_manifest_jsonl(output_path, utterances, base_dir=base_dir)
        return list(utterances)
    curated = apply_curation_ledger(utterances, curation_ledger, strict=False)
    write_manifest_jsonl(source_manifest_path(output_path), utterances, base_dir=base_dir)
    write_manifest_jsonl(output_path, curated.utterances, base_dir=base_dir)
    if curated.excluded:
        logger.info(
            "Curation ledger excluded %d row(s) from %s: %s",
            len(curated.excluded),
            output_path.name,
            "; ".join(f"{item.sample_id} ({item.reason})" for item in curated.excluded),
        )
    return list(curated.utterances)


def _dataset_root(rows: Sequence[Utterance]) -> Path | None:
    for row in rows:
        relative = row.sample_id.partition(":")[2]
        text = row.audio_path.as_posix()
        if relative and text.endswith(f"/{relative}"):
            return Path(text[: -len(relative) - 1])
    return None


def apply_curation_ledger(
    utterances: Sequence[Utterance],
    ledger: CurationLedger,
    *,
    strict: bool = True,
    media_digests: dict[Path, str] | None = None,
) -> CurationResult:
    """Drops ledger-listed rows after checking every entry against the rows and the media.

    Entries for a corpus that is not loaded are skipped. For a loaded corpus every entry
    must match the corpus and revision of the rows, name a clip whose current media still
    has the recorded content, and cover its whole duplicate group. A listed clip may be
    absent from the rows, as in a curated manifest; its file under the dataset root is
    checked instead. Rows without a dataset revision cannot be matched: strict mode
    rejects them, otherwise their corpus is left uncurated. ``media_digests`` caches the
    current content hashes so a caller can reuse them.
    """
    by_id = {row.sample_id: row for row in utterances}
    rows_by_scope: defaultdict[tuple[str, str | None], list[Utterance]] = defaultdict(list)
    for row in utterances:
        rows_by_scope[(row.corpus, row.dataset_revision)].append(row)
    loaded: defaultdict[str, set[str | None]] = defaultdict(set)
    for corpus, revision in rows_by_scope:
        loaded[corpus].add(revision)

    applicable: list[LedgerEntry] = []
    unversioned: set[str] = set()
    for entry in ledger.entries:
        revisions = loaded.get(entry.corpus)
        if revisions is None:
            continue
        if None in revisions:
            if strict:
                raise CurationLedgerError(
                    f"{entry.corpus} rows carry no dataset_revision, so the curation ledger "
                    "cannot be matched to them; rebuild the manifest to record it."
                )
            unversioned.add(entry.corpus)
            continue
        if entry.dataset_revision not in revisions:
            raise CurationLedgerError(
                f"Ledger entry {entry.sample_id!r} is for {entry.corpus} revision "
                f"{entry.dataset_revision!r}, but the loaded rows carry "
                f"{sorted(cast(set[str], revisions))}."
            )
        applicable.append(entry)
    for corpus in sorted(unversioned):
        logger.warning(
            "Curation ledger not applied to %s: its rows carry no dataset_revision; "
            "rebuild the manifest.",
            corpus,
        )

    excluded: dict[str, LedgerEntry] = {}
    for entry in applicable:
        if entry.sample_id in excluded:
            raise CurationLedgerError(f"Ledger lists {entry.sample_id!r} more than once.")
        excluded[entry.sample_id] = entry
    if {entry.kept_sample_id for entry in applicable} & excluded.keys():
        raise CurationLedgerError("A clip named as kept is also excluded by the ledger.")

    roots: dict[tuple[str, str], Path | None] = {}
    current: dict[Path, str] = {} if media_digests is None else media_digests

    def clip_path(corpus: str, revision: str, sample_id: str) -> Path:
        row = by_id.get(sample_id)
        if row is not None and (row.corpus, row.dataset_revision) == (corpus, revision):
            return row.audio_path
        if row is None and sample_id.startswith(f"{corpus}:"):
            if (corpus, revision) not in roots:
                roots[(corpus, revision)] = _dataset_root(rows_by_scope[(corpus, revision)])
            root = roots[(corpus, revision)]
            if root is not None and (root / sample_id.partition(":")[2]).is_file():
                return root / sample_id.partition(":")[2]
        raise CurationLedgerError(
            f"Ledger entry {sample_id!r} does not exist in {corpus} revision {revision!r}."
        )

    def verify_media(sample_id: str, path: Path, expected: str) -> None:
        if media_digest(path, current) != expected:
            raise CurationLedgerError(
                f"Media of {sample_id!r} changed since its hash was recorded: expected "
                f"{expected}, found {current[path]}. Rebuild the manifest and review the ledger."
            )

    groups: defaultdict[tuple[str, str, str], list[LedgerEntry]] = defaultdict(list)
    for entry in applicable:
        path = clip_path(entry.corpus, entry.dataset_revision, entry.sample_id)
        found = by_id.get(entry.sample_id)
        if found is not None and found.normalized_audio_sha256 != entry.normalized_audio_sha256:
            raise CurationLedgerError(
                f"Ledger hash mismatch for {entry.sample_id!r}: ledger "
                f"{entry.normalized_audio_sha256}, data {found.normalized_audio_sha256}."
            )
        verify_media(entry.sample_id, path, entry.normalized_audio_sha256)
        groups[(entry.corpus, entry.dataset_revision, entry.normalized_audio_sha256)].append(entry)

    for (corpus, revision, digest), members in groups.items():
        peers = [
            row
            for row in rows_by_scope[(corpus, revision)]
            if row.normalized_audio_sha256 == digest
        ]
        for row in peers:
            verify_media(row.sample_id, row.audio_path, digest)
        listed = {entry.sample_id for entry in members}
        reasons = {entry.reason for entry in members}
        if len(reasons) > 1:
            raise CurationLedgerError(f"Ledger mixes curation reasons for content {digest}.")
        if reasons == {"conflicting_label_duplicate"}:
            unlisted = sorted(row.sample_id for row in peers if row.sample_id not in listed)
            if unlisted:
                raise CurationLedgerError(
                    f"Ledger excludes only part of the conflicting-label group {digest}: "
                    f"{unlisted[0]!r} is not listed. Exclude every clip of the group."
                )
            if len(listed) < 2:
                raise CurationLedgerError(
                    f"Ledger entry {members[0].sample_id!r} is unused: no other clip shares "
                    "its content."
                )
            if len(peers) == len(listed) and len({row.label for row in peers}) < 2:
                raise CurationLedgerError(
                    f"Ledger entry {members[0].sample_id!r} is not a conflicting duplicate: "
                    "all clips with this content share one label."
                )
            continue
        for entry in members:
            kept_id = cast(str, entry.kept_sample_id)
            kept_path = clip_path(corpus, revision, kept_id)
            kept = by_id.get(kept_id)
            found = by_id.get(entry.sample_id)
            if kept is not None and (
                kept.normalized_audio_sha256 != digest
                or (found is not None and kept.label != found.label)
            ):
                raise CurationLedgerError(
                    f"Ledger entry {entry.sample_id!r} names kept clip {kept_id!r} that "
                    "differs in content or carries another label."
                )
            verify_media(kept_id, kept_path, digest)

    return CurationResult(
        utterances=tuple(row for row in utterances if row.sample_id not in excluded),
        excluded=tuple(
            CurationExclusion(sample_id=sample_id, reason=excluded[sample_id].reason)
            for sample_id in sorted(excluded)
        ),
    )
