"""Versioned curation ledger for known duplicate clips in staged datasets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Literal, cast, get_args

import numpy as np
import soundfile as sf

from ser._internal.data.manifest import Utterance

type CurationReason = Literal["conflicting_label_duplicate", "exact_repeat_same_label"]
CURATION_REASONS: frozenset[str] = frozenset(get_args(CurationReason.__value__))
_BUILTIN_LEDGER_PATH = Path(__file__).with_name("curation_ledger_english_v1.json")


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
    revisions: dict[str, str] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible record."""
        return {
            "ledger_id": self.ledger_id,
            "version": self.version,
            "revisions": dict(sorted(self.revisions.items())),
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
    raw_revisions = payload.get("revisions", {})
    if not isinstance(raw_entries, list) or not isinstance(raw_revisions, dict):
        raise CurationLedgerError("Curation ledger needs an 'entries' list and 'revisions' map.")
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
        if len(digest) != 64 or digest != digest.lower():
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
    revisions = {str(key): str(value) for key, value in raw_revisions.items()}
    return CurationLedger(
        ledger_id=cast(str, _text(payload, "ledger_id")),
        version=cast(str, _text(payload, "version")),
        entries=tuple(entries),
        revisions=revisions,
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


def stamp_curated_corpora(
    utterances: Sequence[Utterance], ledger: CurationLedger | None = None
) -> list[Utterance]:
    """Fills a missing hash and revision on rows of corpora the ledger covers."""
    active = ledger if ledger is not None else load_builtin_curation_ledger()
    stamped: list[Utterance] = []
    for row in utterances:
        revision = active.revisions.get(row.corpus)
        if revision is None:
            stamped.append(row)
            continue
        if row.normalized_audio_sha256 is None:
            row = replace(row, normalized_audio_sha256=normalized_audio_sha256(row.audio_path))
        if row.dataset_revision is None:
            row = replace(row, dataset_revision=revision)
        stamped.append(row)
    return stamped


def apply_curation_ledger(
    utterances: Sequence[Utterance], ledger: CurationLedger
) -> CurationResult:
    """Drops ledger-listed rows after checking every entry against the data.

    Entries for a corpus revision that is not among the rows are not applicable and are
    skipped. Applicable entries must name an existing clip with a matching hash, and must
    resolve a real duplicate group; otherwise the ledger is stale and raises.
    """
    by_id = {row.sample_id: row for row in utterances}
    present = {(row.corpus, row.dataset_revision) for row in utterances}
    by_hash: defaultdict[str, list[Utterance]] = defaultdict(list)
    for row in utterances:
        if row.normalized_audio_sha256 is not None:
            by_hash[row.normalized_audio_sha256].append(row)

    excluded_ids: dict[str, CurationReason] = {}
    for entry in ledger.entries:
        if (entry.corpus, entry.dataset_revision) not in present:
            continue
        found = by_id.get(entry.sample_id)
        if found is None or found.dataset_revision != entry.dataset_revision:
            raise CurationLedgerError(
                f"Ledger entry {entry.sample_id!r} does not exist in {entry.corpus} "
                f"revision {entry.dataset_revision!r}."
            )
        if found.normalized_audio_sha256 != entry.normalized_audio_sha256:
            raise CurationLedgerError(
                f"Ledger hash mismatch for {entry.sample_id!r}: ledger "
                f"{entry.normalized_audio_sha256}, data {found.normalized_audio_sha256}."
            )
        group = by_hash[entry.normalized_audio_sha256]
        if len(group) < 2:
            raise CurationLedgerError(
                f"Ledger entry {entry.sample_id!r} is unused: no other clip shares its content."
            )
        if entry.reason == "conflicting_label_duplicate":
            if len({member.label for member in group}) < 2:
                raise CurationLedgerError(
                    f"Ledger entry {entry.sample_id!r} is not a conflicting duplicate: "
                    "all clips with this content share one label."
                )
        else:
            kept = by_id.get(cast(str, entry.kept_sample_id))
            if (
                kept is None
                or kept.normalized_audio_sha256 != entry.normalized_audio_sha256
                or kept.label != found.label
            ):
                raise CurationLedgerError(
                    f"Ledger entry {entry.sample_id!r} names kept clip "
                    f"{entry.kept_sample_id!r} that is missing, differs in content, or "
                    "carries another label."
                )
        if entry.sample_id in excluded_ids:
            raise CurationLedgerError(f"Ledger lists {entry.sample_id!r} more than once.")
        excluded_ids[entry.sample_id] = entry.reason

    kept_ids = {
        entry.kept_sample_id
        for entry in ledger.entries
        if entry.sample_id in excluded_ids and entry.kept_sample_id is not None
    }
    if kept_ids & excluded_ids.keys():
        raise CurationLedgerError("A clip named as kept is also excluded by the ledger.")
    return CurationResult(
        utterances=tuple(row for row in utterances if row.sample_id not in excluded_ids),
        excluded=tuple(
            CurationExclusion(sample_id=row.sample_id, reason=excluded_ids[row.sample_id])
            for row in sorted(utterances, key=lambda item: item.sample_id)
            if row.sample_id in excluded_ids
        ),
    )
