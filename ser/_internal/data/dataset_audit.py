"""Content deduplication and leakage-safe dataset split ledgers."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from ser._internal.data.manifest import SplitName, Utterance
from ser._internal.data.recipe import (
    DatasetRecipe,
    RouteDisposition,
    RoutedUtterance,
    route_utterance,
)

type LedgerSplit = Literal["train", "dev", "test", "ssl_only", "quarantined"]


class DatasetAuditError(ValueError):
    """Raised when a recipe cannot produce a defensible benchmark."""


@dataclass(frozen=True)
class SplitLedgerEntry:
    """Immutable split assignment for one manifest row."""

    sample_id: str
    corpus: str
    split: LedgerSplit
    group_id: str | None
    normalized_audio_sha256: str | None
    tasks: tuple[str, ...]
    disposition: RouteDisposition
    reason: str

    def to_record(self) -> dict[str, object]:
        """Returns a deterministic JSON-compatible ledger record."""
        return {
            "sample_id": self.sample_id,
            "corpus": self.corpus,
            "split": self.split,
            "group_id": self.group_id,
            "normalized_audio_sha256": self.normalized_audio_sha256,
            "tasks": list(self.tasks),
            "disposition": self.disposition,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DatasetAuditReport:
    """Reproducible recipe, manifest, routing, and split audit output."""

    recipe_id: str
    recipe_revision: str
    recipe_digest: str
    manifest_digest: str
    split_ledger_digest: str
    seed: int
    counters: dict[str, int]
    ledger: tuple[SplitLedgerEntry, ...]


def _canonical_manifest_digest(utterances: list[Utterance]) -> str:
    records: list[dict[str, object]] = []
    for row in sorted(utterances, key=lambda item: item.sample_id):
        record = row.to_record()
        record.pop("audio_path", None)
        records.append(record)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ledger_digest(entries: list[SplitLedgerEntry]) -> str:
    payload = json.dumps(
        [entry.to_record() for entry in sorted(entries, key=lambda row: row.sample_id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_components(rows: list[RoutedUtterance]) -> dict[str, str | None]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for routed in rows:
        identities = [
            value
            for value in (routed.utterance.speaker_id, routed.utterance.session_id)
            if value is not None
        ]
        if identities:
            find(identities[0])
            for identity in identities[1:]:
                union(identities[0], identity)

    result: dict[str, str | None] = {}
    for routed in rows:
        identities = [
            value
            for value in (routed.utterance.speaker_id, routed.utterance.session_id)
            if value is not None
        ]
        result[routed.utterance.sample_id] = find(identities[0]) if identities else None
    return result


def _group_assignments(group_ids: set[str], *, corpus: str, seed: int) -> dict[str, SplitName]:
    ordered = sorted(
        group_ids,
        key=lambda group: hashlib.sha256(f"{seed}:{corpus}:{group}".encode()).digest(),
    )
    count = len(ordered)
    if count == 1:
        return {ordered[0]: "train"}
    if count == 2:
        return {ordered[0]: "train", ordered[1]: "test"}
    train_count = max(1, min(count - 2, round(count * 0.70)))
    remaining = count - train_count
    dev_count = max(1, min(remaining - 1, round(count * 0.15)))
    return {
        group: (
            "train" if index < train_count else "dev" if index < train_count + dev_count else "test"
        )
        for index, group in enumerate(ordered)
    }


def _validate_partition_isolation(entries: list[SplitLedgerEntry]) -> None:
    supervised = [entry for entry in entries if entry.split in {"train", "dev", "test"}]
    for attribute in ("group_id", "normalized_audio_sha256"):
        owners: dict[str, LedgerSplit] = {}
        for entry in supervised:
            value = getattr(entry, attribute)
            if value is None:
                continue
            previous = owners.setdefault(value, entry.split)
            if previous != entry.split:
                raise DatasetAuditError(
                    f"Split leakage: {attribute} {value!r} appears in {previous!r} and {entry.split!r}."
                )


def audit_dataset_recipe(
    utterances: list[Utterance],
    *,
    recipe: DatasetRecipe,
    seed: int = 17,
    strict: bool = True,
) -> DatasetAuditReport:
    """Audits all rows, deduplicates content, and builds per-corpus split assignments.

    Strict mode rejects missing revisions or hashes, duplicate content, leakage, and a
    primary task with fewer than two populated classes.
    """
    recipe.validate()
    sample_ids: set[str] = set()
    content_samples: defaultdict[str, list[str]] = defaultdict(list)
    routes: list[RoutedUtterance] = []
    duplicate_ids: set[str] = set()
    missing_hash_ids: set[str] = set()
    for utterance in utterances:
        if utterance.sample_id in sample_ids:
            raise DatasetAuditError(
                f"Duplicate sample_id {utterance.sample_id!r} across manifests."
            )
        sample_ids.add(utterance.sample_id)
        content_hash = utterance.normalized_audio_sha256
        if content_hash is None:
            missing_hash_ids.add(utterance.sample_id)
        else:
            content_samples[content_hash].append(utterance.sample_id)
        routes.append(route_utterance(utterance, recipe))

    for sample_group in content_samples.values():
        if len(sample_group) > 1:
            duplicate_ids.update(sample_group)

    if strict and duplicate_ids:
        raise DatasetAuditError(
            f"Duplicate normalized audio content detected for {len(duplicate_ids)} row(s)."
        )
    if strict and missing_hash_ids:
        raise DatasetAuditError(
            f"normalized_audio_sha256 is missing for {len(missing_hash_ids)} row(s)."
        )
    if strict:
        missing_revisions = [row.sample_id for row in utterances if row.dataset_revision is None]
        if missing_revisions:
            raise DatasetAuditError(
                f"dataset_revision is missing for {len(missing_revisions)} row(s)."
            )

    counters: Counter[str] = Counter(route.disposition for route in routes)
    entries: list[SplitLedgerEntry] = []
    by_corpus: dict[str, list[RoutedUtterance]] = defaultdict(list)
    for route in routes:
        by_corpus[route.utterance.corpus].append(route)

    for corpus, corpus_routes in sorted(by_corpus.items()):
        identities = _identity_components(corpus_routes)
        eligible = [
            route
            for route in corpus_routes
            if route.disposition not in {"dropped", "missing", "quarantined"}
            and route.utterance.sample_id not in duplicate_ids
        ]
        official = bool(eligible) and all(
            (route.utterance.native_split or route.utterance.split) is not None
            for route in eligible
        )
        group_ids = {
            identity
            for route in eligible
            if (identity := identities[route.utterance.sample_id]) is not None
        }
        assignments = (
            _group_assignments(group_ids, corpus=corpus, seed=seed) if not official else {}
        )

        for route in corpus_routes:
            utterance = route.utterance
            group_id = identities[utterance.sample_id]
            if utterance.sample_id in duplicate_ids:
                split: LedgerSplit = "quarantined"
                reason = "duplicate_normalized_audio"
                disposition: RouteDisposition = "quarantined"
                counters[route.disposition] -= 1
                counters["quarantined"] += 1
            elif route.disposition in {"dropped", "missing", "quarantined"}:
                split = "quarantined"
                reason = route.reason
                disposition = route.disposition
            elif official:
                native = utterance.native_split or utterance.split
                assert native is not None
                split = native
                reason = "verified_native_split"
                disposition = route.disposition
            elif group_id is None:
                split = "ssl_only"
                reason = "missing_speaker_or_session_group"
                disposition = route.disposition
            else:
                split = assignments[group_id]
                reason = "deterministic_grouped_split"
                disposition = route.disposition
            entries.append(
                SplitLedgerEntry(
                    sample_id=utterance.sample_id,
                    corpus=utterance.corpus,
                    split=split,
                    group_id=group_id,
                    normalized_audio_sha256=utterance.normalized_audio_sha256,
                    tasks=tuple(sorted(route.tasks)),
                    disposition=disposition,
                    reason=reason,
                )
            )

    if sum(counters.values()) != len(utterances):
        raise DatasetAuditError("Internal audit accounting did not classify every manifest row.")
    _validate_partition_isolation(entries)
    if strict:
        utterance_by_id = {utterance.sample_id: utterance for utterance in utterances}
        train_labels = {
            label
            for entry in entries
            if entry.split == "train" and "primary_emotion" in entry.tasks
            if (label := utterance_by_id[entry.sample_id].label) is not None
        }
        if len(train_labels) < 2:
            raise DatasetAuditError(
                "Primary emotion training partition must contain at least two populated classes."
            )
        evaluation_labels = {
            label
            for entry in entries
            if entry.split in {"dev", "test"} and "primary_emotion" in entry.tasks
            if (label := utterance_by_id[entry.sample_id].label) is not None
        }
        missing_train_labels = evaluation_labels - train_labels
        if missing_train_labels:
            raise DatasetAuditError(
                "Primary emotion evaluation classes are absent from train: "
                + ", ".join(sorted(missing_train_labels))
            )

    return DatasetAuditReport(
        recipe_id=recipe.recipe_id,
        recipe_revision=recipe.revision,
        recipe_digest=recipe.digest,
        manifest_digest=_canonical_manifest_digest(utterances),
        split_ledger_digest=_ledger_digest(entries),
        seed=seed,
        counters=dict(sorted(counters.items())),
        ledger=tuple(sorted(entries, key=lambda row: row.sample_id)),
    )
