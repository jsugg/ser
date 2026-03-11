"""RAVDESS reference parsing and sampling helpers for transcription profiling."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RavdessMetadata:
    """Parsed metadata fields from a RAVDESS filename."""

    emotion_code: str
    statement_code: str
    actor_id: str


def reference_text(file_path: Path, *, statement_text: Mapping[str, str]) -> str | None:
    """Return ground-truth transcript text from a RAVDESS filename."""

    parts = file_path.stem.split("-")
    if len(parts) < 5:
        return None
    return statement_text.get(parts[4])


def parse_metadata(file_path: Path) -> RavdessMetadata | None:
    """Extract actor, emotion, and statement metadata from a RAVDESS filename."""

    parts = file_path.stem.split("-")
    if len(parts) < 7:
        return None
    return RavdessMetadata(
        emotion_code=parts[2],
        statement_code=parts[4],
        actor_id=parts[6],
    )


def stratified_reference_subset(
    references: Sequence[Path],
    *,
    limit: int,
    random_seed: int,
) -> list[Path]:
    """Return a deterministic near-uniform subset across actor and statement strata."""

    if limit >= len(references):
        return list(references)

    strata: dict[tuple[str, str], list[Path]] = {}
    for file_path in references:
        metadata = parse_metadata(file_path)
        if metadata is None:
            continue
        key = (metadata.actor_id, metadata.statement_code)
        strata.setdefault(key, []).append(file_path)

    if not strata:
        return list(references[:limit])

    rng = random.Random(random_seed)
    keys = sorted(strata.keys())
    rng.shuffle(keys)
    for key in keys:
        strata[key] = sorted(strata[key])
        rng.shuffle(strata[key])

    selected: list[Path] = []
    consumed: dict[tuple[str, str], int] = {key: 0 for key in keys}

    while len(selected) < limit:
        progressed = False
        for key in keys:
            group = strata[key]
            index = consumed[key]
            if index >= len(group):
                continue
            selected.append(group[index])
            consumed[key] = index + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break

    return sorted(selected)


def summarize_subset_coverage(files: Sequence[Path]) -> dict[str, int]:
    """Summarize actor, emotion, and statement diversity in selected references."""

    actors: set[str] = set()
    emotions: set[str] = set()
    statements: set[str] = set()
    for file_path in files:
        metadata = parse_metadata(file_path)
        if metadata is None:
            continue
        actors.add(metadata.actor_id)
        emotions.add(metadata.emotion_code)
        statements.add(metadata.statement_code)
    return {
        "actors": len(actors),
        "emotions": len(emotions),
        "statements": len(statements),
    }


def collect_reference_files(
    *,
    glob_pattern: str,
    statement_text: Mapping[str, str],
    limit: int | None,
    sampling_strategy: str,
    random_seed: int,
    glob_paths: Callable[[str, bool], Sequence[str]],
) -> list[Path]:
    """Collect RAVDESS files with known reference transcripts."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")

    files = sorted(Path(raw_path) for raw_path in glob_paths(glob_pattern, True))
    references = [
        path for path in files if reference_text(path, statement_text=statement_text) is not None
    ]
    if limit is None:
        return references
    if sampling_strategy == "head":
        return references[:limit]
    if sampling_strategy == "stratified":
        return stratified_reference_subset(
            references,
            limit=limit,
            random_seed=random_seed,
        )
    raise ValueError("sampling_strategy must be one of: 'stratified', 'head'.")


__all__ = [
    "RavdessMetadata",
    "collect_reference_files",
    "parse_metadata",
    "reference_text",
    "stratified_reference_subset",
    "summarize_subset_coverage",
]
