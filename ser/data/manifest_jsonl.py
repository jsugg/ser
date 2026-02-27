"""JSONL loader for SER training manifests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ser.data.manifest import Utterance
from ser.data.ontology import LabelOntology


def load_manifest_jsonl(
    path: Path,
    *,
    ontology: LabelOntology,
    base_dir: Path | None = None,
) -> list[Utterance]:
    """Loads a JSONL manifest into validated utterance records."""
    resolved_base = base_dir if base_dir is not None else path.parent
    utterances: list[Utterance] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON in manifest {path} at line {line_number}: {err}"
                ) from err
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Manifest {path} line {line_number} must be a JSON object."
                )
            utterance = Utterance.from_record(
                cast(dict[str, object], payload),
                base_dir=resolved_base,
                ontology=ontology,
            )
            if utterance.sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate sample_id {utterance.sample_id!r} in manifest {path}."
                )
            seen_ids.add(utterance.sample_id)
            utterances.append(utterance)
    return utterances


def write_manifest_jsonl(
    path: Path,
    utterances: Sequence[Utterance],
    *,
    base_dir: Path | None = None,
) -> None:
    """Writes utterances to a deterministic JSONL manifest."""
    resolved_base = base_dir if base_dir is not None else path.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for utterance in utterances:
            record = utterance.to_record(base_dir=resolved_base)
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
