"""Canonical label ontology utilities for multi-corpus SER training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

type UnknownLabelPolicy = Literal["drop", "error", "map_to_other"]


@dataclass(frozen=True)
class LabelOntology:
    """Defines the canonical label space and unknown-label behavior."""

    ontology_id: str
    allowed_labels: frozenset[str]
    unknown_label_policy: UnknownLabelPolicy = "drop"
    other_label: str = "other"


def normalize_label(label: str) -> str:
    """Normalizes label strings for stable comparisons."""
    return label.strip().lower()


def ensure_label_allowed(*, label: str, ontology: LabelOntology) -> None:
    """Raises when a label is not part of the canonical ontology."""
    if label not in ontology.allowed_labels:
        raise ValueError(
            f"Label {label!r} is not part of ontology {ontology.ontology_id!r}."
        )


def remap_label(
    *,
    raw_label: str,
    mapping: Mapping[str, str] | None,
    ontology: LabelOntology,
) -> str | None:
    """Remaps a raw dataset label into the canonical ontology.

    Args:
        raw_label: Raw label string from a dataset.
        mapping: Optional mapping of raw labels to canonical labels.
        ontology: Canonical ontology definition.

    Returns:
        Canonical label when allowed by the ontology or ``None`` when dropped.

    Raises:
        ValueError: When the ontology policy is ``error`` and the label is unknown.
    """

    raw = raw_label.strip()
    mapped = mapping.get(raw, "") if mapping is not None else raw
    canonical = normalize_label(mapped) if mapped else ""
    if canonical and canonical in ontology.allowed_labels:
        return canonical

    policy = ontology.unknown_label_policy
    if policy == "drop":
        return None
    if policy == "map_to_other":
        other = normalize_label(ontology.other_label)
        ensure_label_allowed(label=other, ontology=ontology)
        return other
    raise ValueError(
        f"Unknown label {raw_label!r} under ontology {ontology.ontology_id!r}."
    )
