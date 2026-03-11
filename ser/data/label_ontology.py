"""Public label-ontology resolution helpers shared by data workflows."""

from __future__ import annotations

import os
from typing import cast

from ser.config import AppConfig
from ser.data.ontology import LabelOntology, UnknownLabelPolicy, normalize_label


def _read_unknown_label_policy_env() -> UnknownLabelPolicy:
    """Reads unknown-label policy from environment with strict fallback."""
    raw = os.getenv("SER_UNKNOWN_LABEL_POLICY", "drop").strip().lower()
    if raw in {"drop", "error", "map_to_other"}:
        return cast(UnknownLabelPolicy, raw)
    return "drop"


def resolve_label_ontology(settings: AppConfig) -> LabelOntology:
    """Resolves the active label ontology from settings and environment overrides."""
    ontology_id = os.getenv("SER_LABEL_ONTOLOGY_ID", "default_v1").strip() or "default_v1"
    allowed_labels_raw = os.getenv("SER_ALLOWED_LABELS", "").strip()
    if allowed_labels_raw:
        allowed = {
            normalize_label(item) for item in allowed_labels_raw.split(",") if normalize_label(item)
        }
    else:
        allowed = {normalize_label(label) for label in settings.emotions.values()}
    if not allowed:
        raise RuntimeError(
            "Resolved SER label ontology contains zero allowed labels. "
            "Check SER_ALLOWED_LABELS / configured emotion mapping."
        )
    other_label = os.getenv("SER_OTHER_LABEL", "other").strip() or "other"
    return LabelOntology(
        ontology_id=ontology_id,
        allowed_labels=frozenset(allowed),
        unknown_label_policy=_read_unknown_label_policy_env(),
        other_label=normalize_label(other_label),
    )
