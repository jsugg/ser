"""Canonical runtime phase names for SER workflow observability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

PHASE_WORKFLOW_TOTAL: Final[str] = "workflow_total"
PHASE_EMOTION_SETUP: Final[str] = "emotion_setup"
PHASE_EMOTION_INFERENCE: Final[str] = "emotion_inference"
PHASE_TRANSCRIPTION_MODEL_LOAD: Final[str] = "transcription_model_load"
PHASE_TRANSCRIPTION: Final[str] = "transcription"
PHASE_TIMELINE_BUILD: Final[str] = "timeline_build"
PHASE_TIMELINE_OUTPUT: Final[str] = "timeline_output"

PHASE_LABELS: Final[Mapping[str, str]] = {
    PHASE_WORKFLOW_TOTAL: "SER workflow",
    PHASE_EMOTION_SETUP: "Emotion setup",
    PHASE_EMOTION_INFERENCE: "Emotion inference",
    PHASE_TRANSCRIPTION_MODEL_LOAD: "Transcription model load",
    PHASE_TRANSCRIPTION: "Transcription",
    PHASE_TIMELINE_BUILD: "Timeline build",
    PHASE_TIMELINE_OUTPUT: "Timeline output",
}


def phase_label(phase_name: str) -> str:
    """Returns a human-readable label for one phase identifier."""
    label = PHASE_LABELS.get(phase_name)
    if label is not None:
        return label
    fallback = phase_name.strip().replace("_", " ")
    if not fallback:
        return "Workflow step"
    return fallback[0].upper() + fallback[1:]
