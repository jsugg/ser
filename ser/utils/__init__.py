"""Public utility helpers with lazy imports for heavy optional dependencies."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from .common_utils import display_elapsed_time
from .logger import get_logger

if TYPE_CHECKING:
    import numpy as np

    from ser.config import AppConfig
    from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord


def _resolve_boundary_settings() -> AppConfig:
    """Returns a fresh settings snapshot for public utility wrappers."""
    from ser.config import reload_settings

    return reload_settings()


def read_audio_file(
    file_path: str,
    *,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Reads and normalizes an audio file."""
    reader = cast(
        Callable[..., tuple[np.ndarray, int]],
        importlib.import_module("ser._internal.utils.audio_utils").read_audio_file,
    )
    return reader(
        file_path,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        audio_read_config=_resolve_boundary_settings().audio_read,
    )


def build_timeline(
    text_with_timestamps: list[TranscriptWord],
    emotion_with_timestamps: list[EmotionSegment],
) -> list[TimelineEntry]:
    """Merges transcript and emotion segments into timeline rows."""
    builder = cast(
        Callable[[list[TranscriptWord], list[EmotionSegment]], list[TimelineEntry]],
        importlib.import_module("ser._internal.utils.timeline_utils").build_timeline,
    )
    return builder(text_with_timestamps, emotion_with_timestamps)


def print_timeline(timeline: list[TimelineEntry]) -> None:
    """Prints timeline rows in a terminal-friendly format."""
    printer = importlib.import_module("ser._internal.utils.timeline_utils").print_timeline
    printer(timeline)


def save_timeline_to_csv(timeline: list[TimelineEntry], file_name: str) -> str:
    """Writes timeline rows to CSV and returns the output path."""
    writer = cast(
        Callable[..., str],
        importlib.import_module("ser._internal.utils.timeline_utils").save_timeline_to_csv,
    )
    return writer(
        timeline,
        file_name,
        timeline_config=_resolve_boundary_settings().timeline,
    )


__all__ = [
    "get_logger",
    "read_audio_file",
    "display_elapsed_time",
    "build_timeline",
    "print_timeline",
    "save_timeline_to_csv",
]
