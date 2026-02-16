"""Public utility helpers with lazy imports for heavy optional dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .common_utils import display_elapsed_time
from .logger import get_logger

if TYPE_CHECKING:
    import numpy as np

    from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord


def read_audio_file(file_path: str) -> tuple[np.ndarray, int]:
    """Reads and normalizes an audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        A tuple of `(audio_samples, sample_rate)`.
    """
    from .audio_utils import read_audio_file as _read_audio_file

    return _read_audio_file(file_path)


def build_timeline(
    text_with_timestamps: list[TranscriptWord],
    emotion_with_timestamps: list[EmotionSegment],
) -> list[TimelineEntry]:
    """Merges transcript and emotion segments into timeline rows."""
    from .timeline_utils import build_timeline as _build_timeline

    return _build_timeline(text_with_timestamps, emotion_with_timestamps)


def print_timeline(timeline: list[TimelineEntry]) -> None:
    """Prints timeline rows in a terminal-friendly format."""
    from .timeline_utils import print_timeline as _print_timeline

    _print_timeline(timeline)


def save_timeline_to_csv(timeline: list[TimelineEntry], file_name: str) -> str:
    """Writes timeline rows to CSV and returns the output path."""
    from .timeline_utils import save_timeline_to_csv as _save_timeline_to_csv

    return _save_timeline_to_csv(timeline, file_name)


__all__ = [
    "get_logger",
    "read_audio_file",
    "display_elapsed_time",
    "build_timeline",
    "print_timeline",
    "save_timeline_to_csv",
]
