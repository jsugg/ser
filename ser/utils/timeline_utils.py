"""Utilities to merge, render, and persist transcript-emotion timelines."""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from types import TracebackType
from typing import Protocol

from ser.config import get_settings
from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord
from ser.utils.common_utils import display_elapsed_time
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class HaloContext(Protocol):
    """Runtime protocol for the context manager returned by `halo.Halo`."""

    def __enter__(self) -> object: ...

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool | None: ...


class HaloFactory(Protocol):
    """Callable protocol for constructing spinner context managers."""

    def __call__(self, *args: object, **kwargs: object) -> HaloContext: ...


class ColorFunction(Protocol):
    """Callable protocol for color formatter helpers."""

    def __call__(self, name: str) -> str: ...


class AttrFunction(Protocol):
    """Callable protocol for terminal attribute formatter helpers."""

    def __call__(self, name: str) -> str: ...


halo_factory: HaloFactory | None
halo_factory = None
try:
    from halo import Halo as _Halo
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI envs.
    pass
else:
    halo_factory = _Halo

attr_fn: AttrFunction | None
bg_fn: ColorFunction | None
fg_fn: ColorFunction | None
attr_fn = bg_fn = fg_fn = None
try:
    from colored import attr as _attr
    from colored import bg as _bg
    from colored import fg as _fg
except ModuleNotFoundError:  # pragma: no cover - exercised in lightweight CI envs.
    pass
else:
    attr_fn = _attr
    bg_fn = _bg
    fg_fn = _fg


def save_timeline_to_csv(timeline: list[TimelineEntry], file_name: str) -> str:
    """Saves timeline rows as CSV under the configured transcript folder.

    Args:
        timeline: Sequence of timeline row objects.
        file_name: Source audio path used to derive the output CSV name.

    Returns:
        The generated CSV path.
    """
    if halo_factory is None:
        raise RuntimeError(
            "Missing timeline output dependency 'halo'. Install project dependencies."
        )

    settings = get_settings()
    logger.info(msg="Starting to save timeline to CSV.")
    output_folder: Path = settings.timeline.folder
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_folder / f"{Path(file_name).stem}.csv"

    with halo_factory(
        text=f"Saving transcript to {output_path}",
        spinner="dots",
        text_color="green",
    ):
        with open(output_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Emotion", "Speech"])
            logger.debug("Header written to CSV file.")

            for entry in timeline:
                rounded_time = round(float(entry.timestamp_seconds), 2)
                writer.writerow([rounded_time, entry.emotion, entry.speech])
                logger.debug(
                    msg=f"Written row: {[rounded_time, entry.emotion, entry.speech]}"
                )

    logger.info(msg=f"Timeline successfully saved to {output_path}")
    return str(output_path)


def _to_milliseconds(seconds: float) -> int:
    """Converts seconds to integer milliseconds for stable timeline joins."""
    return int(round(seconds * 1000))


def _emotion_at_timestamp(
    timestamp_ms: int, emotion_segments: list[tuple[str, int, int]]
) -> str:
    """Returns the active emotion for a timestamp based on segment intervals."""
    for emotion, start_ms, end_ms in emotion_segments:
        if start_ms <= timestamp_ms < end_ms:
            return emotion

    if emotion_segments and timestamp_ms == emotion_segments[-1][2]:
        return emotion_segments[-1][0]
    return ""


def build_timeline(
    text_with_timestamps: list[TranscriptWord],
    emotion_with_timestamps: list[EmotionSegment],
) -> list[TimelineEntry]:
    """Merges transcript and emotion timestamp streams into a single timeline.

    Args:
        text_with_timestamps: Transcript words with start/end timing.
        emotion_with_timestamps: Emotion segments with start/end timing.

    Returns:
        Timeline rows keyed on observed starts.
    """
    logger.info("Building timeline from text and emotion data.")
    if not text_with_timestamps and not emotion_with_timestamps:
        logger.info("No transcript or emotion timestamps provided.")
        return []

    words_by_timestamp: dict[int, list[str]] = defaultdict(list)
    for word in sorted(text_with_timestamps, key=lambda item: item.start_seconds):
        words_by_timestamp[_to_milliseconds(float(word.start_seconds))].append(
            word.word.strip()
        )

    emotion_segments: list[tuple[str, int, int]] = []
    for emotion in sorted(emotion_with_timestamps, key=lambda item: item.start_seconds):
        start_ms: int = _to_milliseconds(float(emotion.start_seconds))
        end_ms: int = _to_milliseconds(float(emotion.end_seconds))

        if end_ms < start_ms:
            logger.warning(
                "Skipping invalid emotion segment with end < start: %s (%s -> %s)",
                emotion.emotion,
                emotion.start_seconds,
                emotion.end_seconds,
            )
            continue
        if end_ms == start_ms:
            end_ms += 1

        emotion_segments.append((emotion.emotion, start_ms, end_ms))

    all_timestamps: list[int] = sorted(
        set(words_by_timestamp.keys())
        | {start_ms for _, start_ms, _ in emotion_segments}
    )

    logger.debug(msg=f"All timestamps: {all_timestamps}")
    logger.debug(msg=f"Text with timestamps: {text_with_timestamps}")
    logger.debug(msg=f"Emotion with timestamps: {emotion_with_timestamps}")

    timeline: list[TimelineEntry] = []
    for timestamp_ms in all_timestamps:
        text: str = " ".join(words_by_timestamp.get(timestamp_ms, [])).strip()
        active_emotion: str = _emotion_at_timestamp(timestamp_ms, emotion_segments)
        timeline.append(
            TimelineEntry(
                timestamp_seconds=timestamp_ms / 1000.0,
                emotion=active_emotion,
                speech=text,
            )
        )

    logger.info("Timeline built with %s entries.", len(timeline))
    return timeline


def color_txt(string: str, fg_color: str, bg_color: str, padding: int = 0) -> str:
    """Applies foreground/background ANSI colors to terminal text.

    Args:
        string: Text to colorize.
        fg_color: Foreground color name.
        bg_color: Background color name.
        padding: Optional right-padding width for alignment.

    Returns:
        ANSI-formatted text.
    """
    if attr_fn is None or bg_fn is None or fg_fn is None:
        raise RuntimeError(
            "Missing terminal color dependency 'colored'. Install project dependencies."
        )

    if padding:
        string = string.ljust(padding)

    return f"{fg_fn(fg_color)}{bg_fn(bg_color)}{string}{attr_fn('reset')}"


def print_timeline(timeline: list[TimelineEntry]) -> None:
    """Prints the timeline in a colorized tabular terminal format.

    Args:
        timeline: Sequence of timeline row objects.
    """
    logger.info(msg=f"Printing timeline with {len(timeline)} entries.")
    if not timeline:
        print("No timeline data available.")
        logger.info(msg="No timeline entries to print.")
        return

    max_time_width: int = max(
        len("Time"),
        *(
            len(display_elapsed_time(float(entry.timestamp_seconds), _format="short"))
            for entry in timeline
        ),
    )
    max_emotion_width: int = max(
        len("Emotion"), *(len(entry.emotion.capitalize()) for entry in timeline)
    )
    max_text_width: int = max(
        len("Speech"), *(len(entry.speech.strip()) for entry in timeline)
    )

    print(color_txt("Time", "black", "green", max_time_width), end="")
    print(color_txt("Emotion", "black", "yellow", max_emotion_width), end="")
    print(color_txt("Speech", "black", "blue", max_text_width))

    for entry in timeline:
        time_str: str = display_elapsed_time(
            float(entry.timestamp_seconds), _format="short"
        ).ljust(max_time_width)
        emotion_str: str = f"{entry.emotion.capitalize()}".ljust(max_emotion_width)
        text_str: str = f"{entry.speech.strip()}".ljust(max_text_width)

        print(f"{time_str} {emotion_str} {text_str}")
