"""Utilities to merge, render, and persist transcript-emotion timelines."""

import csv
import logging
import os

from colored import attr, bg, fg
from halo import Halo

from ser.config import Config
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


def save_timeline_to_csv(timeline: list[tuple], file_name: str) -> str:
    """Saves timeline rows as CSV under the configured transcript folder.

    Args:
        timeline: Sequence of `(timestamp, emotion, speech)` rows.
        file_name: Source audio path used to derive the output CSV name.

    Returns:
        The generated CSV path.
    """
    logger.info(msg="Starting to save timeline to CSV.")
    file_name = file_name.split("/")[-1]
    file_name = ".".join(
        [
            "/".join([Config.TIMELINE_CONFIG["folder"], file_name.split(".")[0]]),
            "csv",
        ]
    )
    os.makedirs(Config.TIMELINE_CONFIG["folder"], exist_ok=True)

    with Halo(
        text=f"Saving transcript to {file_name}",
        spinner="dots",
        text_color="green",
    ):
        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Time (s)", "Emotion", "Speech"])
            logger.debug("Header written to CSV file.")

            for timestamp, emotion, speech in timeline:
                rounded_time = round(float(timestamp), 2)
                writer.writerow([rounded_time, emotion, speech])
                logger.debug(msg=f"Written row: {[rounded_time, emotion, speech]}")

    logger.info(msg=f"Timeline successfully saved to {file_name}")
    return file_name


def display_elapsed_time(elapsed_time: float, _format: str = "long") -> str:
    """Formats elapsed seconds as either verbose or compact text.

    Args:
        elapsed_time: Elapsed time in seconds.
        _format: Output style, either `"long"` or `"short"`.

    Returns:
        Human-readable elapsed time text.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == "long":
        return (
            f"{minutes} min {seconds} seconds" if minutes else f"{elapsed_time} seconds"
        )
    return f"{minutes}m{seconds}s" if minutes else f"{elapsed_time:.2f}s"


def build_timeline(
    text_with_timestamps, emotion_with_timestamps
) -> list[tuple[float, str, str]]:
    """Merges transcript and emotion timestamp streams into a single timeline.

    Args:
        text_with_timestamps: Transcript tuples `(word, start, end)`.
        emotion_with_timestamps: Emotion tuples `(emotion, start, end)`.

    Returns:
        Timeline tuples `(timestamp, emotion, speech)` keyed on observed starts.
    """
    logger.info("Building timeline from text and emotion data.")
    timeline: list[tuple[float, str, str]] = []
    all_timestamps: list[float] = sorted(
        set(
            [t for _, t, _ in text_with_timestamps]
            + [t for _, t, _ in emotion_with_timestamps]
            + [t for _, _, t in emotion_with_timestamps]
        )
    )

    logger.debug(msg=f"All timestamps: {all_timestamps}")
    logger.debug(msg=f"Text with timestamps: {text_with_timestamps}")
    logger.debug(msg=f"Emotion with timestamps: {emotion_with_timestamps}")

    text_dict: dict = {t: text for text, t, _ in text_with_timestamps}
    emotion_dict: dict = {t: emotion for emotion, t, _ in emotion_with_timestamps}

    logger.debug(msg=f"Text dict: {text_dict}")
    logger.debug(msg=f"Emotion dict: {emotion_dict}")

    for timestamp in all_timestamps:
        text: str = text_dict.get(timestamp, "")
        emotion: str = emotion_dict.get(timestamp, "")
        timeline.append((timestamp, emotion, text))

    logger.info(msg=f"Timeline built with {len(timeline)} entries.")
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
    if padding:
        string = string.ljust(padding)

    return f"{fg(fg_color)}{bg(bg_color)}{string}{attr('reset')}"


def print_timeline(timeline: list[tuple]) -> None:
    """Prints the timeline in a colorized tabular terminal format.

    Args:
        timeline: Sequence of `(timestamp, emotion, speech)` rows.
    """
    logger.info(msg=f"Printing timeline with {len(timeline)} entries.")
    max_time_width: int = max(
        len(display_elapsed_time(float(ts), _format="short")) for ts, _, _ in timeline
    )
    max_emotion_width: int = max(len(em.capitalize()) for _, em, _ in timeline)
    max_text_width: int = max(len(txt.strip()) for _, _, txt in timeline)

    print(color_txt("Time", "black", "green", max_time_width), end="")
    print(color_txt("Emotion", "black", "yellow", max_time_width), end="")
    print(color_txt("Speech", "black", "blue", max_time_width))

    for ts, em, txt in timeline:
        time_str: str = f"{display_elapsed_time(float(ts), _format='short')}".ljust(
            max_time_width
        )
        emotion_str: str = f"{em.capitalize()}".ljust(max_emotion_width)
        text_str: str = f"{txt.strip()}".ljust(max_text_width)

        print(f"{time_str} {emotion_str} {text_str}")
