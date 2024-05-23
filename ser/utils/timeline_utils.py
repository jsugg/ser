"""
Timeline Utility Functions for Speech Emotion Recognition (SER) Tool

This module provides functions to build, print, and save timelines that integrate recognized
emotions with corresponding transcripts. It includes functions to build the timeline, print it,
and save it to a CSV file.

Functions:
    - save_timeline_to_csv: Saves the timeline to a CSV file.
    - display_elapsed_time: Displays elapsed time in a formatted string.
    - build_timeline: Builds a timeline from text and emotion data.
    - print_timeline: Prints the ASCII timeline vertically.
    - color_txt: Colorizes a string.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import csv
import logging
from typing import List, Tuple

from colored import attr, bg, fg
from halo import Halo

from ser.utils import get_logger
from ser.config import Config


logger: logging.Logger = get_logger(__name__)


def save_timeline_to_csv(timeline: List[tuple], file_name: str) -> str:
    """
    Saves the timeline to a CSV file.

    Arguments:
        timeline (List[tuple]): The timeline data to be saved.
        file_name (str): The name of the file to save the timeline to.
    

    Returns:
        str: The path to the saved CSV file.
    """
    logger.info(msg="Starting to save timeline to CSV.")
    file_name = file_name.split("/")[-1]
    file_name = ".".join(
        [
            "/".join(
                [Config.TIMELINE_CONFIG["folder"], file_name.split(".")[0]]
            ),
            "csv",
        ]
    )

    with Halo(
        text=f"Saving transcript to {file_name}",
        spinner="dots",
        text_color="green",
    ):
        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(["Time (s)", "Emotion", "Speech"])
            logger.debug("Header written to CSV file.")

            # Write the data
            for time, emotion, speech in timeline:
                time: float = round(float(time), 2)
                writer.writerow([time, emotion, speech])
                logger.debug(msg=f"Written row: {[time, emotion, speech]}")

    logger.info(msg=f"Timeline successfully saved to {file_name}")
    return file_name


def display_elapsed_time(elapsed_time: float, _format: str = "long") -> str:
    """
    Returns the elapsed time in seconds in long or short format.

    Arguments:
        elapsed_time (Union[int, float]): Elapsed time in seconds.
        format (str, optional): Format of the elapsed time 
            ('long' or 'short'), by default 'long'.

    Returns:
        str: Formatted elapsed time.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == "long":
        return (
            f"{minutes} min {seconds} seconds"
            if minutes
            else f"{elapsed_time} seconds"
        )
    return f"{minutes}m{seconds}s" if minutes else f"{elapsed_time:.2f}s"


def build_timeline(
    text_with_timestamps, emotion_with_timestamps
) -> List[Tuple[float, str, str]]:
    """
    Builds a timeline from text and emotion data.

    Arguments:
        text_with_timestamps (List[tuple]): Transcript data with timestamps.
        emotion_with_timestamps (List[tuple]): Emotion data with timestamps.

    Returns:
        List[Tuple[float, str, str]]: Combined timeline with timestamps, 
            emotions, and text.
    """
    logger.info("Building timeline from text and emotion data.")
    timeline: List[Tuple[float, str, str]] = []
    all_timestamps: List[float] = sorted(
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
    emotion_dict: dict = {
        t: emotion for emotion, t, _ in emotion_with_timestamps
    }
    
    logger.debug(msg=f"Text dict: {text_dict}")
    logger.debug(msg=f"Emotion dict: {emotion_dict}")

    for timestamp in all_timestamps:
        text: str = text_dict.get(timestamp, "")
        emotion: str = emotion_dict.get(timestamp, "")
        timeline.append((timestamp, emotion, text))

    logger.info(msg=f"Timeline built with {len(timeline)} entries.")
    return timeline


def color_txt(
    string: str, fg_color: str, bg_color: str, padding: int = 0
) -> str:
    """
    Colorizes a string.

    Arguments:
        string (str): String to be colorized.
        fg_color (str): Foreground color.
        bg_color (str): Background color.

    Returns:
        str: Colorized string.
    """
    if padding:
        string = string.ljust(padding)

    return f"{fg(fg_color)}{bg(bg_color)}{string}{attr('reset')}"


def print_timeline(timeline: List[tuple]) -> None:
    """
    Prints the ASCII timeline vertically.

    Arguments:
        timeline (List[Tuple[Union[int, float], str, str]]): ASCII timeline.
    """
    # Calculate maximum width for each column
    logger.info(msg=f"Printing timeline with {len(timeline)} entries.")
    max_time_width: int = max(
        len(display_elapsed_time(float(ts), _format="short"))
        for ts, _, _ in timeline
    )
    max_emotion_width: int = max(len(em.capitalize()) for _, em, _ in timeline)
    max_text_width: int = max(len(txt.strip()) for _, _, txt in timeline)

    # Header
    print(color_txt("Time", "black", "green", max_time_width), end="")
    print(color_txt("Emotion", "black", "yellow", max_time_width), end="")
    print(color_txt("Speech", "black", "blue", max_time_width))

    # Print each entry vertically
    for ts, em, txt in timeline:
        time_str: str = (
            f"{display_elapsed_time(float(ts), _format='short')}".ljust(
                max_time_width
            )
        )
        emotion_str: str = f"{em.capitalize()}".ljust(max_emotion_width)
        text_str: str = f"{txt.strip()}".ljust(max_text_width)

        print(f"{time_str} {emotion_str} {text_str}")
