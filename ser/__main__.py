"""
Speech Emotion Recognition (SER) Tool

This module serves as the entry point for the Speech Emotion Recognition (SER)
tool. It provides command-line interface (CLI) options for training the 
emotion classification model or predicting emotions and generating transcripts
from audio files.

Usage:
    The tool can be operated in two primary modes:
    1. Training mode: Trains the model using labeled audio data.
    2. Prediction mode: Predicts emotions in a given audio file 
        and extracts the transcript.

Author: Juan Sugg (juanpedrosugg@gmail.com)
Version: 1.0
License: MIT
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from ser.models.emotion_model import predict_emotions, train_model
from ser.transcript import extract_transcript
from ser.utils import (
    get_logger,
    build_timeline,
    print_timeline,
    save_timeline_to_csv,
)
from ser.utils.subtitles import SubtitleGenerator, FORMATTERS, timeline_to_subtitles
from ser.config import Config


logger: logging.Logger = get_logger("ser")


def main() -> None:
    """
    Main function to handle the command line interface logic.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Speech Emotion Recognition Tool"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the emotion classification model",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the audio file for emotion prediction",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=Config.DEFAULT_LANGUAGE,
        help="Language of the audio file",
    )
    parser.add_argument(
        "--save_transcript",
        action="store_true",
        help="Save the transcript to a CSV file",
    )
    parser.add_argument(
        "--subtitle-format",
        choices=tuple(FORMATTERS.keys()),
        help=(
            "Export the generated timeline as subtitles in the chosen format. "
            "If omitted, the format is inferred from --subtitle-output when possible."
        ),
    )
    parser.add_argument(
        "--subtitle-output",
        type=str,
        help=(
            "File path for the exported subtitle file. The format is inferred from "
            "the extension when --subtitle-format is not provided."
        ),
    )
    args: argparse.Namespace = parser.parse_args()

    if args.train:
        logger.info("Starting model training...")
        start_time: float = time.time()
        train_model()
        logger.info(
            msg=f"Training completed in {time.time() - start_time:.2f} seconds"
        )
        sys.exit(0)

    if not args.file:
        logger.error(msg="No audio file provided for prediction.")
        sys.exit(1)

    logger.info(msg="Starting emotion prediction...")
    start_time = time.time()
    emotions: list[tuple[str, float, float]] = predict_emotions(args.file)
    transcript: list[tuple[str, float, float]] = extract_transcript(
        args.file, args.language
    )
    timeline: list[tuple[float, str, str]] = build_timeline(transcript, emotions)
    print_timeline(timeline)

    if args.subtitle_format or args.subtitle_output:
        if not args.subtitle_output:
            logger.error(
                msg="--subtitle-output is required to export subtitles.",
            )
            sys.exit(1)

        subtitle_format: str | None = args.subtitle_format
        if not subtitle_format:
            subtitle_format = _infer_subtitle_format(args.subtitle_output)
            if not subtitle_format:
                logger.error(
                    "Unable to infer subtitle format from %s. Provide --subtitle-format.",
                    args.subtitle_output,
                )
                sys.exit(1)
        else:
            inferred_format: str | None = _infer_subtitle_format(args.subtitle_output)
            if inferred_format and inferred_format != subtitle_format:
                logger.info(
                    "Using subtitle format %s (overriding inferred format %s from output path)",
                    subtitle_format,
                    inferred_format,
                )

        subtitles: list[tuple[float, float, str, str]] = timeline_to_subtitles(timeline)
        if not subtitles:
            logger.warning("Timeline did not produce any subtitle entries to export.")
        else:
            try:
                generator = SubtitleGenerator(FORMATTERS[subtitle_format])
                generator.generate_file(subtitles, args.subtitle_output)
                logger.info(
                    "Subtitle file exported to %s",
                    args.subtitle_output,
                )
            except Exception as err:
                logger.error(
                    msg=f"Failed to export subtitles: {err}",
                    exc_info=True,
                )
                sys.exit(1)

    if args.save_transcript:
        csv_file_name: str = save_timeline_to_csv(timeline, args.file)
        logger.info(msg=f"Timeline saved to {csv_file_name}")

    logger.info(
        msg=f"Emotion prediction completed in {time.time() - start_time:.2f} seconds"
    )


def _infer_subtitle_format(output_path: str) -> str | None:
    suffix: str = Path(output_path).suffix.lower().lstrip(".")
    return suffix if suffix in FORMATTERS else None


if __name__ == "__main__":
    main()
