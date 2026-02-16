"""Command-line entry point for the speech emotion recognition tool."""

import argparse
import logging
import sys
import time

from ser.config import Config
from ser.models.emotion_model import predict_emotions, train_model
from ser.transcript import extract_transcript
from ser.utils.logger import get_logger
from ser.utils.timeline_utils import (
    build_timeline,
    print_timeline,
    save_timeline_to_csv,
)

logger: logging.Logger = get_logger("ser")


def main() -> None:
    """Parses CLI arguments and runs training or inference workflows."""
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
    args: argparse.Namespace = parser.parse_args()

    if args.train:
        logger.info("Starting model training...")
        start_time: float = time.time()
        train_model()
        logger.info(msg=f"Training completed in {time.time() - start_time:.2f} seconds")
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
    timeline: list = build_timeline(transcript, emotions)
    print_timeline(timeline)

    if args.save_transcript:
        csv_file_name: str = save_timeline_to_csv(timeline, args.file)
        logger.info(msg=f"Timeline saved to {csv_file_name}")

    logger.info(
        msg=f"Emotion prediction completed in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
