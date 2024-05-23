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
import sys
import time
import logging
from typing import List, Tuple

from ser.models.emotion_model import predict_emotions, train_model
from ser.transcript import extract_transcript
from ser.utils import (
    get_logger,
    build_timeline,
    print_timeline,
    save_timeline_to_csv,
)
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
    emotions: List[Tuple[str, float, float]] = predict_emotions(args.file)
    transcript: List[Tuple[str, float, float]] = extract_transcript(
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
