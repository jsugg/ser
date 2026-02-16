"""Command-line entry point for the speech emotion recognition tool."""

import argparse
import logging
import sys
import time
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from ser.config import reload_settings
from ser.utils.logger import get_logger

if TYPE_CHECKING:
    from ser.domain import EmotionSegment, TimelineEntry, TranscriptWord

logger: logging.Logger = get_logger("ser")


def main() -> None:
    """Parses CLI arguments and runs training or inference workflows."""
    load_dotenv()
    settings = reload_settings()

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
        default=settings.default_language,
        help="Language of the audio file",
    )
    parser.add_argument(
        "--save_transcript",
        action="store_true",
        help="Save the transcript to a CSV file",
    )
    args: argparse.Namespace = parser.parse_args()

    if args.train:
        from ser.models.emotion_model import train_model

        logger.info("Starting model training...")
        start_time: float = time.time()
        try:
            train_model()
        except RuntimeError as err:
            logger.error("%s", err)
            sys.exit(2)
        except Exception as err:
            logger.error("Training workflow failed: %s", err, exc_info=True)
            sys.exit(1)
        logger.info(msg=f"Training completed in {time.time() - start_time:.2f} seconds")
        sys.exit(0)

    if not args.file:
        logger.error(msg="No audio file provided for prediction.")
        sys.exit(1)

    from ser.models.emotion_model import predict_emotions
    from ser.transcript import TranscriptionError, extract_transcript
    from ser.utils.timeline_utils import (
        build_timeline,
        print_timeline,
        save_timeline_to_csv,
    )

    logger.info(msg="Starting emotion prediction...")
    start_time = time.time()
    try:
        emotions: list[EmotionSegment] = predict_emotions(args.file)
        transcript: list[TranscriptWord] = extract_transcript(args.file, args.language)
        timeline: list[TimelineEntry] = build_timeline(transcript, emotions)
        print_timeline(timeline)

        if args.save_transcript:
            csv_file_name: str = save_timeline_to_csv(timeline, args.file)
            logger.info(msg=f"Timeline saved to {csv_file_name}")
    except TranscriptionError as err:
        logger.error("Transcription failed: %s", err, exc_info=True)
        sys.exit(3)
    except FileNotFoundError as err:
        logger.error("%s", err)
        sys.exit(2)
    except Exception as err:
        logger.error("Prediction workflow failed: %s", err, exc_info=True)
        sys.exit(1)

    logger.info(
        msg=f"Emotion prediction completed in {time.time() - start_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()
