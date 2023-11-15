#!/usr/bin/env python3
"""
Speech Emotion Recognition (SER) System

This module serves as the entry point for SER, an application designed 
to identify and analyze emotions from spoken language. The system leverages 
advanced machine learning techniques and audio processing algorithms to 
classify emotions in speech, providing insights into the emotional state
conveyed in audio recordings.

The SER system is capable of handling the following key functionalities:
- Training the emotion classification model on a dataset of audio files.
- Predicting emotions from provided audio files.
- Extracting a transcript of spoken words in the audio file.
- Building a comprehensive timeline that integrates recognized emotions 
  with the corresponding transcript.
- Offering command-line interface (CLI) options for user interaction, 
  such as specifying the audio file, selecting language, and choosing 
  specific features for extraction 
  and analysis.

Usage:
    The system can be operated in two primary modes:
    1. Training mode: Trains the model using labeled audio data.
    2. Prediction mode: Predicts emotions in a given audio file and extracts the transcript.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import argparse
import sys
import time
import warnings

from ser.transcript.transcript_extractor import extract_transcript
from ser.utils.common_utils import display_elapsed_time
from ser.utils.timeline_utils import build_timeline, print_timeline, save_timeline_to_csv
from ser.models.emotion_model import predict_emotions, train_model
from ser.config import DEFAULT_CONFIG

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

def main(arguments: argparse.Namespace) -> None:
    """
    Main function to predict emotions from an audio file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    start_time: float
    end_time: float

    if arguments.train:
        start_time = time.perf_counter()
        train_model()
        end_time = time.perf_counter()
        print(
            f'Execution time: {display_elapsed_time(end_time - start_time)}')
        sys.exit(0)

    if not arguments.file:
        print("Please provide an audio file path.")
        sys.exit(1)

    start_time = time.perf_counter()

    print(f'Predicting emotions from {arguments.file}')
    recognized_emotions = predict_emotions(arguments.file)
    text_with_timestamps = extract_transcript(arguments.file, arguments.language)
    timeline = build_timeline(text_with_timestamps, recognized_emotions)
    print_timeline(timeline)
    
    if arguments.save_transcript:
        csv_file_name: str = save_timeline_to_csv(timeline, arguments.file)
        print(f'Timeline saved to {csv_file_name}')

    end_time = time.perf_counter()
    print(f'Execution time: {display_elapsed_time(end_time - start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Speech Emotion Recognition from audio')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--file', type=str, help='Path to the audio file')
    parser.add_argument('--language', type=str,
        default=DEFAULT_CONFIG['language'],
        help='Language of the audio file. Defaults to English.')
    parser.add_argument('--save_transcript', action='store_true',
                        help='Save the transcript to a CSV file')

    args = parser.parse_args()

    main(args)
