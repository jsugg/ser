"""
Feature Extraction for Speech Emotion Recognition (SER) Tool

This module provides functions to extract various audio features 
(e.g., MFCC, chroma, mel, spectral contrast, tonnetz) from audio
files. It includes both basic and extended feature extraction methods.

Functions:
    - extract_feature: Extracts features from an audio file.
    - extended_extract_feature: Extracts features from audio frames
        with extended audio frame handling.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

from typing import List
import os
import logging
import warnings

import numpy as np
import librosa
import soundfile as sf
from halo import Halo

from ser.utils import get_logger
from ser.config import Config
from ser.utils.audio_utils import read_audio_file


logger: logging.Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings(
    "ignore", message=".*is too large for input signal of length.*"
)


def extract_feature(file: str) -> np.ndarray:
    """
    Extract features (mfcc, chroma, mel, contrast, tonnetz) from an audio file.

    Arguments:
        file (str): Path to the audio file.

    Returns:
        np.ndarray: Extracted features.
    """
    audio: np.ndarray
    sample_rate: float
    try:
        audio, sample_rate = read_audio_file(file)
    except Exception as err:
        logger.error(msg=f"Error reading file {file}: {err}")
        raise err.with_traceback(err.__traceback__) from err

    n_fft: int = min(len(audio), 2048)
    stft: np.ndarray = np.abs(librosa.stft(audio, n_fft=n_fft))
    result: np.ndarray = np.array([])

    try:
        if Config.DEFAULT_FEATURE_CONFIG["mfcc"]:
            mfccs: np.ndarray = np.mean(
                librosa.feature.mfcc(
                    y=audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft
                ).T,
                axis=0,
            )
            result = np.hstack((result, mfccs))

        if Config.DEFAULT_FEATURE_CONFIG["chroma"]:
            chroma: np.ndarray = np.mean(
                librosa.feature.chroma_stft(
                    S=stft, sr=sample_rate, n_fft=n_fft
                ).T,
                axis=0,
            )
            result = np.hstack((result, chroma))

        if Config.DEFAULT_FEATURE_CONFIG["mel"]:
            mel: np.ndarray = np.mean(
                librosa.feature.melspectrogram(
                    y=audio, sr=sample_rate, n_fft=n_fft
                ).T,
                axis=0,
            )
            result = np.hstack((result, mel))

        if Config.DEFAULT_FEATURE_CONFIG["contrast"]:
            spectral_contrast: np.ndarray = np.mean(
                librosa.feature.spectral_contrast(
                    S=librosa.power_to_db(stft), sr=sample_rate, n_fft=n_fft
                ).T,
                axis=0,
            )
            result = np.hstack((result, spectral_contrast))

        if Config.DEFAULT_FEATURE_CONFIG["tonnetz"]:
            y: np.ndarray = librosa.effects.harmonic(audio)
            tonnetz: np.ndarray = np.mean(
                librosa.feature.tonnetz(y=y, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, tonnetz))
    except Exception as err:
        logger.error(msg=f"Error extracting features from file {file}: {err}")
        raise err.with_traceback(err.__traceback__) from err

    return result


def extended_extract_feature(
    audiofile: str, frame_size: int = 3, frame_stride: int = 1
) -> List[np.ndarray]:
    """
    Extract features (mfcc, chroma, mel) from an audio file using
    extended audio frames.

    Arguments:
        audiofile (str) Path to the audio file.
        mfcc (bool): Whether to include MFCC features.
        chroma (bool): Whether to include chroma features.
        mel (bool): Whether to include mel features.
        frame_size (int, optional): Size of the frame in seconds, by default 3.
        frame_stride (int, optional): Stride between frames in seconds, by default 1.

    Returns:
        List[np.ndarray]: List of extracted features.
    """
    temp_filename: str = f"{Config.TMP_FOLDER}/temp.wav"
    features: List[np.ndarray] = []
    audio: np.ndarray
    sample_rate: float
    audio, sample_rate = read_audio_file(audiofile)
    frame_length: int = int(frame_size * sample_rate)
    frame_step: int = int(frame_stride * sample_rate)
    num_frames: int = int(np.ceil(len(audio) / frame_step))
    spinner = Halo(text="Processing", spinner="dots", text_color="green")

    spinner.start()
    for frame in range(num_frames):
        start: int = frame * frame_step
        end: int = min(start + frame_length, len(audio))
        frame_data: np.ndarray = audio[start:end]

        sf.write(temp_filename, frame_data, sample_rate)
        feature: np.ndarray = extract_feature(temp_filename)
        features.append(feature)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    spinner.stop()

    return features
