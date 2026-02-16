"""Audio feature extraction utilities used by the SER model pipeline."""

import logging
import os
import tempfile
import warnings

import librosa
import numpy as np
import soundfile as sf
from halo import Halo

from ser.config import Config
from ser.utils.audio_utils import read_audio_file
from ser.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*is too large for input signal of length.*")


def extract_feature(file: str) -> np.ndarray:
    """Extracts the configured spectral features from one audio file.

    Args:
        file: Path to the audio file.

    Returns:
        A one-dimensional feature vector combining all enabled feature groups.
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
                librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft).T,
                axis=0,
            )
            result = np.hstack((result, mfccs))

        if Config.DEFAULT_FEATURE_CONFIG["chroma"]:
            chroma: np.ndarray = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=n_fft).T,
                axis=0,
            )
            result = np.hstack((result, chroma))

        if Config.DEFAULT_FEATURE_CONFIG["mel"]:
            mel: np.ndarray = np.mean(
                librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft).T,
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
) -> list[np.ndarray]:
    """Extracts frame-wise feature vectors from an audio file.

    Args:
        audiofile: Path to the audio file.
        frame_size: Duration of each frame, in seconds.
        frame_stride: Step between successive frames, in seconds.

    Returns:
        A list of feature vectors, one for each extracted frame.
    """
    os.makedirs(Config.TMP_FOLDER, exist_ok=True)
    temp_filename: str
    with tempfile.NamedTemporaryFile(
        suffix=".wav", dir=Config.TMP_FOLDER, delete=False
    ) as tmp_file:
        temp_filename = tmp_file.name
    features: list[np.ndarray] = []
    audio: np.ndarray
    sample_rate: float
    audio, sample_rate = read_audio_file(audiofile)
    frame_length: int = int(frame_size * sample_rate)
    frame_step: int = int(frame_stride * sample_rate)
    num_frames: int = int(np.ceil(len(audio) / frame_step))
    spinner = Halo(text="Processing", spinner="dots", text_color="green")

    spinner.start()
    try:
        for frame in range(num_frames):
            start: int = frame * frame_step
            end: int = min(start + frame_length, len(audio))
            frame_data: np.ndarray = audio[start:end]

            sf.write(temp_filename, frame_data, sample_rate)
            feature: np.ndarray = extract_feature(temp_filename)
            features.append(feature)
    finally:
        spinner.stop()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return features
