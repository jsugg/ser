import os
from typing import List
import warnings 

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from ser.utils.audio_utils import read_audio_file
from ser.config import DEFAULT_FEATURE_CONFIG, TMP_FOLDER

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


def extract_feature(file: str) -> np.ndarray:
    """
    Extract features (mfcc, chroma, mel, contrast, tonnetz) from an audio file.

    Parameters
    ----------
    file : str
        Path to the audio file.

    Returns
    -------
    np.ndarray
        Extracted features.
    """
    try:
        X, sample_rate = read_audio_file(file)
    except Exception as err:
        print(f"Error reading file {file}: {err}")
        raise err.with_traceback(err.__traceback__)

    X, sample_rate = read_audio_file(file)
    n_fft: int = min(len(X), 2048)
    stft: np.ndarray = np.abs(librosa.stft(X, n_fft=n_fft))
    result: np.ndarray = np.array([])

    try:
        if DEFAULT_FEATURE_CONFIG['mfcc']:
            mfccs: np.ndarray = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, mfccs))

        if DEFAULT_FEATURE_CONFIG['chroma']:
            chroma: np.ndarray = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, chroma))

        if DEFAULT_FEATURE_CONFIG['mel']:
            mel: np.ndarray = np.mean(librosa.feature.melspectrogram(
                y=X, sr=sample_rate, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, mel))

        if DEFAULT_FEATURE_CONFIG['contrast']:
            spectral_contrast: np.ndarray = np.mean(librosa.feature.spectral_contrast(
                S=librosa.power_to_db(stft), sr=sample_rate, n_fft=n_fft).T, axis=0)
            result = np.hstack((result, spectral_contrast))

        if DEFAULT_FEATURE_CONFIG['tonnetz']:
            y: np.ndarray = librosa.effects.harmonic(X)
            tonnetz: np.ndarray = np.mean(
                librosa.feature.tonnetz(y=y, sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    except Exception as err:
        print(f"Error extracting features from file {file}: {err}")
        raise err.with_traceback(err.__traceback__)

    return result


def extended_extract_feature(
        audiofile: str,
        frame_size: int = 3,
        frame_stride: int = 1
) -> List[np.ndarray]:
    """
    Extract features (mfcc, chroma, mel) from an audio file using 
    extended audio frames.

    Parameters
    ----------
    audiofile : str
        Path to the audio file.
    mfcc : bool
        Whether to include MFCC features.
    chroma : bool
        Whether to include chroma features.
    mel : bool
        Whether to include mel features.
    frame_size : int, optional
        Size of the frame in seconds, by default 3.
    frame_stride : int, optional
        Stride between frames in seconds, by default 1.

    Returns
    -------
    List[np.ndarray]
        List of extracted features.
    """
    temp_filename: str = f'{TMP_FOLDER}/temp.wav'
    features: List[np.ndarray] = []
    X, sample_rate = read_audio_file(audiofile)
    frame_length: int = int(frame_size * sample_rate)
    frame_step: int = int(frame_stride * sample_rate)
    num_frames: int = int(np.ceil(len(X) / frame_step))

    for frame in tqdm(range(num_frames)):
        start: int = frame * frame_step
        end: int = min(start + frame_length, len(X))
        frame_data: np.ndarray = X[start:end]

        sf.write(temp_filename, frame_data, sample_rate)
        feature: np.ndarray = extract_feature(temp_filename)
        features.append(feature)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return features
