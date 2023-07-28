import argparse
import glob
import os
import pickle
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as soundfile
from colored import attr, bg, fg
from halo import Halo
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from whisper import Whisper
from whisper.utils import ResultWriter

import stable_whisper


# Emotions supported by the RAVDESS dataset
EMOTIONS: Dict[str, str] = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

TMP_FOLDER: str = "./tmp"
MODELS_CONFIG: Dict = {
    'models_folder': 'models/',
    'whisper:large-v2': {
        'name': 'large-v2',
        'path': 'OpenAI/whisper/'
    }
}

def read_audio_file(file_path: str, max_retries: int = 3, retry_delay: int = 1) -> Tuple[np.ndarray, int]:
    """
    Read an audio file.

    Parameters
    ----------
    file_path : str
        Path to the audio file to read.
    max_retries : int, optional
        Maximum number of retries to read the audio file.
    retry_delay : int, optional
        Delay in seconds between retries.

    Returns
    -------
    np.ndarray
        Audio data.
    int
        Sample rate.
    """
    for i in range(max_retries):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                audiofile: np.ndarray
                current_sample_rate: int
                audiofile, current_sample_rate = librosa.load(file_path, sr=None)
            audiofile = np.array(audiofile, dtype=np.float32)

            max_abs_value: np.float32 = np.max(np.abs(audiofile))
            if max_abs_value == 0:
                audiofile = np.zeros_like(audiofile)
            else:
                audiofile /= max_abs_value

            return audiofile, current_sample_rate

        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Falling back to soundfile...")
            try:
                with soundfile.SoundFile(file_path) as sf:
                    audiofile: np.ndarray = sf.read(dtype='float32')
                    current_sample_rate: int = sf.samplerate

                max_abs_value: np.float32 = np.max(np.abs(audiofile))
                if max_abs_value == 0:
                    audiofile = np.zeros_like(audiofile)
                else:
                    audiofile /= max_abs_value

                return audiofile, current_sample_rate

            except Exception as e:
                print(f"Error with soundfile: {e}")
                print(f"Retrying with librosa in {retry_delay} seconds...")
                time.sleep(retry_delay)


def extract_feature(file: str, mfcc: bool, chroma: bool, mel: bool) -> np.ndarray:
    """
    Extract features (mfcc, chroma, mel) from an audio file.

    Parameters
    ----------
    file : str
        Path to the audio file.
    mfcc : bool
        Whether to include MFCC features.
    chroma : bool
        Whether to include chroma features.
    mel : bool
        Whether to include mel features.

    Returns
    -------
    np.ndarray
        Extracted features.
    """
    X, sample_rate = read_audio_file(file)
    n_fft: int = min(len(X), 2048)
    stft: np.ndarray = np.abs(librosa.stft(X, n_fft=n_fft))
    result: np.ndarray = np.array([])

    if mfcc:
        mfccs: np.ndarray = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40, n_fft=n_fft).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma: np.ndarray = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=n_fft).T, axis=0)
        result = np.hstack((result, chroma))

    if mel:
        mel: np.ndarray = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=n_fft).T, axis=0)
        result = np.hstack((result, mel))

    return result

def extended_extract_feature(
        audiofile: str,
        language: str,
        mfcc: bool,
        chroma: bool,
        mel: bool,
        frame_size: int = 3,
        frame_stride: int = 1
) -> List[np.ndarray]:
    """
    Extract features (mfcc, chroma, mel) from an audio file using extended audio frames.

    Parameters
    ----------
    audiofile : str
        Path to the audio file.
    language : str
        Language of the audio file.
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

        soundfile.write(temp_filename, frame_data, samplerate=sample_rate)
        feature: np.ndarray = extract_feature(temp_filename, mfcc, chroma, mel)
        features.append(feature)

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return features

def load_data(
        test_size: float = 0.2,
        feature_extraction: str = 'regular',
        language: str = 'en'
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Loads the data and extracts features for each sound file.

    Parameters
    ----------
    test_size : float, optional
        Ratio of test data, by default 0.2.
    feature_extraction : str, optional
        Type of feature extraction ('regular' or 'extended'), by default 'regular'.
    language : str, optional
        Language of the audio files, by default 'en'.

    Returns
    -------
    Tuple[np.ndarray, List[str], np.ndarray, List[str]]
        Tuple containing train and test data and labels.
    """
    observed_emotions: List[str] = list(EMOTIONS.values())
    x: List[np.ndarray] = []
    y: List[str] = []

    for file in tqdm(glob.glob("./dataset/combined-datasets/Actor_*/*.wav")):
        file_name: str = os.path.basename(file)
        emotion: str = EMOTIONS[file_name.split("-")[2]]

        if emotion not in observed_emotions:
            continue

        if feature_extraction == 'extended':
            feature: List[np.ndarray] = extended_extract_feature(file, language, mfcc=True, chroma=True, mel=True)
        else:
            feature: np.ndarray = extract_feature(file, mfcc=True, chroma=True, mel=True)

        x.append(feature)
        y.append(emotion)

    print(f"{fg('green')}Feature data loaded successfully{attr('reset')}")

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

def train_model() -> None:
    """
    Train the emotion classification model.
    """
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: List[str]
    y_test: List[str]
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    model: MLPClassifier = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(x_train, y_train)
    y_pred: np.ndarray = model.predict(x_test)
    accuracy: float = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"{fg('green')}Accuracy: {accuracy * 100:.2f}%{attr('reset')}")
    pickle.dump(model, open('ser_model.pkl', 'wb'))

def load_model() -> Optional[MLPClassifier]:
    """
    Load the trained emotion classification model.

    Returns
    -------
    Optional[MLPClassifier]
        Trained model or None if the model is not found.
    """
    model_path: str = 'ser_model.pkl'
    if os.path.exists(model_path):
        model: Optional[MLPClassifier] = pickle.load(open(model_path, 'rb'))
        return model
    else:
        return None

def predict_audio_emotion(
        file: str,
        language: str
) -> List[Tuple[str, float, float]]:
    """
    Predict emotions from an audio file.

    Parameters
    ----------
    file : str
        Path to the audio file.
    language : str
        Language of the audio file.

    Returns
    -------
    List[Tuple[str, float, float]]
        List of predicted emotions with their start and end timestamps.
    """
    model: Optional[MLPClassifier] = load_model()

    if model is None:
        print(f"{fg('red')}Model not found. Please train the model first.{attr('reset')}")
        sys.exit(1)

    feature: List[np.ndarray] = extended_extract_feature(file, language, mfcc=True, chroma=True, mel=True)
    feature = np.array(feature).reshape(len(feature), -1)
    predicted_emotions: np.ndarray = model.predict(feature)

    X, _ = read_audio_file(file)
    audio_duration: float = librosa.get_duration(y=X)
    emotion_with_timestamps: List[Tuple[str, float, float]] = []
    prev_emotion: Optional[str] = None
    start_time: Optional[float] = None

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time: float = (timestamp - 1) * audio_duration / len(predicted_emotions)
                emotion_with_timestamps.append((prev_emotion, start_time, end_time))
            prev_emotion = emotion
            start_time = timestamp * audio_duration / len(predicted_emotions)

    if prev_emotion is not None:
        end_time = audio_duration
        emotion_with_timestamps.append((prev_emotion, start_time, end_time))

    return emotion_with_timestamps

def format_transcript(result: Dict[str, Union[str, List]]) -> List[Tuple[str, float, float]]:
    """
    Formats the transcript into a list of tuples containing the word, start time, and end time.

    Parameters
    ----------
    result : dict[str, str | list]
        The transcript result.

    Returns
    -------
    List[Tuple[str, float, float]]
        Formatted transcript with timestamps.
    """
    text_with_timestamps: List[Tuple[str, float, float]] = []
    words = result.all_words()

    for word in words:
        text_with_timestamps.append((word.word, word.start, word.end))

    return text_with_timestamps

def extract_transcript(filename: str, language: str) -> List[Tuple[str, float, float]]:
    """
    Extracts the transcript from the audio file.

    Parameters
    ----------
    filename : str
        Path to the audio file.
    language : str
        Language of the audio file.

    Returns
    -------
    List[Tuple[str, float, float]]
        Transcript with word timestamps.
    """
    with Halo(text='Loading the speech recognition model...', spinner='dots', text_color='green'):
        model_download_path: str = ''.join([MODELS_CONFIG['models_folder'], MODELS_CONFIG['whisper:large-v2']['path']])
        model: Whisper = stable_whisper.load_model(name=MODELS_CONFIG['whisper:large-v2']['name'], device="cpu", dq=True, download_root=model_download_path)
        print(f"{fg('green')}\nSpeech recognition model {MODELS_CONFIG['whisper:large-v2']['name']} loaded successfully{attr('reset')}")

    print(f"{fg('green')}Extracting the transcript...{attr('reset')}")
    transcript = model.transcribe(audio=filename, language=language, verbose=False, word_timestamps=True, demucs=True, vad=True, append_punctuations="\"'.。,，!！?？:：”)]}、")
    formated_transcript: List[Tuple[str, float, float]] = format_transcript(transcript)

    return formated_transcript

def build_timeline(
        text_with_timestamps: List[Tuple[str, float, float]],
        emotion_with_timestamps: List[Tuple[str, float, float]]
) -> List[Tuple[str, float, str]]:
    """
    Builds the ASCII timeline from the text and emotions.

    Parameters
    ----------
    text_with_timestamps : List[Tuple[str, float, float]]
        Transcript with word timestamps.
    emotion_with_timestamps : List[Tuple[str, float, float]]
        Emotions with their corresponding timestamps.

    Returns
    -------
    List[Tuple[str, float, str]]
        ASCII timeline.
    """
    print(f"{fg('green')}Building the timeline{attr('reset')}")
    timeline: List[Tuple[str, float, str]] = []
    all_timestamps: List[float] = sorted(set([timestamp for _, timestamp, _ in text_with_timestamps + emotion_with_timestamps]))

    for timestamp in all_timestamps:
        emotion_entry: str = next((emotion for emotion, start, end in emotion_with_timestamps if start <= float(timestamp) <= end), '')
        timeline.append((timestamp, emotion_entry, ''))

    for text in text_with_timestamps:
        for index, element in enumerate(timeline):
            timestamp, emotion, _ = element

            if text[1] == timestamp:
                text_entry: Tuple[str, float, str] = (timestamp, emotion, text[0])
                timeline[index] = text_entry
                break

    return timeline

def display_elapsed_time(
        elapsed_time: Union[int, float],
        format: str = 'long'
) -> str:
    """
    Returns the elapsed time in seconds in long or short format.

    Parameters
    ----------
    elapsed_time : Union[int, float]
        Elapsed time in seconds.
    format : str, optional
        Format of the elapsed time ('long' or 'short'), by default 'long'.

    Returns
    -------
    str
        Formatted elapsed time.
    """
    if float(elapsed_time) > 60:
        minutes: int = int(elapsed_time // 60)
        seconds: int = int(elapsed_time % 60)
        if format == 'long':
            formatted_time: str = f"{minutes} min {seconds} seconds"
        if format == 'short':
            formatted_time: str = f"{minutes}m{seconds}s"
    else:
        if format == 'long':
            formatted_time: str = f"{elapsed_time} seconds"
        if format == 'short':
            formatted_time: str = f"{elapsed_time}s"

    return formatted_time

def print_timeline(timeline: List[Tuple[Union[int, float], str, str]]) -> None:
    """
    Prints the ASCII timeline.

    Parameters
    ----------
    timeline : List[Tuple[Union[int, float], str, str]]
        ASCII timeline.
    """
    timestamps: List[str] = [f"{timestamp:.2f}" for timestamp, _, _ in timeline]
    text_entries: List[str] = [entry for _, _, entry in timeline]
    emotion_entries: List[str] = [entry for _, entry, _ in timeline]
    time_row: str = ''
    emotion_row: str = ''
    text_row: str = ''

    for timestamp, text, emotion in zip(timestamps, text_entries, emotion_entries):
        text = text.strip()
        timestamp = str(display_elapsed_time(float(timestamp), format='short'))
        emotion = emotion.capitalize()

        time_row_width: int = len(timestamp)
        emotion_row_width: int = len(emotion)
        text_row_width: int = len(text)

        column_width: int = max(time_row_width, emotion_row_width, text_row_width)
        time_row_delta: int = sum([-column_width, time_row_width])
        emotion_row_delta: int = sum([-column_width, emotion_row_width])
        text_row_delta: int = sum([-column_width, text_row_width])

        time_row = ' '.join([time_row, timestamp, * [str(i) for i in range(time_row_delta)], ' '])
        emotion_row = ' '.join([emotion_row, emotion, * [str(i) for i in range(emotion_row_delta)], ' '])
        text_row = ' '.join([text_row, text, * [str(i) for i in range(text_row_delta)], ' '])

    print(''.join([attr('reset'), fg('white'), bg('green'), 'Time (s): ', time_row, attr('reset')]))
    print(''.join([attr('reset'), fg('blue'), bg('yellow'), 'Emotion : ', emotion_row, attr('reset')]))
    print(''.join([attr('reset'), fg('white'), 'Speech  : ', text_row, attr('reset')]))


def main(args: argparse.Namespace) -> None:
    """
    Main function to predict emotions from an audio file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    start_time: float = time.perf_counter()

    if args.train:
        train_model()
        exit(0)

    if not args.train and not args.file:
        print(f"{fg('red')}Please provide an audio file path or use --train to train the model.{attr('reset')}")
        sys.exit(1)

    file: str = args.file
    recognized_emotions: List[Tuple[str, float, float]] = predict_audio_emotion(file, args.language)
    text: List[Tuple[str, float, float]] = extract_transcript(file, args.language)
    timeline: List[Tuple[str, float, str]] = build_timeline(text, recognized_emotions)
    print_timeline(timeline)

    end_time: float = time.perf_counter()
    print(''.join(['\n', 'Execution time: ', display_elapsed_time(end_time - start_time, format='long')]))


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Emotion Prediction from Audio')
    parser.add_argument('--file', type=str, help='Path to the audio file')
    parser.add_argument('--language', type=str, help='Language of the audio file')
    parser.add_argument('--mfcc', action='store_true', help='Include MFCC features', default=True)
    parser.add_argument('--chroma', action='store_true', help='Include chroma features', default=True)
    parser.add_argument('--mel', action='store_true', help='Include mel features', default=True)
    parser.add_argument('--train', action='store_true', help='Train the model', default=False)
    args: argparse.Namespace = parser.parse_args()
    main(args)

