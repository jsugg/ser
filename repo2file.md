# Project Codebase
## Root directory
 `/Users/juanpedrosugg/dev/github/ser`
---
## Directory structure:
```
.
├── Pipfile
├── Pipfile.lock
├── README.md
├── repo2file.md
├── sample.wav
├── ser
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── configure
│   ├── data
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── dataset
│   ├── features
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── models
│   │   ├── OpenAI
│   │   │   └── whisper
│   │   │       └── large-v2.pt
│   │   ├── __init__.py
│   │   ├── emotion_model.py
│   │   └── ser_model.pkl
│   ├── tests
│   │   └── test_suite.py
│   ├── tmp
│   ├── transcript
│   │   ├── __init__.py
│   │   └── transcript_extractor.py
│   └── utils
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── common_utils.py
│       ├── logger.py
│       └── timeline_utils.py
├── setup.py
└── transcripts

```

---
## File: ser/__main__.py
```
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
  specific features for extraction and analysis.

Usage:
    The system can be operated in two primary modes:
    1. Training mode: Trains the model using labeled audio data.
    2. Prediction mode: Predicts emotions in a given audio file and 
       extracts the transcript.

Author: Juan Sugg (juanpedrosugg [at] gmail.com)
Version: 1.0
License: MIT
"""

import argparse
import sys
import time
import logging
from typing import List, Tuple

from ser.models.emotion_model import predict_emotions, train_model
from ser.transcript.transcript_extractor import extract_transcript
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
        description="Speech Emotion Recognition System"
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
```

---
## File: ser/tests/test_suite.py
```
import unittest

import psutil

from ser.features.feature_extractor import extract_feature
from ser.models.emotion_model import train_model, predict_emotions
from ser.transcript.transcript_extractor import extract_transcript
from ser.utils.timeline_utils import build_timeline


class TestFeatureExtraction(unittest.TestCase):
    def test_feature_extraction(self):
        # Test Case ID: TC_FE_001
        # Test Case Title: Basic Feature Extraction
        audio_file = "sample.wav"
        expected_features = [...]  # Replace with expected feature values

        features = extract_feature(audio_file)
        self.assertEqual(
            features.tolist(),
            expected_features,
            "Basic feature extraction failed.",
        )


class TestRegressionTesting(unittest.TestCase):
    def test_data_compatibility(self):
        # Test Case ID: TC_FE_007
        # Test Case Title: Data Compatibility Testing
        audio_files = [
            "path/to/audio1.wav",
            "path/to/audio2.mp3",
            "path/to/audio3.flac",
        ]

        for audio_file in audio_files:
            features = extract_feature(audio_file)
            self.assertIsNotNone(
                features, f"Feature extraction failed for {audio_file}"
            )


class TestAudioFileCompatibility(unittest.TestCase):
    def test_compatibility_with_multiple_formats(self):
        # Test Case ID: TC_COMPAT_001
        # Test Case Title: Compatibility with Multiple Formats
        audio_files = [
            "path/to/audio1.wav",
            "path/to/audio2.mp3",
            "path/to/audio3.flac",
        ]

        for audio_file in audio_files:
            features = extract_feature(audio_file)
            self.assertIsNotNone(
                features, f"Feature extraction failed for {audio_file}"
            )


class TestBasicSystemFunctionality(unittest.TestCase):
    def test_system_initialization(self):
        # Test Case ID: TC_SMK_001
        # Test Case Title: System Initialization
        try:
            train_model()
        except Exception as e:
            self.fail(f"System initialization failed: {e}")

    def test_audio_input(self):
        # Test Case ID: TC_SMK_002
        # Test Case Title: Audio Input
        audio_file = "path/to/sample.wav"

        try:
            features = extract_feature(audio_file)
            self.assertIsNotNone(
                features, "System failed to accept audio input."
            )
        except Exception as e:
            self.fail(f"System failed to accept audio input: {e}")

    def test_emotion_classification(self):
        # Test Case ID: TC_SMK_003
        # Test Case Title: Emotion Classification
        audio_file = "path/to/emotion_audio.wav"

        try:
            emotions = predict_emotions(audio_file)
            self.assertIsInstance(
                emotions, list, "Emotion classification failed."
            )
        except Exception as e:
            self.fail(f"Emotion classification failed: {e}")


class TestSystemStability(unittest.TestCase):
    def test_load_testing(self):
        # Test Case ID: TC_PERF_002
        # Test Case Title: Load Testing
        try:
            for _ in range(1000):  # Simulate high load
                features = extract_feature("path/to/sample.wav")
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Load testing failed: {e}")

    def test_resource_utilization(self):
        # Test Case ID: TC_PERF_003
        # Test Case Title: Resource Utilization

        try:
            process = psutil.Process()
            initial_cpu = process.cpu_percent(interval=1)
            initial_memory = process.memory_info().rss

            for _ in range(1000):  # Simulate high load
                features = extract_feature("path/to/sample.wav")

            final_cpu = process.cpu_percent(interval=1)
            final_memory = process.memory_info().rss

            self.assertLess(
                final_cpu, initial_cpu * 2, "CPU utilization is too high."
            )
            self.assertLess(
                final_memory,
                initial_memory * 2,
                "Memory utilization is too high.",
            )
        except Exception as e:
            self.fail(f"Resource utilization test failed: {e}")


class TestCommandLineInterface(unittest.TestCase):
    def test_command_line_usage(self):
        # Test Case ID: TC_SMK_003
        # Test Case Title: Command-Line Usage
        import subprocess

        try:
            result = subprocess.run(
                ["python3", "ser/ser.py", "--file", "path/to/sample.wav"],
                capture_output=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                "Command-line interface failed with valid arguments.",
            )
        except Exception as e:
            self.fail(f"Command-line interface usage test failed: {e}")

    def test_help_command(self):
        # Test Case ID: TC_USABILITY_001
        # Test Case Title: Help Command
        import subprocess

        try:
            result = subprocess.run(
                ["python3", "ser/ser.py", "--help"], capture_output=True
            )
            self.assertEqual(result.returncode, 0, "Help command failed.")
            self.assertIn(
                "usage:",
                result.stdout.decode(),
                "Help information is not displayed.",
            )
        except Exception as e:
            self.fail(f"Help command test failed: {e}")

    def test_invalid_command(self):
        # Test Case ID: TC_USABILITY_002
        # Test Case Title: Invalid Command
        import subprocess

        try:
            result = subprocess.run(
                ["python3", "ser/ser.py", "--invalid"], capture_output=True
            )
            self.assertNotEqual(
                result.returncode, 0, "Invalid command handling failed."
            )
            self.assertIn(
                "error:",
                result.stderr.decode(),
                "Invalid command error message is not displayed.",
            )
        except Exception as e:
            self.fail(f"Invalid command test failed: {e}")


class TestEdgeCaseHandling(unittest.TestCase):
    def test_out_of_memory_scenario(self):
        # Test Case ID: TC_EXP_001
        # Test Case Title: Out-of-Memory Scenario
        import resource

        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (1024 * 1024 * 512, 1024 * 1024 * 512)
            )  # Set limit to 512MB
            with self.assertRaises(MemoryError):
                features = extract_feature("path/to/large_audio.wav")
        except Exception as e:
            self.fail(f"Out-of-memory scenario test failed: {e}")

    def test_unexpected_input(self):
        # Test Case ID: TC_EXP_002
        # Test Case Title: Unexpected Input
        try:
            features = extract_feature("path/to/unexpected_input.wav")
            self.assertIsNotNone(
                features, "System failed to handle unexpected input."
            )
        except Exception as e:
            self.fail(f"Unexpected input test failed: {e}")

    def test_network_disruption(self):
        # Test Case ID: TC_EXP_003
        # Test Case Title: Network Disruption
        import requests
        from unittest.mock import patch

        try:
            with patch("requests.get", side_effect=requests.ConnectionError):
                features = extract_feature("path/to/network_audio.wav")
            self.assertIsNotNone(
                features, "System failed to handle network disruption."
            )
        except Exception as e:
            self.fail(f"Network disruption test failed: {e}")


class TestPerformanceTesting(unittest.TestCase):
    def test_large_file_handling(self):
        # Test Case ID: TS_PERF_001
        # Test Title: Large File Handling
        audio_file = "path/to/large_audio_file.wav"
        try:
            features = extract_feature(audio_file)
            self.assertIsNotNone(
                features, "System failed to handle large audio file."
            )
        except Exception as e:
            self.fail(f"Large file handling test failed: {e}")


class TestSecurityTesting(unittest.TestCase):
    def test_data_security_and_compliance(self):
        # Test Case ID: TS_SEC_001
        # Test Title: Data Security and Compliance
        try:
            # Assuming extract_feature does not expose sensitive data and uses secure methods
            features = extract_feature("path/to/sample.wav")
            self.assertIsNotNone(
                features, "Data security and compliance test failed."
            )
        except Exception as e:
            self.fail(f"Data security and compliance test failed: {e}")


class TestUsabilityTesting(unittest.TestCase):
    def test_command_line_interface(self):
        # Test Case ID: TS_USABILITY_001
        # Test Title: Command-Line Interface Tests
        import subprocess

        try:
            result = subprocess.run(
                ["python3", "ser/ser.py", "--file", "path/to/sample.wav"],
                capture_output=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                "Command-line interface failed with valid arguments.",
            )
            self.assertIn(
                "Emotion prediction completed",
                result.stdout.decode(),
                "CLI did not function as expected.",
            )
        except Exception as e:
            self.fail(f"Command-line interface test failed: {e}")


class TestIntegrationTesting(unittest.TestCase):
    def test_end_to_end_workflow(self):
        # Test Case ID: TS_INT_001
        # Test Title: End-to-End Workflow Tests
        audio_file = "path/to/audio.wav"
        language = "en"
        try:
            emotions = predict_emotions(audio_file)
            transcript = extract_transcript(audio_file, language)
            timeline = build_timeline(transcript, emotions)
            self.assertTrue(timeline, "End-to-end workflow test failed.")
        except Exception as e:
            self.fail(f"End-to-end workflow test failed: {e}")

    def test_multi_module_coordination(self):
        # Test Case ID: TS_INT_002
        # Test Title: Multi-Module Coordination Tests
        audio_file = "path/to/audio.wav"
        language = "en"
        try:
            emotions = predict_emotions(audio_file)
            transcript = extract_transcript(audio_file, language)
            timeline = build_timeline(transcript, emotions)
            self.assertTrue(timeline, "Multi-module coordination test failed.")
        except Exception as e:
            self.fail(f"Multi-module coordination test failed: {e}")


class TestAcceptanceTesting(unittest.TestCase):
    def test_real_world_scenario(self):
        # Test Case ID: TS_ACCEPTANCE_001
        # Test Title: Real-World Scenario Simulation
        audio_file = "path/to/real_world_audio.wav"
        try:
            emotions = predict_emotions(audio_file)
            self.assertIsInstance(
                emotions, list, "Real-world scenario test failed."
            )
        except Exception as e:
            self.fail(f"Real-world scenario test failed: {e}")


if __name__ == "__main__":
    unittest.main()
```

---
## File: ser/utils/logger.py
```
import logging

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger: logging.Logger = logging.getLogger(name)
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(format=LOG_FORMAT, level=level)
    return logger
```

---
## File: .github/workflows/python-publish.yml
```
# This workflow will upload a Python Package using Twine when a release is created
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries

# This workflow uses actions that are not certified by GitHub.
# They are provided by a third-party and are governed by
# separate terms of service, privacy policy, and support
# documentation.

name: Upload Python Package

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  deploy:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build
    - name: Build package
      run: python -m build
    - name: Publish package
      uses: pypa/gh-action-pypi-publish@27b31702a0e7fc50959f5ad993c78deac1bdfc29
      with:
        user: __token__
        password: ${{ secrets.PYPI_API_TOKEN }}
```

---
## File: ser/.gitignore
```
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
#   For a library or package, you might want to ignore these files since the code is
#   intended to run in multiple environments; otherwise, check them in:
# .python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# poetry
#   Similar to Pipfile.lock, it is generally recommended to include poetry.lock in version control.
#   This is especially recommended for binary packages to ensure reproducibility, and is more
#   commonly ignored for libraries.
#   https://python-poetry.org/docs/basic-usage/#commit-your-poetrylock-file-to-version-control
#poetry.lock

# pdm
#   Similar to Pipfile.lock, it is generally recommended to include pdm.lock in version control.
#pdm.lock
#   pdm stores project-wide configurations in .pdm.toml, but it is recommended to not include it
#   in version control.
#   https://pdm.fming.dev/#use-with-ide
.pdm.toml

# PEP 582; used by e.g. github.com/David-OConnor/pyflow and github.com/pdm-project/pdm
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
#  JetBrains specific template is maintained in a separate JetBrains.gitignore that can
#  be found at https://github.com/github/gitignore/blob/main/Global/JetBrains.gitignore
#  and can be added to the global gitignore or merged into this file.  For a more nuclear
#  option (not recommended) you can uncomment the following to ignore the entire idea folder.
#.idea/

.venv
.DS_Store

*/**/__pycache__
*/__pycache__```

---
## File: ser/__init__.py
```
from .models.emotion_model import predict_emotions, train_model
from .config import Config
from .transcript.transcript_extractor import extract_transcript
from .utils.timeline_utils import build_timeline, print_timeline, save_timeline_to_csv
```

---
## File: ser/config.py
```
import os
from typing import Any
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()


class Config:
    """
    Central configuration class for the SER application.
    """

    # Emotions supported by the dataset
    EMOTIONS: dict[str, str] = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    # Temporary folder for processing
    TMP_FOLDER: str = "./ser/tmp"

    # Default feature extraction configuration
    DEFAULT_FEATURE_CONFIG: dict[str, bool] = {
        "mfcc": True,
        "chroma": True,
        "mel": True,
        "contrast": True,
        "tonnetz": True,
    }

    # Neural network parameters for MLP Classifier
    NN_PARAMS: dict[str, Any] = {
        "alpha": 0.01,
        "batch_size": 256,
        "epsilon": 1e-08,
        "hidden_layer_sizes": (300,),
        "learning_rate": "adaptive",
        "max_iter": 500,
    }

    # Audio file read parameters
    AUDIO_READ_CONFIG: dict[str, int] = {
        "max_retries": 3,
        "retry_delay": 1,  # in seconds
    }

    # Dataset configuration
    DATASET: dict[str, str] = {
        "folder": os.getenv("DATASET_FOLDER", "ser/dataset/ravdess"),
        "subfolder_prefix": "Actor_*",
        "extension": "*.wav",
    }

    # Model configuration
    MODELS_CONFIG: dict[str, Any] = {
        "models_folder": "./ser/models",
        "whisper_model": {"name": "large-v2", "path": "OpenAI/whisper/"},
        "num_cores": os.cpu_count(),
    }

    # Transcript configuration
    TIMELINE_CONFIG: dict[str, str] = {"folder": "./transcripts"}

    # Language settings
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "en")
    FILE_SETTING: None | str = None
    TRAIN_MODE: bool = False
```

---
## File: ser/data/data_loader.py
```
import os
import glob
import logging
from typing import List, Tuple, Optional
import multiprocessing as mp
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

from ser.utils import get_logger
from ser.features.feature_extractor import extract_feature
from ser.config import Config


logger: logging.Logger = get_logger(__name__)


def process_file(
    file: str, observed_emotions: List[str]
) -> Tuple[np.ndarray, str]:
    """
    Process an audio file to extract features and the associated emotion label.

    Arguments:
        file (str): Path to the audio file.
        observed_emotions (List[str]): List of observed emotions.

    Returns:
        Optional[Tuple[np.ndarray, str]]: Extracted features and associated emotion label for the audio file.
        Returns None if the emotion is not in observed_emotions.
    """
    try:
        file_name: str = os.path.basename(file)
        emotion: Optional[str] = Config.EMOTIONS.get(file_name.split("-")[2])

        if not emotion or emotion not in observed_emotions:
            return (np.array([]), "")
        features: np.ndarray = extract_feature(file)

        return (features, emotion)
    except Exception as e:
        logger.error(msg=f"Failed to process file {file}: {e}")
        raise e


def load_data(test_size: float = 0.2) -> Optional[List]:
    """
    Load data from the dataset directory and split into training and testing sets.

    Arguments:
        test_size (float): Fraction of the dataset to be used as test set.

    Returns:
        Tuple containing training features, training labels, test features, and test labels.
    """
    observed_emotions: List[str] = list(Config.EMOTIONS.values())
    data: List[Tuple[np.ndarray, str]]
    data_path_pattern: str = (
        f"{Config.DATASET['folder']}/"
        f"{Config.DATASET['subfolder_prefix']}/"
        f"{Config.DATASET['extension']}"
    )
    files: List[str] = glob.glob(data_path_pattern)

    with mp.Pool(int(Config.MODELS_CONFIG["num_cores"])) as pool:
        data = pool.map(
            partial(process_file, observed_emotions=observed_emotions), files
        )

    # Remove None entries from data list
    data = [item for item in data if item is not None]
    if not data:
        logger.warning("No data found or processed.")
        return None

    features: Tuple[np.ndarray, ...]
    labels: Tuple[str, ...]
    features, labels = zip(*data)
    return train_test_split(
        np.array(features), labels, test_size=test_size, random_state=42
    )
```

---
## File: ser/features/feature_extractor.py
```
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
warnings.filterwarnings("ignore", message=".*is too large for input signal of length.*")


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
```

---
## File: ser/models/emotion_model.py
```
import os
import warnings
import logging
import pickle
from typing import Optional, Tuple, List

import numpy as np
import librosa
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from halo import Halo

from ser.utils import get_logger
from ser.config import Config
from ser.data.data_loader import load_data
from ser.features.feature_extractor import extended_extract_feature
from ser.utils.audio_utils import read_audio_file


logger: logging.Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*Cannot set number of intraop threads after parallel work has started.*")


def train_model() -> None:
    """
    Train the emotion classification model.

    This function loads the dataset, trains an MLPClassifier on the training data,
    measures the model's accuracy on the test data, and saves the trained model to a file.

    Raises:
        Exception: If the dataset is not loaded successfully.
    """
    with Halo(text="Loading dataset... ", spinner="dots", text_color="green"):
        if data := load_data(test_size=0.25):
            x_train, x_test, y_train, y_test = data
            model: MLPClassifier = MLPClassifier(**Config.NN_PARAMS)
            logger.info(msg="Dataset loaded successfully.")
        else:
            logger.error("Dataset not loaded. Please load the dataset first.")
            raise RuntimeError(
                "Dataset not loaded. Please load the dataset first."
            )

    with Halo(
        text="Training the model... ", spinner="dots", text_color="green"
    ):
        model.fit(x_train, y_train)
    logger.info(msg=f"Model trained with {len(x_train)} samples")

    with Halo(
        text="Measuring accuracy... ", spinner="dots", text_color="green"
    ):
        y_pred: np.ndarray = model.predict(x_test)
        accuracy: float = float(accuracy_score(y_true=y_test, y_pred=y_pred))
        model_file: str = (
            f"{Config.MODELS_CONFIG['models_folder']}/ser_model.pkl"
        )
    logger.info(msg=f"Accuracy: {accuracy * 100:.2f}%")

    with Halo(text="Saving the model... ", spinner="dots", text_color="green"):
        pickle.dump(model, open(model_file, "wb"))
    logger.info(msg=f"Model saved to {model_file}")


def load_model() -> MLPClassifier:
    """
    Load the trained emotion classification model.

    This function loads the trained MLPClassifier model from a file.

    Returns:
        MLPClassifier: The trained emotion classification model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path: str = f'{Config.MODELS_CONFIG["models_folder"]}/ser_model.pkl'
    model: MLPClassifier | None = None

    with Halo(
        text=f"Loading SER model from {model_path}... ",
        spinner="dots",
        text_color="green",
    ):
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, "rb"))
    if model:
        logger.info(msg=f"Model loaded from {model_path}")
        return model

    logger.error(
        msg=(
            "Model not found. Please train the model first. "
            "If you already trained the model, "
            f"please ensure that {model_path} exists and it's a valid .pkl file"
        )
    )
    raise FileNotFoundError(
        "Model not found. Please train the model first. "
        "If you already trained the model, "
        f"please ensure that {model_path} exists and it's a valid .pkl file."
    )


def predict_emotions(file: str) -> List[Tuple[str, float, float]]:
    """
    Predict emotions from an audio file.

    This function loads a trained model, extracts features from the audio file,
    predicts emotions at each timestamp, and returns a list of predicted emotions
    with their start and end timestamps.

    Arguments:
        file (str): Path to the audio file.

    Returns:
        List[Tuple[str, float, float]]: A list of tuples where each tuple contains the
        predicted emotion, start time, and end time.

    Raises:
        Exception: If the model is not loaded.
    """
    model: MLPClassifier = load_model()
    if model is None:
        raise RuntimeError("Model not loaded.")

    with Halo(
        text="Inferring Emotions from Audio File... ",
        spinner="dots",
        text_color="green",
    ):
        feature: List[np.ndarray] = extended_extract_feature(file)
        predicted_emotions: np.ndarray = model.predict(feature)
    logger.info(msg="Emotion inference completed.")

    audio_duration: float = librosa.get_duration(y=read_audio_file(file)[0])
    emotion_timestamps: List[Tuple[str, float, float]] = []
    prev_emotion: Optional[str] = None
    start_time: float = 0

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            if prev_emotion is not None:
                end_time = timestamp * audio_duration / len(predicted_emotions)
                emotion_timestamps.append((prev_emotion, start_time, end_time))
            (
                prev_emotion,
                start_time,
            ) = emotion, timestamp * audio_duration / len(predicted_emotions)

    if prev_emotion is not None:
        emotion_timestamps.append((prev_emotion, start_time, audio_duration))

    logger.info("Emotion prediction and timestamp extraction completed.")
    return emotion_timestamps
```

---
## File: ser/transcript/transcript_extractor.py
```
import logging
from typing import Tuple, List, Any
import warnings

import stable_whisper
from halo import Halo
from whisper.model import Whisper

from ser.utils import get_logger
from ser.config import Config


logger: logging.Logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings(
    "ignore",
    message=".*Cannot set number of intraop threads after parallel work has started.*",
)


def load_whisper_model() -> Whisper:
    """
    Loads the Whisper model specified in the configuration.

    Returns:
        stable_whisper.Whisper: Loaded Whisper model.
    """
    try:
        model: Whisper = stable_whisper.load_model(
            name=Config.MODELS_CONFIG["whisper_model"]["name"],
            device="cpu",
            dq=False,
            download_root=(
                f"{Config.MODELS_CONFIG['models_folder']}/"
                f"{Config.MODELS_CONFIG['whisper_model']['path']}"
            ),
            in_memory=True,
        )
        return model
    except Exception as e:
        logger.error(msg=f"Failed to load Whisper model: {e}", exc_info=True)
        raise


def extract_transcript(
    file_path: str, language: str = Config.DEFAULT_LANGUAGE
) -> List[Tuple[str, float, float]]:
    """
    Extracts the transcript from an audio file using the Whisper model.

    Arguments:
        file_path (str): Path to the audio file.
        language (str): Language of the audio.

    Returns:
        list: List of tuples (word, start_time, end_time).
    """
    try:
        with Halo(
            text="Loading the Whisper model...",
            spinner="dots",
            text_color="green",
        ):
            model: Whisper = load_whisper_model()
        logger.info(msg="Whisper model loaded successfully.")

        with Halo(
            text="Generating the transcript...",
            spinner="dots",
            text_color="green",
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                transcript: dict = model.transcribe(
                    audio=file_path,
                    language=language,
                    verbose=False,
                    word_timestamps=True,
                    no_speech_threshold=None,
                    demucs=True,
                    vad=True,
                )
            formatted_transcript: List[Tuple[str, float, float]] = (
                format_transcript(transcript)
            )

        logger.info("Transcript extraction completed successfully.")
        return formatted_transcript
    except Exception as e:
        logger.error(msg=f"Failed to extract transcript: {e}", exc_info=True)
        raise


def format_transcript(result) -> List[Tuple[str, float, float]]:
    """
    Formats the transcript into a list of tuples containing the word,
    start time, and end time.

    Args:
        result (dict): The transcript result.

    Returns:
        List[Tuple[str, float, float]]: Formatted transcript with timestamps.
    """
    words: Any = result.all_words()

    text_with_timestamps: List[Tuple[str, float, float]] = [
        (word.word, word.start, word.end) for word in words
    ]
    return text_with_timestamps
```

---
## File: ser/utils/__init__.py
```
from .logger import get_logger
from .audio_utils import read_audio_file
from .common_utils import display_elapsed_time
from .timeline_utils import build_timeline, print_timeline, save_timeline_to_csv
```

---
## File: ser/utils/audio_utils.py
```
import time
import warnings
import logging
from typing import Any

from numpy import ndarray
import numpy as np
import librosa
import soundfile as sf

from ser.utils import get_logger
from ser.config import Config


logger: logging.Logger = get_logger(__name__)


def read_audio_file(file_path: str) -> tuple[ndarray, Any]:
    """
    Read an audio file.

    Arguments:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Audio data.
    """
    logger.debug(msg=f"Starting to read audio file: {file_path}")
    for attempt in range(Config.AUDIO_READ_CONFIG["max_retries"]):
        logger.debug(
            msg=f"Attempt {attempt + 1} to read audio file using librosa."
        )
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                audiofile, current_sample_rate = librosa.load(
                    file_path, sr=None
                )
            audiofile: ndarray = np.array(audiofile, dtype=np.float32)

            max_abs_value: Any = np.max(np.abs(audiofile))
            if max_abs_value == 0:
                audiofile = np.zeros_like(audiofile)
            else:
                audiofile /= max_abs_value

            logger.debug(
                msg=f"Successfully read audio file using librosa: {file_path}"
            )
            return audiofile, current_sample_rate

        except Exception as e:
            logger.warning(msg=f"Librosa failed to read audio file: {e}")
            logger.warning(msg="Falling back to soundfile...")
            try:
                with sf.SoundFile(file_path) as sound_file:
                    audiofile: ndarray = sound_file.read(dtype="float32")
                    current_sample_rate = sound_file.samplerate

                max_abs_value: Any = np.max(np.abs(audiofile))
                if max_abs_value == 0:
                    audiofile = np.zeros_like(audiofile)
                else:
                    audiofile /= max_abs_value

                logger.debug(
                    msg=(
                        "Successfully read audio file using soundfile: "
                        f"{file_path}"
                    )
                )
                return audiofile, current_sample_rate

            except Exception as err:
                logger.warning(msg=f"Soundfile also failed: {err}")
                logger.info(
                    msg=(
                        "Retrying with librosa in "
                        f"{Config.AUDIO_READ_CONFIG['retry_delay']} seconds..."
                    )
                )
                time.sleep(Config.AUDIO_READ_CONFIG["retry_delay"])

    logger.error(
        msg=(
            f"Failed to read audio file {file_path} "
            f"after {Config.AUDIO_READ_CONFIG['max_retries']} retries."
        )
    )
    raise IOError(f"Error reading {file_path}")
```

---
## File: ser/utils/common_utils.py
```
def display_elapsed_time(elapsed_time: float, _format: str = 'long') -> str:
    """
    Returns the elapsed time in seconds in long or short format.

    Arguments:
        elapsed_time (float): Elapsed time in seconds.
        format (str, optional): Format of the elapsed time 
            ('long' or 'short'), by default 'long'.

    Returns:
        str: Formatted elapsed time.
    """
    minutes, seconds = divmod(int(elapsed_time), 60)
    if _format == 'long':
        return f"{minutes} min {seconds} seconds"
    return f"{minutes}m{seconds}s"
```

---
## File: ser/utils/timeline_utils.py
```
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
```

---
