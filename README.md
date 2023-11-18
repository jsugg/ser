<div align="center">
    <img src="https://raw.githubusercontent.com/jsugg/ser/main/.github/assets/DALL%C2%B7E%202023-11-15%2020.42.19%20-%20A%20creative%20and%20informative%20header%20image%20for%20a%20GitHub%20repository%20about%20a%20Speech%20Emotion%20Recognition%20(SER)%20System.%20The%20image%20includes%20a%20symbolic%20represe.png" width="600">
</div>


# Speech Emotion Recognition (SER)
---
## Overview
The `ser` package is a Python package designed to identify and analyze emotions from spoken language. Utilizing cutting-edge machine learning techniques and audio processing algorithms, this package classifies emotions in speech, providing insights into the emotional states conveyed in audio recordings.

```mermaid
sequenceDiagram;
    participant A as Emotion Prediction
    participant B as Transcript Extraction
    participant C as Timeline Integration
    A->>C: Emotion with Timestamps
    B->>C: Transcript with Timestamps
    C->>C: Integrate and Align
```

### Features
- **Emotion Classification Model**: Trains on a dataset of audio files for accurate emotion recognition.
- **Emotion Prediction**: Predicts emotions from provided audio files.
- **Transcript Extraction**: Extracts a transcript of spoken words in the audio file.
- **Timeline Integration**: Builds a comprehensive timeline integrating recognized emotions with the corresponding transcript.
- **CLI Interface**: Offers command-line options for user interaction.
-------
### Workflows and architectures

```mermaid
graph TD;
    A[Audio Input] --> B[Feature Extraction];
    B --> C[Emotion Classification Model];
    A --> D[Transcript Extraction];
    C --> E[Emotion Prediction];
    D --> F[Transcript];
    E --> G[Timeline Integration];
    F --> G;
    G --> H[Output];
```
-------

## Installation

```bash
git clone https://github.com/jsugg/ser/
cd ser
pip install -r requirements.txt
```

-----
## Usage
------
### Training the Model
To train the emotion classification model:

```bash
python -m ser.ser --train
```
```mermaid
graph TD;
    A[Data Loading] --> B[Data Splitting];
    B --> C[Train Model];
    B --> D[Test Model];
    C --> E[Model Validation];
    E --> F[Trained Model];
```
-------
### Predicting Emotions
To predict emotions in an audio file:

```bash
python -m ser.ser --file audio.mp3
```
```mermaid
graph LR;
    A[Audio Data] -->|Preprocessing| B[Feature Extraction];
    B -->|Feature Set| C[Model Prediction];
    C -->|Emotion Labels| D[Output];
    A -->|Transcription| E[Transcript Extraction];
    E -->|Transcript| D;
```
-------
### Additional Options
* Specify language: **`--language <language>--`**
* Save transcript: **`--save_transcript`**

---
## Modules
* **`transcript_extractor`**: Extracts transcripts from audio files.
* **`audio_utils`**: Utilities for audio processing.
* **`feature_extractor`**: Extracts audio features for model training.
* **`emotion_model`**: Contains the emotion classification model.

```mermaid
graph TD;
    A[User Input] -->|Train Command| B[Train Model];
    A -->|Predict Command| C[Predict Emotion];
    C --> D[Display Emotion];
    A -->|Transcript Command| E[Extract Transcript];
    E --> F[Display Transcript];
```
---

## Configuration
Edit **`ser/config.py`** to modify default configurations, including model paths, dataset paths, and feature extraction settings.

---

## Contributing
Contributions to the SER System are welcome! Please read **`CONTRIBUTING.md`** for details on our code of conduct and the process for submitting pull requests.

---

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

---

## Acknowledgments
- **Libraries and Frameworks**: Special thanks to the developers and maintainers of `librosa`, `openai-whisper`, `stable-whisper`, `numpy`, `scikit-learn`, `soundfile`, `tqdm`, and for their invaluable tools that made this project possible.
- **Datasets**: Gratitude to the creators of the RAVDESS and Emo-DB datasets for providing high-quality audio data essential for training the models.
- **Inspirational Sources**: Inspired by [Models-based representations for speech emotion recognition](https://arxiv.org/abs/2311.00394)
