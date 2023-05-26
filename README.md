# ser
Speech Emotion Recognition

# Speech Emotion Recognition (SER)

![Speech Emotion Recognition Workflow](https://showme.redstarplugin.com/s/PiLLulcn)

This project is a small-scale implementation of Speech Emotion Recognition (SER) using a model trained on a small dataset. The goal is to improve the model further in the next iterations.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction
The SER model is trained on the RAVDESS dataset and uses MFCC (Mel Frequency Cepstral Coefficients), Chroma, and Mel features extracted from the audio files. The model is a Multi-Layer Perceptron (MLP) classifier implemented using the sklearn library.

The project structure is as follows:
```
.
├── dataset/ravdess/
├── .gitignore
├── Pipfile
├── README.md
├── model.pkl
├── sample.wav
└── ser.py
```

## Installation
To get started with this project, clone the repository and install the dependencies listed in the Pipfile.

The datasets under dataset/combined-datasets onsolidate the RAVDESS and the TESS datasets.
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) can be downloaded free of charge in the PLOSE ONE journal for Medicine & Health [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391).
The Toronto emotional speech set (TESS) can be downloaded free of charge in the University of Toronto's Department of Psychology website [here](https://tspace.library.utoronto.ca/handle/1807/24501)



## Usage
The main script is `ser.py`. It can be used to train the model or to predict the emotion from an audio file.

To train the model, use the following command:
```bash
python ser.py --train
```

To predict the emotion from an audio file, use the following command:
```bash
python ser.py --file <path_to_audio_file>
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See `LICENSE` for more information.

The diagram above provides a high-level overview of the workflow of the Speech Emotion Recognition system. The audio input is first processed to extract MFCC, Chroma, and Mel features. The data is then split into a training set and a testing set. The MLPClassifier model is trained on the training set. The model's performance is evaluated by making predictions on the testing set and calculating the accuracy score. The accuracy of the current model is 85%.
