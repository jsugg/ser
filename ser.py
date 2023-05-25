import argparse
import sys
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tqdm

# Emotions supported by the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file)
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

# Extract features (mfcc, chroma, mel) from extended audio frames
def extended_extract_feature(file, mfcc, chroma, mel, frame_size=3, frame_stride=1):
    X, sample_rate = librosa.load(file)
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    num_frames = int(np.ceil(len(X) / frame_step))
    features = []

    for frame in tqdm.tqdm(range(num_frames)):
        start = frame * frame_step
        end = min(start + frame_length, len(X))
        frame_data = X[start:end]

        # Save frame data to temporary file
        temp_filename = 'temp.wav'
        soundfile.write(temp_filename, frame_data, samplerate=sample_rate)

        feature = extract_feature(temp_filename, mfcc, chroma, mel)

        # Remove temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        features.append(feature)

    return features

# Load the data and extract features for each sound file
def load_data(test_size=0.2, feature_extraction='regular'):
    observed_emotions=list(emotions.values()) # Observe all supported emotions, or a subset: observed_emotions=['calm', 'happy', 'fearful', 'disgust']
    x,y=[],[]
    for file in tqdm.tqdm(glob.glob("./dataset/combined-datasets/Actor_*/*.wav")):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue

        if feature_extraction == 'extended':
            feature = extended_extract_feature(file, mfcc=True, chroma=True, mel=True)
        else:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)

        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Train the model
def train_model():
    x_train, x_test, y_train, y_test = load_data(test_size=0.25)
    model = MLPClassifier(
        alpha=0.01, batch_size=256, epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive', max_iter=500
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    pickle.dump(model, open('model.pkl', 'wb'))

# Load the trained model
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))
        return model
    else:
        return None

# Make predictions on an audio file and return emotion with corresponding timestamps
def predict_audio_emotion(file):
    model = load_model()  # Load the trained model
    if model is None:
        print("Model not found. Please train the model first.")
        sys.exit(1)
    feature = extended_extract_feature(file, mfcc=True, chroma=True, mel=True)  # Extract features from the audio file
    feature = np.array(feature).reshape(len(feature), -1)  # Reshape the feature array
    predicted_emotions = model.predict(feature)  # Make the predictions on the feature array

    audio_duration = librosa.get_duration(path=file)  # Get the duration of the audio file

    emotion_with_timestamps = []
    prev_emotion = None
    start_time = None

    for timestamp, emotion in enumerate(predicted_emotions):
        if emotion != prev_emotion:
            # Store the recognized emotion with its start and end timestamps
            if prev_emotion is not None:
                end_time = (timestamp - 1) * audio_duration / len(predicted_emotions)
                emotion_with_timestamps.append((prev_emotion, start_time, end_time))
            prev_emotion = emotion
            start_time = timestamp * audio_duration / len(predicted_emotions)

    # Add the last recognized emotion
    if prev_emotion is not None:
        end_time = audio_duration
        emotion_with_timestamps.append((prev_emotion, start_time, end_time))

    return emotion_with_timestamps

def main(args):
    if args.train:
        train_model()
        exit(0)

    if not args.train and not args.file:
        print("Please provide an audio file path or use --train to train the model.")
        sys.exit(1)

    file = args.file
    recognized_emotions = predict_audio_emotion(file)
    for emotion, start_time, end_time in recognized_emotions:
        print("Emotion: {}, Start Time: {:.2f} seconds, End Time: {:.2f} seconds".format(emotion, start_time, end_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emotion Prediction from Audio')
    parser.add_argument('--file', type=str, help='Path to the audio file')
    parser.add_argument('--mfcc', action='store_true', help='Include MFCC features', default=True)
    parser.add_argument('--chroma', action='store_true', help='Include chroma features', default=True)
    parser.add_argument('--mel', action='store_true', help='Include mel features', default=True)
    parser.add_argument('--train', action='store_true', help='Train the model', default=False)
    args = parser.parse_args()
    main(args)
