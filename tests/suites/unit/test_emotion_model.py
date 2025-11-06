import numpy as np

from ser.models import emotion_model


def test_predict_emotions_collapses_repeated_segments(monkeypatch):
    class DummyModel:
        def predict(self, features):
            return np.array(["happy", "happy", "sad", "sad", "sad"], dtype=object)

    monkeypatch.setattr(emotion_model, "load_model", lambda: DummyModel())
    monkeypatch.setattr(emotion_model, "extended_extract_feature", lambda file: [np.zeros(1)])
    monkeypatch.setattr(emotion_model, "read_audio_file", lambda file: (np.zeros(5), 16000))
    monkeypatch.setattr(emotion_model.librosa, "get_duration", lambda y: 10.0)

    segments = emotion_model.predict_emotions("fake.wav")

    assert segments == [("happy", 0, 4.0), ("sad", 4.0, 10.0)]
