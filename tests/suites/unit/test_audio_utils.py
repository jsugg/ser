from unittest import mock

import numpy as np
import pytest

from ser.utils.audio_utils import read_audio_file


@pytest.fixture(autouse=True)
def _patch_sleep(monkeypatch):
    monkeypatch.setattr("ser.utils.audio_utils.time.sleep", lambda *_: None)


def test_read_audio_file_falls_back_and_eventually_raises(monkeypatch):
    audio_data = np.array([0.0, 0.5, -0.5], dtype=np.float32)

    load_calls = {}

    def failing_librosa_load(*args, **kwargs):
        load_calls["librosa"] = load_calls.get("librosa", 0) + 1
        raise RuntimeError("boom")

    class DummySoundFile:
        def __init__(self, *args, **kwargs):
            load_calls["soundfile"] = load_calls.get("soundfile", 0) + 1

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, dtype=None):
            return audio_data

        @property
        def samplerate(self):
            return 16000

    monkeypatch.setattr("ser.utils.audio_utils.librosa.load", failing_librosa_load)
    monkeypatch.setattr("ser.utils.audio_utils.sf.SoundFile", DummySoundFile)

    audio, rate = read_audio_file("fake.wav")

    assert pytest.approx(audio.tolist()) == [0.0, 1.0, -1.0]
    assert rate == 16000
    assert load_calls["librosa"] == 1
    assert load_calls["soundfile"] == 1

    def failing_soundfile(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("ser.utils.audio_utils.sf.SoundFile", failing_soundfile)

    with pytest.raises(IOError):
        read_audio_file("fake.wav")
