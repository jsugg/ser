"""Behavior tests for in-memory feature extraction paths."""

from types import SimpleNamespace, TracebackType

import numpy as np
import pytest

from ser.features import feature_extractor as fe


class DummyHalo:
    """No-op replacement for spinner context manager during tests."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> "DummyHalo":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        return None


def test_extract_feature_from_signal_combines_enabled_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All enabled feature blocks should be concatenated into one vector."""
    monkeypatch.setattr(
        fe,
        "get_settings",
        lambda: SimpleNamespace(
            feature_flags=SimpleNamespace(
                mfcc=True,
                chroma=True,
                mel=True,
                contrast=True,
                tonnetz=True,
            )
        ),
    )
    monkeypatch.setattr(
        fe.librosa, "stft", lambda _audio, n_fft: np.ones((n_fft // 2 + 1, 4))
    )
    monkeypatch.setattr(
        fe.librosa, "power_to_db", lambda array, ref=None: np.asarray(array)
    )
    monkeypatch.setattr(
        fe.librosa.feature, "mfcc", lambda **_kwargs: np.ones((40, 4), dtype=np.float32)
    )
    monkeypatch.setattr(
        fe.librosa.feature,
        "chroma_stft",
        lambda **_kwargs: np.ones((12, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        fe.librosa.feature,
        "melspectrogram",
        lambda **_kwargs: np.ones((128, 4), dtype=np.float32),
    )
    monkeypatch.setattr(
        fe.librosa.feature,
        "spectral_contrast",
        lambda **_kwargs: np.ones((7, 4), dtype=np.float32),
    )
    monkeypatch.setattr(fe.librosa.effects, "harmonic", lambda audio: audio)
    monkeypatch.setattr(
        fe.librosa.feature,
        "tonnetz",
        lambda **_kwargs: np.ones((6, 4), dtype=np.float32),
    )

    feature_vector = fe.extract_feature_from_signal(
        np.ones(1024, dtype=np.float32), sample_rate=16000
    )

    assert feature_vector.shape == (193,)


def test_extract_feature_from_signal_rejects_empty_audio() -> None:
    """Empty buffers must fail with a clear validation error."""
    with pytest.raises(ValueError, match="no samples"):
        fe.extract_feature_from_signal(
            np.asarray([], dtype=np.float32), sample_rate=16000
        )


def test_extended_extract_feature_uses_in_memory_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame extraction should avoid temporary files and process in-memory slices."""
    monkeypatch.setattr(
        fe, "read_audio_file", lambda _path: (np.arange(10, dtype=np.float32), 2)
    )
    monkeypatch.setattr(fe, "Halo", DummyHalo)
    monkeypatch.setattr(
        fe,
        "extract_feature_from_signal",
        lambda audio, _sample_rate: np.asarray([float(audio.size)], dtype=np.float64),
    )

    features = fe.extended_extract_feature("sample.wav", frame_size=2, frame_stride=1)

    assert [int(item[0]) for item in features] == [4, 4, 4, 4, 2]


def test_extended_extract_feature_rejects_invalid_window_arguments() -> None:
    """Frame size and stride must be positive."""
    with pytest.raises(ValueError, match="frame_size"):
        fe.extended_extract_feature("sample.wav", frame_size=0, frame_stride=1)
    with pytest.raises(ValueError, match="frame_stride"):
        fe.extended_extract_feature("sample.wav", frame_size=1, frame_stride=0)
