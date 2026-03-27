"""Unit tests for DSP-level handcrafted feature extraction helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ser.config import FeatureFlags
from ser.utils import dsp

pytestmark = [
    pytest.mark.unit,
    pytest.mark.filterwarnings(
        r"ignore:path is deprecated\. Use files\(\) instead\..*:DeprecationWarning"
    ),
]


def _patch_common_feature_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Installs deterministic librosa stubs shared by DSP tests."""
    monkeypatch.setattr(
        dsp.librosa,
        "power_to_db",
        lambda array, ref=None: np.asarray(array, dtype=np.float32),
    )
    monkeypatch.setattr(dsp.librosa.effects, "harmonic", lambda audio: audio)
    monkeypatch.setattr(
        dsp.librosa.feature,
        "tonnetz",
        lambda **_kwargs: np.ones((6, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        dsp.librosa.feature,
        "spectral_contrast",
        lambda **_kwargs: np.ones((7, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        dsp.librosa.feature,
        "melspectrogram",
        lambda **_kwargs: np.ones((128, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        dsp.librosa.feature,
        "mfcc",
        lambda **_kwargs: np.ones((40, 2), dtype=np.float32),
    )


def test_configure_feature_extraction_warning_filters_ignores_known_short_signal_warning() -> None:
    """Global helper should ignore the known non-actionable short-signal warning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        dsp.configure_feature_extraction_warning_filters()
        warnings.warn_explicit(
            "n_fft=512 is too large for input signal of length=93",
            category=UserWarning,
            filename="spectrum.py",
            lineno=1,
            module="librosa.core.spectrum",
        )

    assert captured == []


def test_extract_feature_from_signal_suppresses_known_librosa_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoped extraction should suppress the known librosa warning families."""
    _patch_common_feature_dependencies(monkeypatch)

    def _warn_stft(_audio: np.ndarray, *, n_fft: int) -> np.ndarray:
        warnings.warn_explicit(
            "n_fft=512 is too large for input signal of length=93",
            category=UserWarning,
            filename="spectrum.py",
            lineno=1,
            module="librosa.core.spectrum",
        )
        return np.ones((n_fft // 2 + 1, 2), dtype=np.float32)

    def _warn_chroma_stft(**_kwargs: object) -> np.ndarray:
        warnings.warn_explicit(
            "Trying to estimate tuning from empty frequency set.",
            category=UserWarning,
            filename="pitch.py",
            lineno=1,
            module="librosa.core.pitch",
        )
        return np.ones((12, 2), dtype=np.float32)

    monkeypatch.setattr(dsp.librosa, "stft", _warn_stft)
    monkeypatch.setattr(dsp.librosa.feature, "chroma_stft", _warn_chroma_stft)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        feature_vector = dsp.extract_feature_from_signal(
            np.ones(256, dtype=np.float32),
            sample_rate=16_000,
            feature_flags=FeatureFlags(
                mfcc=False,
                chroma=True,
                mel=False,
                contrast=False,
                tonnetz=False,
            ),
        )

    assert feature_vector.shape == (12,)
    assert captured == []


def test_extract_feature_from_signal_preserves_unexpected_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected warnings should remain visible to callers."""
    _patch_common_feature_dependencies(monkeypatch)
    monkeypatch.setattr(
        dsp.librosa,
        "stft",
        lambda _audio, *, n_fft: np.ones((n_fft // 2 + 1, 2), dtype=np.float32),
    )

    def _unexpected_chroma(**_kwargs: object) -> np.ndarray:
        warnings.warn_explicit(
            "unexpected feature warning",
            category=UserWarning,
            filename="pitch.py",
            lineno=1,
            module="librosa.core.pitch",
        )
        return np.ones((12, 2), dtype=np.float32)

    monkeypatch.setattr(dsp.librosa.feature, "chroma_stft", _unexpected_chroma)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        feature_vector = dsp.extract_feature_from_signal(
            np.ones(256, dtype=np.float32),
            sample_rate=16_000,
            feature_flags=FeatureFlags(
                mfcc=False,
                chroma=True,
                mel=False,
                contrast=False,
                tonnetz=False,
            ),
        )

    assert feature_vector.shape == (12,)
    assert [str(item.message) for item in captured] == ["unexpected feature warning"]
