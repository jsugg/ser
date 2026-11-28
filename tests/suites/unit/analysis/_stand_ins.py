"""Stand-in model files and generated audio for the utterance-analysis contract tests.

The classifiers are fitted on seeded random vectors so that a model file with the
current artifact envelope exists; their outputs carry no meaning.
"""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from ser._internal.models.artifact_envelope import build_model_artifact

LABELS: tuple[str, ...] = (
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)
CANONICAL_RATE_HZ = 16_000


class FailingClassifier(MLPClassifier):
    """Fits normally and fails at prediction time, to provoke ``model_error``."""

    def predict(self, X: object) -> NDArray[np.object_]:
        raise RuntimeError("stand-in prediction failure")

    def predict_proba(self, X: object) -> NDArray[np.float64]:
        raise RuntimeError("stand-in prediction failure")


class WindowClassifier(MLPClassifier):
    """Gives the first window a different class from all subsequent windows."""

    def predict_proba(self, X: object) -> NDArray[np.float64]:
        assert isinstance(X, np.ndarray)
        probabilities = np.zeros((len(X), 2), dtype=np.float64)
        probabilities[0, 0] = 1.0
        probabilities[1:, 1] = 1.0
        return probabilities


def write_model(
    path: Path,
    *,
    feature_size: int,
    labels: Sequence[str] = LABELS,
    seed: int = 0,
    profile: str = "fast",
    backend_id: str = "handcrafted",
    classifier_type: type[MLPClassifier] = MLPClassifier,
    model_revision: str | None = None,
) -> Path:
    """Writes one pickle model envelope like the ones `ser --train` produces."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(4 * len(labels), feature_size))
    targets = [labels[index % len(labels)] for index in range(features.shape[0])]
    classifier = classifier_type(hidden_layer_sizes=(8,), max_iter=50, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        classifier.fit(features, targets)
    artifact = build_model_artifact(
        classifier,
        feature_vector_size=feature_size,
        training_samples=features.shape[0],
        labels=list(labels),
        backend_id=backend_id,
        profile=profile,
        model_revision=model_revision,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    return path


def tone(
    seconds: float,
    *,
    sample_rate_hz: int = CANONICAL_RATE_HZ,
    rms_dbfs: float = -20.0,
    num_samples: int | None = None,
) -> NDArray[np.float32]:
    """Two tones below 3.5 kHz with a slow envelope, at a given RMS level."""
    count = round(seconds * sample_rate_hz) if num_samples is None else num_samples
    t = np.arange(count, dtype=np.float64) / sample_rate_hz
    signal = np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 1330.0 * t)
    signal *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    return _scale_to_rms(signal, rms_dbfs)


def sine(
    frequency_hz: float,
    seconds: float,
    *,
    sample_rate_hz: int = CANONICAL_RATE_HZ,
    rms_dbfs: float = -20.0,
) -> NDArray[np.float32]:
    """One pure tone at a given RMS level."""
    t = np.arange(round(seconds * sample_rate_hz), dtype=np.float64) / sample_rate_hz
    return _scale_to_rms(np.sin(2 * np.pi * frequency_hz * t), rms_dbfs)


def noise(
    seconds: float,
    *,
    sample_rate_hz: int = CANONICAL_RATE_HZ,
    rms_dbfs: float = -20.0,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Seeded white noise at a given RMS level."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(round(seconds * sample_rate_hz))
    return _scale_to_rms(signal, rms_dbfs)


def silence(seconds: float, *, sample_rate_hz: int = CANONICAL_RATE_HZ) -> NDArray[np.float32]:
    """Digital silence."""
    return np.zeros(round(seconds * sample_rate_hz), dtype=np.float32)


def write_wav(
    path: Path,
    samples: NDArray[np.float32],
    *,
    sample_rate_hz: int = CANONICAL_RATE_HZ,
    subtype: str = "FLOAT",
) -> Path:
    """Writes samples (frames x channels for multi-channel) to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate_hz, subtype=subtype, format="WAV")
    return path


def _scale_to_rms(signal: NDArray[np.float64], rms_dbfs: float) -> NDArray[np.float32]:
    rms = float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0
    if rms == 0.0:
        return signal.astype(np.float32)
    target = 10.0 ** (rms_dbfs / 20.0)
    scaled: NDArray[np.float64] = signal * (target / rms)
    return scaled.astype(np.float32)
