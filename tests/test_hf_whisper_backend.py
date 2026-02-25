"""Contract tests for Whisper accurate-profile backend behavior."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from ser.repr import EncodedSequence, PoolingWindow, WhisperBackend


@dataclass(frozen=True)
class _FakeModelConfig:
    """Minimal model config stub exposing hidden-size metadata."""

    hidden_size: int | None = None
    d_model: int | None = None


@dataclass(frozen=True)
class _FakeEncoderOutput:
    """Minimal encoder output stub exposing hidden-state payload."""

    last_hidden_state: NDArray[np.float32]


class _FakeFeatureExtractor:
    """Deterministic extractor stub that returns chunk input for encoder calls."""

    def __init__(self) -> None:
        self.paddings: list[str | bool] = []

    def __call__(
        self,
        audio: NDArray[np.float32],
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: str | bool,
    ) -> dict[str, object]:
        del sampling_rate, return_tensors
        self.paddings.append(padding)
        return {"input_features": np.asarray(audio, dtype=np.float32)}


class _FakeEncoder:
    """Deterministic encoder stub producing chunk-size dependent frame outputs."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.call_sizes: list[int] = []

    def __call__(self, **kwargs: object) -> _FakeEncoderOutput:
        input_features = np.asarray(kwargs["input_features"], dtype=np.float32)
        self.call_sizes.append(int(input_features.size))
        frame_count = max(1, int(np.ceil(input_features.size / 4.0)))
        base = np.arange(
            frame_count * self.hidden_size,
            dtype=np.float32,
        ).reshape(frame_count, self.hidden_size)
        offset = float(len(self.call_sizes) - 1) * 100.0
        return _FakeEncoderOutput(last_hidden_state=np.expand_dims(base + offset, axis=0))


class _FakeModel:
    """Deterministic whisper model stub exposing a nested encoder module."""

    def __init__(self, hidden_size: int) -> None:
        self.config = _FakeModelConfig(hidden_size=hidden_size)
        self.model = self
        self.encoder = _FakeEncoder(hidden_size=hidden_size)

    def eval(self) -> None:
        """No-op eval mode for protocol compatibility."""


def test_whisper_backend_feature_dim_is_resolved_from_model_config() -> None:
    """feature_dim should come from config.hidden_size when present."""
    backend = WhisperBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeModel(hidden_size=11),
    )
    assert backend.backend_id == "hf_whisper"
    assert backend.feature_dim == 11


def test_whisper_backend_feature_dim_falls_back_to_d_model() -> None:
    """feature_dim should fall back to d_model when hidden_size is unavailable."""

    class _DModelOnlyModel(_FakeModel):
        def __init__(self, d_model: int) -> None:
            super().__init__(hidden_size=1)
            self.config = _FakeModelConfig(hidden_size=None, d_model=d_model)

    backend = WhisperBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_DModelOnlyModel(d_model=13),
    )
    assert backend.feature_dim == 13


def test_whisper_backend_encode_sequence_preserves_shape_and_chunk_timestamps() -> None:
    """Encoding should concatenate chunk outputs with monotonic frame timestamps."""
    backend = WhisperBackend(
        max_chunk_seconds=1.5,
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeModel(hidden_size=3),
    )
    audio = np.arange(12, dtype=np.float32)  # 3.0s at 4 Hz

    encoded = backend.encode_sequence(audio, sample_rate=4)

    assert encoded.embeddings.shape == (4, 3)
    np.testing.assert_allclose(
        encoded.frame_start_seconds,
        np.asarray([0.0, 0.75, 1.5, 2.25], dtype=np.float64),
    )
    np.testing.assert_allclose(
        encoded.frame_end_seconds,
        np.asarray([0.75, 1.5, 2.25, 3.0], dtype=np.float64),
    )
    assert encoded.backend_id == "hf_whisper"


def test_whisper_backend_pool_is_deterministic_for_overlap_windows() -> None:
    """Pooling should compute stable means over overlapping frame windows."""
    backend = WhisperBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeModel(hidden_size=2),
    )
    encoded = EncodedSequence(
        embeddings=np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        frame_start_seconds=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        backend_id=backend.backend_id,
    )

    pooled = backend.pool(
        encoded,
        [
            PoolingWindow(start_seconds=0.0, end_seconds=2.0),
            PoolingWindow(start_seconds=1.0, end_seconds=3.0),
        ],
    )

    np.testing.assert_allclose(
        pooled,
        np.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
    )


def test_whisper_backend_missing_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing torch/transformers should fail with explicit dependency message."""
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(module_name: str, package: str | None = None) -> object | None:
        if module_name in {"torch", "transformers"}:
            return None
        return original_find_spec(module_name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    backend = WhisperBackend(model_id="unit-test/whisper")
    with pytest.raises(RuntimeError, match="optional dependencies"):
        _ = backend.feature_dim


def test_whisper_backend_chunked_encoding_is_bounded_and_monotonic() -> None:
    """Long audio should be chunked while preserving monotonic global timestamps."""
    model = _FakeModel(hidden_size=4)
    feature_extractor = _FakeFeatureExtractor()
    backend = WhisperBackend(
        max_chunk_seconds=1.0,
        feature_extractor=feature_extractor,
        model=model,
    )
    audio = np.linspace(0.0, 1.0, 12, dtype=np.float32)  # 3 chunks at 4 Hz

    encoded = backend.encode_sequence(audio, sample_rate=4)

    assert model.encoder.call_sizes == [4, 4, 4]
    assert np.all(np.diff(encoded.frame_start_seconds) >= 0.0)
    assert np.all(np.diff(encoded.frame_end_seconds) >= 0.0)
    assert float(encoded.frame_start_seconds[0]) == pytest.approx(0.0)
    assert float(encoded.frame_end_seconds[-1]) == pytest.approx(3.0)
    assert np.all(encoded.frame_end_seconds > encoded.frame_start_seconds)
    assert feature_extractor.paddings == ["max_length", "max_length", "max_length"]


def test_whisper_backend_rejects_invalid_audio_contracts() -> None:
    """Encoder should enforce mono/positive-rate/non-empty input contracts."""
    backend = WhisperBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeModel(hidden_size=2),
    )
    with pytest.raises(ValueError, match="sample_rate"):
        backend.encode_sequence(np.ones(4, dtype=np.float32), sample_rate=0)
    with pytest.raises(ValueError, match="mono"):
        backend.encode_sequence(np.ones((2, 2), dtype=np.float32), sample_rate=4)
    with pytest.raises(ValueError, match="at least one sample"):
        backend.encode_sequence(np.asarray([], dtype=np.float32), sample_rate=4)
