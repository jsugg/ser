"""Contract tests for Whisper accurate-profile backend behavior."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

import ser.repr.hf_whisper as hf_whisper_module
from ser.repr import EncodedSequence, PoolingWindow, WhisperBackend
from ser.utils.torch_inference import TorchRuntime


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
        self.config = _FakeModelConfig(hidden_size=hidden_size)
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
        return _FakeEncoderOutput(
            last_hidden_state=np.expand_dims(base + offset, axis=0)
        )


class _NaNThenFiniteEncoder(_FakeEncoder):
    """Encoder stub that returns NaN output once, then finite output."""

    def __call__(self, **kwargs: object) -> _FakeEncoderOutput:
        output = super().__call__(**kwargs)
        if len(self.call_sizes) == 1:
            hidden = np.asarray(output.last_hidden_state, dtype=np.float32)
            hidden.fill(np.nan)
            return _FakeEncoderOutput(last_hidden_state=hidden)
        return output


class _FakeModel:
    """Deterministic whisper model stub exposing a nested encoder module."""

    def __init__(self, hidden_size: int) -> None:
        self.config = _FakeModelConfig(hidden_size=hidden_size)
        self.model = self
        self.encoder = _FakeEncoder(hidden_size=hidden_size)

    def eval(self) -> None:
        """No-op eval mode for protocol compatibility."""


class _FakeEncoderOnlyModel:
    """Deterministic encoder-only model stub with callable forward contract."""

    def __init__(self, hidden_size: int) -> None:
        self.config = _FakeModelConfig(hidden_size=hidden_size)
        self._encoder = _FakeEncoder(hidden_size=hidden_size)

    def __call__(self, **kwargs: object) -> _FakeEncoderOutput:
        return self._encoder(**kwargs)

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


def test_whisper_backend_accepts_callable_encoder_only_model() -> None:
    """Encoder-only callable models should be supported by encoder resolution."""
    backend = WhisperBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeEncoderOnlyModel(hidden_size=3),
    )

    encoded = backend.encode_sequence(np.arange(8, dtype=np.float32), sample_rate=4)

    assert encoded.embeddings.shape[1] == 3
    assert np.all(np.diff(encoded.frame_start_seconds) >= 0.0)


def test_whisper_backend_prefers_encoder_only_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime loader should select AutoModel backbone before seq2seq fallback."""
    calls = {"backbone": 0, "seq2seq": 0}

    class _FeatureExtractorLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeFeatureExtractor:
            del model_id, kwargs
            return _FakeFeatureExtractor()

    class _BackboneLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            del model_id, kwargs
            calls["backbone"] += 1
            return _FakeModel(hidden_size=9)

    class _Seq2SeqLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            del model_id, kwargs
            calls["seq2seq"] += 1
            return _FakeModel(hidden_size=11)

    class _TransformersModule:
        AutoFeatureExtractor = _FeatureExtractorLoader
        AutoModel = _BackboneLoader
        AutoModelForSpeechSeq2Seq = _Seq2SeqLoader

    def fake_import_module(module_name: str) -> object:
        if module_name == "transformers":
            return _TransformersModule()
        raise AssertionError(f"Unexpected import requested: {module_name!r}")

    monkeypatch.setattr(
        hf_whisper_module.importlib, "import_module", fake_import_module
    )
    monkeypatch.setattr(
        hf_whisper_module.WhisperBackend,
        "_ensure_dependencies_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: None,
    )

    backend = WhisperBackend(model_id="unit-test/whisper")
    assert backend.feature_dim == 9
    assert calls["backbone"] == 1
    assert calls["seq2seq"] == 0


def test_whisper_backend_falls_back_to_seq2seq_when_backbone_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime loader should fallback to seq2seq when AutoModel init fails."""
    calls = {"backbone": 0, "seq2seq": 0}

    class _FeatureExtractorLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeFeatureExtractor:
            del model_id, kwargs
            return _FakeFeatureExtractor()

    class _BackboneLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            del model_id, kwargs
            calls["backbone"] += 1
            raise RuntimeError("backbone loader unavailable")

    class _Seq2SeqLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            del model_id, kwargs
            calls["seq2seq"] += 1
            return _FakeModel(hidden_size=7)

    class _TransformersModule:
        AutoFeatureExtractor = _FeatureExtractorLoader
        AutoModel = _BackboneLoader
        AutoModelForSpeechSeq2Seq = _Seq2SeqLoader

    def fake_import_module(module_name: str) -> object:
        if module_name == "transformers":
            return _TransformersModule()
        raise AssertionError(f"Unexpected import requested: {module_name!r}")

    monkeypatch.setattr(
        hf_whisper_module.importlib, "import_module", fake_import_module
    )
    monkeypatch.setattr(
        hf_whisper_module.WhisperBackend,
        "_ensure_dependencies_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: None,
    )

    backend = WhisperBackend(model_id="unit-test/whisper")
    assert backend.feature_dim == 7
    assert calls["backbone"] == 1
    assert calls["seq2seq"] == 1


def test_whisper_backend_rejects_incomplete_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader should fail when transformers reports missing/mismatched keys."""
    calls = {"backbone": 0, "seq2seq": 0}

    class _FeatureExtractorLoader:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeFeatureExtractor:
            del model_id, kwargs
            return _FakeFeatureExtractor()

    class _BackboneLoader:
        @staticmethod
        def from_pretrained(
            model_id: str, **kwargs: object
        ) -> tuple[_FakeModel, dict[str, object]]:
            del model_id, kwargs
            calls["backbone"] += 1
            return _FakeModel(hidden_size=9), {
                "missing_keys": ["encoder.layers.0.self_attn.q_proj.weight"],
                "mismatched_keys": [],
            }

    class _Seq2SeqLoader:
        @staticmethod
        def from_pretrained(
            model_id: str, **kwargs: object
        ) -> tuple[_FakeModel, dict[str, object]]:
            del model_id, kwargs
            calls["seq2seq"] += 1
            return _FakeModel(hidden_size=11), {
                "missing_keys": [],
                "mismatched_keys": [("model.encoder.conv1.weight", (1,), (2,))],
            }

    class _TransformersModule:
        AutoFeatureExtractor = _FeatureExtractorLoader
        AutoModel = _BackboneLoader
        AutoModelForSpeechSeq2Seq = _Seq2SeqLoader

    def fake_import_module(module_name: str) -> object:
        if module_name == "transformers":
            return _TransformersModule()
        raise AssertionError(f"Unexpected import requested: {module_name!r}")

    monkeypatch.setattr(
        hf_whisper_module.importlib, "import_module", fake_import_module
    )
    monkeypatch.setattr(
        hf_whisper_module.WhisperBackend,
        "_ensure_dependencies_available",
        lambda self: None,
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: None,
    )

    backend = WhisperBackend(model_id="unit-test/whisper")
    with pytest.raises(RuntimeError, match="incomplete checkpoint load"):
        _ = backend.feature_dim
    assert calls["backbone"] == 1
    assert calls["seq2seq"] == 1


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


def test_whisper_backend_retries_non_finite_chunk_in_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend should rerun one chunk in float32 when half precision returns NaN/Inf."""
    fp16_runtime = TorchRuntime(
        device=object(),
        dtype=object(),
        device_spec="cuda:0",
        device_type="cuda",
        dtype_name="float16",
    )
    fp32_runtime = TorchRuntime(
        device=object(),
        dtype=object(),
        device_spec="cuda:0",
        device_type="cuda",
        dtype_name="float32",
    )
    model_move_dtypes: list[str] = []

    monkeypatch.setattr(
        hf_whisper_module,
        "maybe_resolve_torch_runtime",
        lambda *, device, dtype: fp16_runtime,
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "runtime_with_dtype",
        lambda runtime, *, dtype: fp32_runtime,
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "move_inputs_to_runtime",
        lambda inputs, runtime, *, dtype_keys: dict(inputs),
    )
    monkeypatch.setattr(
        hf_whisper_module,
        "move_model_to_runtime",
        lambda model, runtime: model_move_dtypes.append(runtime.dtype_name),
    )

    model = _FakeModel(hidden_size=2)
    model.encoder = _NaNThenFiniteEncoder(hidden_size=2)
    backend = WhisperBackend(
        model_id="unit-test/whisper",
        device="auto",
        dtype="auto",
        feature_extractor=_FakeFeatureExtractor(),
        model=model,
    )

    encoded = backend.encode_sequence(np.ones(4, dtype=np.float32), sample_rate=4)

    assert np.all(np.isfinite(encoded.embeddings))
    assert model_move_dtypes == ["float32"]
