"""Whisper representation backend with encode-once/pool-many contracts."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ser.repr.backend import (
    EncodedSequence,
    FeatureMatrix,
    FeatureVector,
    PoolingWindow,
    overlap_frame_mask,
)
from ser.utils.logger import get_logger
from ser.utils.torch_inference import (
    TorchRuntime,
    inference_context,
    maybe_resolve_torch_runtime,
    move_inputs_to_runtime,
    move_model_to_runtime,
    runtime_with_dtype,
)

logger = get_logger(__name__)


class _FeatureExtractor(Protocol):
    """Runtime protocol for Hugging Face feature extractor callables."""

    def __call__(
        self,
        audio: NDArray[np.float32],
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: str | bool,
    ) -> Mapping[str, object]:
        """Produces model-ready tensors from raw audio."""
        ...


class _ModelConfig(Protocol):
    """Runtime protocol for model configuration metadata."""

    @property
    def hidden_size(self) -> int | None:
        """Returns hidden-state embedding size when available."""
        ...

    @property
    def d_model(self) -> int | None:
        """Returns model dimension used by some Whisper variants."""
        ...


class _EncoderOutput(Protocol):
    """Runtime protocol for encoder forward outputs."""

    @property
    def last_hidden_state(self) -> object:
        """Returns hidden-state tensor-like output."""
        ...


class _EncoderModule(Protocol):
    """Runtime protocol for encoder modules."""

    def __call__(self, **kwargs: object) -> _EncoderOutput:
        """Runs encoder forward pass and returns hidden states."""
        ...


class _EncoderModel(Protocol):
    """Runtime protocol for whisper sequence encoder models."""

    @property
    def config(self) -> _ModelConfig:
        """Returns model configuration metadata."""
        ...

    def eval(self) -> object:
        """Switches model to eval mode."""
        ...


class WhisperBackend:
    """Whisper backend with bounded chunked encoding and deterministic pooling."""

    def __init__(
        self,
        *,
        model_id: str = "openai/whisper-large-v3",
        max_chunk_seconds: float = 30.0,
        cache_dir: Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        feature_extractor: _FeatureExtractor | None = None,
        model: _EncoderModel | None = None,
    ) -> None:
        """Initializes backend and optional injected test doubles.

        Args:
            model_id: Hugging Face model id for Whisper backbone loading.
            max_chunk_seconds: Maximum chunk duration for bounded-memory encoding.
            cache_dir: Optional Hugging Face cache root for model/processor downloads.
            feature_extractor: Optional injected feature extractor for deterministic tests.
            model: Optional injected model for deterministic tests.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if not model_id:
            raise ValueError("model_id must be a non-empty string.")
        if not np.isfinite(max_chunk_seconds) or max_chunk_seconds <= 0.0:
            raise ValueError("max_chunk_seconds must be greater than zero.")
        if (feature_extractor is None) ^ (model is None):
            raise ValueError(
                "feature_extractor and model must be provided together or omitted together."
            )
        self._model_id = model_id
        self._max_chunk_seconds = max_chunk_seconds
        self._cache_dir = cache_dir
        self._device = device
        self._dtype = dtype
        self._feature_extractor = feature_extractor
        self._model = model
        self._torch_runtime: TorchRuntime | None = None

    @property
    def backend_id(self) -> str:
        """Stable backend identifier used by runtime capability registry."""
        return "hf_whisper"

    @property
    def feature_dim(self) -> int:
        """Returns dynamic embedding size from model configuration."""
        _, model = self._ensure_runtime_components()
        hidden_size = self._resolve_hidden_size(model.config)
        if hidden_size <= 0:
            raise RuntimeError(
                "Whisper model configuration is missing a valid positive hidden size."
            )
        return hidden_size

    def prepare_runtime(self) -> None:
        """Preloads model runtime components before timed inference compute."""
        self._ensure_runtime_components()

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        """Encodes audio to frame embeddings with explicit chunk-aware timestamps.

        Args:
            audio: Mono waveform samples.
            sample_rate: Sample rate in Hz.

        Returns:
            Encoded sequence with frame embeddings and timestamp arrays.

        Raises:
            ValueError: If input invariants are violated.
            RuntimeError: If dependencies or model outputs are invalid.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if audio.ndim != 1:
            raise ValueError("audio must be mono (1D array).")
        if audio.size == 0:
            raise ValueError("audio must contain at least one sample.")

        normalized_audio = np.asarray(audio, dtype=np.float32)
        chunk_size_samples = max(
            1,
            int(round(self._max_chunk_seconds * float(sample_rate))),
        )

        chunk_embeddings: list[NDArray[np.float32]] = []
        chunk_starts: list[NDArray[np.float64]] = []
        chunk_ends: list[NDArray[np.float64]] = []

        for start_index in range(0, int(normalized_audio.size), chunk_size_samples):
            end_index = min(
                start_index + chunk_size_samples, int(normalized_audio.size)
            )
            audio_chunk = normalized_audio[start_index:end_index]
            if audio_chunk.size == 0:
                continue

            embeddings = self._encode_chunk(audio_chunk, sample_rate)
            if embeddings.shape[1] != self.feature_dim:
                raise RuntimeError(
                    "Whisper encoder output dimension does not match model hidden size."
                )

            chunk_start_seconds = float(start_index) / float(sample_rate)
            chunk_duration_seconds = float(end_index - start_index) / float(sample_rate)
            starts, ends = self._build_chunk_timestamps(
                chunk_start_seconds=chunk_start_seconds,
                chunk_duration_seconds=chunk_duration_seconds,
                frame_count=int(embeddings.shape[0]),
            )
            chunk_embeddings.append(embeddings)
            chunk_starts.append(starts)
            chunk_ends.append(ends)

        if not chunk_embeddings:
            raise RuntimeError("Whisper backend did not produce frame embeddings.")

        return EncodedSequence(
            embeddings=np.vstack(chunk_embeddings).astype(np.float32, copy=False),
            frame_start_seconds=np.concatenate(chunk_starts).astype(
                np.float64, copy=False
            ),
            frame_end_seconds=np.concatenate(chunk_ends).astype(np.float64, copy=False),
            backend_id=self.backend_id,
        )

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> FeatureMatrix:
        """Mean-pools frame embeddings for every overlapping pooling window."""
        if not windows:
            return np.empty((0, encoded.embeddings.shape[1]), dtype=np.float64)

        pooled_rows: list[FeatureVector] = []
        for window in windows:
            mask = overlap_frame_mask(encoded, window)
            pooled_rows.append(
                np.asarray(encoded.embeddings[mask].mean(axis=0), dtype=np.float64)
            )
        return np.vstack(pooled_rows).astype(np.float64, copy=False)

    def _encode_chunk(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Encodes one chunk and normalizes output to a 2D frame matrix."""
        feature_extractor, model = self._ensure_runtime_components()
        inputs = feature_extractor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            # Whisper encoders expect fixed-length mel features; max-length padding
            # keeps short/final chunks compatible with encoder forward contracts.
            padding="max_length",
        )
        runtime = self._get_torch_runtime()
        if runtime is not None:
            inputs = move_inputs_to_runtime(
                inputs,
                runtime,
                dtype_keys=frozenset({"input_features"}),
            )
        encoder = self._resolve_encoder(model)
        embeddings = self._encode_with_encoder(encoder=encoder, inputs=inputs)
        if runtime is None or runtime.dtype_name == "float32":
            return embeddings
        if np.all(np.isfinite(embeddings)):
            return embeddings

        fallback_runtime = runtime_with_dtype(runtime, dtype="float32")
        if fallback_runtime is None:
            raise RuntimeError(
                "Whisper backend produced non-finite embeddings and torch is "
                "unavailable for float32 fallback."
            )
        logger.warning(
            "Whisper backend produced non-finite embeddings with dtype=%s. "
            "Retrying chunk with float32 fallback.",
            runtime.dtype_name,
        )
        move_model_to_runtime(model, fallback_runtime)
        self._torch_runtime = fallback_runtime
        fallback_inputs = move_inputs_to_runtime(
            inputs,
            fallback_runtime,
            dtype_keys=frozenset({"input_features"}),
        )
        fallback_embeddings = self._encode_with_encoder(
            encoder=encoder,
            inputs=fallback_inputs,
        )
        if not np.all(np.isfinite(fallback_embeddings)):
            raise RuntimeError(
                "Whisper backend produced non-finite embeddings after float32 fallback."
            )
        return fallback_embeddings

    def _ensure_runtime_components(self) -> tuple[_FeatureExtractor, _EncoderModel]:
        """Loads runtime components lazily or returns injected test doubles."""
        if self._feature_extractor is not None and self._model is not None:
            return self._feature_extractor, self._model

        self._ensure_dependencies_available()
        transformers_module = importlib.import_module("transformers")
        auto_feature_extractor = getattr(
            transformers_module, "AutoFeatureExtractor", None
        )
        auto_model = getattr(transformers_module, "AutoModelForSpeechSeq2Seq", None)
        if auto_feature_extractor is None or auto_model is None:
            raise RuntimeError(
                "transformers package does not expose AutoFeatureExtractor/"
                "AutoModelForSpeechSeq2Seq."
            )

        cache_kwargs: dict[str, str] = {}
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_kwargs["cache_dir"] = str(self._cache_dir)

        feature_extractor = cast(
            _FeatureExtractor,
            auto_feature_extractor.from_pretrained(self._model_id, **cache_kwargs),
        )
        try:
            model = cast(
                _EncoderModel,
                auto_model.from_pretrained(
                    self._model_id,
                    use_safetensors=True,
                    **cache_kwargs,
                ),
            )
        except TypeError:
            model = cast(
                _EncoderModel,
                auto_model.from_pretrained(self._model_id, **cache_kwargs),
            )
        except Exception as err:
            raise RuntimeError(
                "Failed to load Whisper model with safetensors-only policy. "
                "Use a model revision that publishes safetensors weights, or "
                "upgrade torch to >=2.6 for legacy checkpoint loading."
            ) from err
        model.eval()
        runtime = self._get_torch_runtime()
        if runtime is not None:
            move_model_to_runtime(model, runtime)
        self._feature_extractor = feature_extractor
        self._model = model
        return feature_extractor, model

    def _get_torch_runtime(self) -> TorchRuntime | None:
        """Lazily resolves optional torch runtime selectors."""
        if self._torch_runtime is None:
            self._torch_runtime = maybe_resolve_torch_runtime(
                device=self._device,
                dtype=self._dtype,
            )
        return self._torch_runtime

    def _resolve_hidden_size(self, config: _ModelConfig) -> int:
        """Resolves hidden size from either hidden_size or d_model config fields."""
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size
        d_model = getattr(config, "d_model", None)
        if isinstance(d_model, int) and d_model > 0:
            return d_model
        return 0

    def _resolve_encoder(self, model: _EncoderModel) -> _EncoderModule:
        """Resolves encoder module across Whisper model wrapper variants."""
        direct_encoder = getattr(model, "encoder", None)
        if callable(direct_encoder):
            return cast(_EncoderModule, direct_encoder)

        nested_model = getattr(model, "model", None)
        nested_encoder = getattr(nested_model, "encoder", None)
        if callable(nested_encoder):
            return cast(_EncoderModule, nested_encoder)

        raise RuntimeError(
            "Whisper model does not expose an encoder module for sequence encoding."
        )

    def _ensure_dependencies_available(self) -> None:
        """Validates optional backend dependencies and raises actionable errors."""
        missing: list[str] = []
        for module_name in ("torch", "transformers"):
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
        if missing:
            modules = ", ".join(missing)
            raise RuntimeError(
                "Whisper backend requires optional dependencies that are not installed: "
                f"{modules}. Install accurate-profile dependencies and retry."
            )

    def _no_grad_context(self) -> AbstractContextManager[object]:
        """Returns best-available inference context when torch is present."""
        return inference_context()

    def _encode_with_encoder(
        self,
        *,
        encoder: _EncoderModule,
        inputs: Mapping[str, object],
    ) -> NDArray[np.float32]:
        """Runs encoder forward pass and normalizes hidden-state outputs."""
        with self._no_grad_context():
            outputs = encoder(**inputs)
        return self._normalize_hidden_state(outputs.last_hidden_state)

    def _normalize_hidden_state(self, hidden_state: object) -> NDArray[np.float32]:
        """Normalizes hidden-state outputs into `(frames, hidden)` matrix."""
        current = hidden_state
        detach = getattr(current, "detach", None)
        if callable(detach):
            current = detach()
        cpu = getattr(current, "cpu", None)
        if callable(cpu):
            current = cpu()
        to_numpy = getattr(current, "numpy", None)
        if callable(to_numpy):
            current = to_numpy()

        embeddings = np.asarray(current, dtype=np.float32)
        if embeddings.ndim == 3:
            if embeddings.shape[0] != 1:
                raise RuntimeError(
                    "Whisper backend expects batch size 1 per chunk during encoding."
                )
            embeddings = embeddings[0]
        if embeddings.ndim != 2:
            raise RuntimeError(
                "Whisper backend produced invalid hidden-state rank; expected 2D "
                "or 3D output."
            )
        if embeddings.shape[0] <= 0:
            raise RuntimeError("Whisper backend produced zero frame embeddings.")
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def _build_chunk_timestamps(
        self,
        *,
        chunk_start_seconds: float,
        chunk_duration_seconds: float,
        frame_count: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Builds monotonic frame start/end timestamps for one encoded chunk."""
        if frame_count <= 0:
            raise RuntimeError("Whisper backend cannot map timestamps for zero frames.")
        if not np.isfinite(chunk_duration_seconds) or chunk_duration_seconds <= 0.0:
            raise RuntimeError("Whisper backend received non-positive chunk duration.")

        frame_duration = chunk_duration_seconds / float(frame_count)
        starts = chunk_start_seconds + (
            np.arange(frame_count, dtype=np.float64) * frame_duration
        )
        ends = starts + frame_duration
        ends[-1] = chunk_start_seconds + chunk_duration_seconds
        return starts, ends
