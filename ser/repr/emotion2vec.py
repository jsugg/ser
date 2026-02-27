"""Emotion2Vec representation backend with encode-once/pool-many contracts."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Final, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ser.repr.backend import (
    EncodedSequence,
    FeatureMatrix,
    FeatureVector,
    PoolingWindow,
    overlap_frame_mask,
)
from ser.utils.logger import (
    DependencyLogPolicy,
    get_logger,
    scoped_dependency_log_policy,
)

logger = get_logger(__name__)
_NOISY_DEPENDENCY_POLICY: Final[DependencyLogPolicy] = DependencyLogPolicy(
    logger_prefixes=frozenset({"funasr", "modelscope"}),
    root_path_markers=frozenset(
        {
            "/site-packages/funasr/",
            "/site-packages/modelscope/",
            "/dist-packages/funasr/",
            "/dist-packages/modelscope/",
        }
    ),
)


class _FeatureExtractor(Protocol):
    """Runtime protocol for Hugging Face feature extractor callables."""

    def __call__(
        self,
        audio: NDArray[np.float32],
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
    ) -> Mapping[str, object]:
        """Produces model-ready tensors from raw audio."""
        ...


class _ModelConfig(Protocol):
    """Runtime protocol for model configuration metadata."""

    @property
    def hidden_size(self) -> int | None:
        """Returns hidden-state embedding size."""
        ...


class _ModelOutput(Protocol):
    """Runtime protocol for model forward outputs."""

    @property
    def last_hidden_state(self) -> object:
        """Returns hidden-state tensor-like output."""
        ...


class _EncoderModel(Protocol):
    """Runtime protocol for sequence encoder models."""

    @property
    def config(self) -> _ModelConfig:
        """Returns model configuration metadata."""
        ...

    def eval(self) -> object:
        """Switches model to eval mode."""
        ...

    def __call__(self, **kwargs: object) -> _ModelOutput:
        """Runs forward pass and returns hidden-state outputs."""
        ...


class _FunASRAutoModel(Protocol):
    """Runtime protocol for FunASR AutoModel inference wrapper."""

    model: object

    def generate(self, input: object, **cfg: object) -> object:
        """Runs inference and returns backend-specific output payload."""
        ...


class Emotion2VecBackend:
    """Emotion2Vec backend with bounded chunked encoding and deterministic pooling."""

    def __init__(
        self,
        *,
        model_id: str = "iic/emotion2vec_plus_large",
        hub: str | None = None,
        device: str = "cpu",
        max_chunk_seconds: float = 30.0,
        modelscope_cache_root: Path | None = None,
        huggingface_cache_root: Path | None = None,
        feature_extractor: _FeatureExtractor | None = None,
        model: _EncoderModel | None = None,
    ) -> None:
        """Initializes backend and optional injected test doubles.

        Args:
            model_id: Emotion2Vec model identifier on the selected hub.
            hub: Optional hub override (`ms`/`modelscope` or `hf`/`huggingface`).
            device: FunASR runtime device selector (`cpu`, `cuda`, or `cuda:N`).
            max_chunk_seconds: Maximum chunk duration for bounded-memory encoding.
            modelscope_cache_root: Optional ModelScope cache root for `hub=ms`.
            huggingface_cache_root: Optional Hugging Face cache root for `hub=hf`.
            feature_extractor: Optional injected extractor for deterministic tests.
            model: Optional injected model for deterministic tests.
        """
        if not model_id:
            raise ValueError("model_id must be a non-empty string.")
        if not np.isfinite(max_chunk_seconds) or max_chunk_seconds <= 0.0:
            raise ValueError("max_chunk_seconds must be greater than zero.")
        if (feature_extractor is None) ^ (model is None):
            raise ValueError(
                "feature_extractor and model must be provided together or omitted together."
            )
        normalized_device = device.strip().lower()
        if normalized_device != "cpu" and not normalized_device.startswith("cuda"):
            raise ValueError(
                "device must be one of 'cpu', 'cuda', or a 'cuda:N' selector."
            )
        self._model_id = model_id
        self._hub = self._resolve_hub(model_id=model_id, hub=hub)
        self._device = normalized_device
        self._max_chunk_seconds = max_chunk_seconds
        self._modelscope_cache_root = modelscope_cache_root
        self._huggingface_cache_root = huggingface_cache_root
        self._feature_extractor = feature_extractor
        self._model = model
        self._funasr_model: _FunASRAutoModel | None = None

    @property
    def backend_id(self) -> str:
        """Stable backend identifier used by runtime capability registry."""
        return "emotion2vec"

    @property
    def feature_dim(self) -> int:
        """Returns dynamic embedding size from model configuration."""
        if self._feature_extractor is not None and self._model is not None:
            _, model = self._ensure_runtime_components()
            hidden_size = getattr(model.config, "hidden_size", None)
            if not isinstance(hidden_size, int) or hidden_size <= 0:
                raise RuntimeError(
                    "Emotion2Vec model configuration is missing a valid positive hidden_size."
                )
            return hidden_size
        return self._read_funasr_embed_dim(self._ensure_funasr_model())

    def prepare_runtime(self) -> None:
        """Preloads runtime components before timed inference compute."""
        if self._feature_extractor is not None and self._model is not None:
            self._ensure_runtime_components()
            return
        self._ensure_funasr_model()

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        """Encodes audio to frame embeddings with explicit chunk-aware timestamps."""
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
                    "Emotion2Vec encoder output dimension does not match model hidden_size."
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
            raise RuntimeError("Emotion2Vec backend did not produce frame embeddings.")

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
        """Encodes a single chunk and normalizes output to 2D frame matrix."""
        if self._feature_extractor is None or self._model is None:
            return self._encode_chunk_with_funasr(audio_chunk, sample_rate)
        feature_extractor, model = self._ensure_runtime_components()
        inputs = feature_extractor(
            audio_chunk,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        with self._no_grad_context():
            outputs = model(**inputs)
        return self._normalize_hidden_state(outputs.last_hidden_state)

    def _encode_chunk_with_funasr(
        self,
        audio_chunk: NDArray[np.float32],
        sample_rate: int,
    ) -> NDArray[np.float32]:
        """Encodes one chunk via FunASR Emotion2Vec and normalizes output shape."""
        funasr_model = self._ensure_funasr_model()
        raw_result = funasr_model.generate(
            audio_chunk,
            fs=sample_rate,
            granularity="frame",
            extract_embedding=True,
        )
        if not isinstance(raw_result, list) or not raw_result:
            raise RuntimeError("Emotion2Vec backend returned no inference result.")
        first = raw_result[0]
        if not isinstance(first, Mapping):
            raise RuntimeError("Emotion2Vec backend result payload is not a mapping.")
        feats = first.get("feats")
        if feats is None:
            raise RuntimeError(
                "Emotion2Vec backend result is missing frame embeddings ('feats')."
            )
        embeddings = np.asarray(feats, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.ndim != 2:
            raise RuntimeError(
                "Emotion2Vec backend produced invalid embedding rank; expected 2D output."
            )
        if embeddings.shape[0] <= 0:
            raise RuntimeError("Emotion2Vec backend produced zero frame embeddings.")
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def _ensure_funasr_model(self) -> _FunASRAutoModel:
        """Loads FunASR AutoModel lazily for runtime Emotion2Vec extraction."""
        if self._funasr_model is not None:
            return self._funasr_model
        self._ensure_dependencies_available()
        self._configure_model_cache_environment()
        model_source, is_local_source = self._resolve_funasr_model_source()
        with self._suppress_third_party_info_logs():
            auto_model_module = importlib.import_module("funasr.auto.auto_model")
            auto_model_class = getattr(auto_model_module, "AutoModel", None)
            if auto_model_class is None:
                raise RuntimeError("funasr.auto.auto_model.AutoModel is unavailable.")

            def _init_auto_model(*, source: str, include_hub: bool) -> object:
                kwargs: dict[str, object] = {
                    "model": source,
                    "disable_update": True,
                    "disable_pbar": True,
                    "device": self._device,
                }
                if include_hub:
                    kwargs["hub"] = self._hub
                try:
                    return auto_model_class(**kwargs)
                except TypeError:
                    kwargs.pop("disable_pbar", None)
                    return auto_model_class(**kwargs)

            try:
                model = _init_auto_model(
                    source=model_source,
                    include_hub=not is_local_source,
                )
            except Exception as err:
                if is_local_source:
                    logger.warning(
                        "Emotion2Vec local cache initialization failed at %s; "
                        "falling back to hub lookup for model_id=%s.",
                        model_source,
                        self._model_id,
                    )
                    try:
                        model = _init_auto_model(
                            source=self._model_id,
                            include_hub=True,
                        )
                    except Exception as fallback_err:
                        raise RuntimeError(
                            "Failed to initialize Emotion2Vec backend via FunASR "
                            f"(model_id={self._model_id!r}, hub={self._hub!r}). "
                            "Ensure the model id exists on the configured hub and "
                            "authentication is configured when required."
                        ) from fallback_err
                else:
                    raise RuntimeError(
                        "Failed to initialize Emotion2Vec backend via FunASR "
                        f"(model_id={self._model_id!r}, hub={self._hub!r}). "
                        "Ensure the model id exists on the configured hub and authentication "
                        "is configured when required."
                    ) from err
        self._funasr_model = cast(_FunASRAutoModel, model)
        return self._funasr_model

    def _resolve_funasr_model_source(self) -> tuple[str, bool]:
        """Resolves local snapshot path when available, otherwise returns model id."""
        model_id_path = Path(self._model_id)
        if model_id_path.exists():
            return str(model_id_path), True

        if self._hub != "ms" or self._modelscope_cache_root is None:
            return self._model_id, False

        cache_root = self._ensure_cache_dir(self._modelscope_cache_root)
        model_relpath = Path(*self._model_id.split("/"))
        candidate_dirs: tuple[Path, ...]
        if cache_root.name == "hub":
            candidate_dirs = (cache_root / "models" / model_relpath,)
        else:
            candidate_dirs = (
                cache_root / "hub" / "models" / model_relpath,
                cache_root / "models" / model_relpath,
            )
        for model_dir in candidate_dirs:
            has_checkpoint = (model_dir / "model.pt").exists()
            has_config = (model_dir / "config.yaml").exists() or (
                model_dir / "configuration.json"
            ).exists()
            if has_checkpoint and has_config:
                return str(model_dir), True
        return self._model_id, False

    @contextmanager
    def _suppress_third_party_info_logs(self) -> Iterator[None]:
        """Demotes noisy dependency INFO logs and captures print noise."""
        if logger.getEffectiveLevel() <= logging.DEBUG:
            yield
            return

        with scoped_dependency_log_policy(
            policy=_NOISY_DEPENDENCY_POLICY,
            keep_demoted=False,
            capture_std_streams=True,
            suppressed_output_logger=logger,
        ):
            yield

    def _configure_model_cache_environment(self) -> None:
        """Maps configured SER model cache roots to backend-provider cache env vars."""
        if self._hub == "ms":
            if self._modelscope_cache_root is None:
                return
            cache_root = self._ensure_cache_dir(self._modelscope_cache_root)
            os.environ["MODELSCOPE_CACHE"] = str(cache_root)
            return

        if self._huggingface_cache_root is None:
            return
        hf_home = self._ensure_cache_dir(self._huggingface_cache_root)
        hub_cache = self._ensure_cache_dir(hf_home / "hub")
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hub_cache)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)

    @staticmethod
    def _ensure_cache_dir(path: Path) -> Path:
        """Creates a cache directory path if needed and returns its resolved value."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ensure_runtime_components(self) -> tuple[_FeatureExtractor, _EncoderModel]:
        """Loads runtime components lazily or returns injected test doubles."""
        if self._feature_extractor is None or self._model is None:
            raise RuntimeError(
                "Injected feature_extractor/model are required for direct tensor path."
            )
        return self._feature_extractor, self._model

    @staticmethod
    def _resolve_hub(
        *,
        model_id: str,
        hub: str | None,
    ) -> Literal["ms", "hf"]:
        """Normalizes hub selection for FunASR model loading."""
        if hub is not None:
            normalized = hub.strip().lower()
            if normalized in {"ms", "modelscope"}:
                return "ms"
            if normalized in {"hf", "huggingface"}:
                return "hf"
            raise ValueError("hub must be one of: ms, modelscope, hf, huggingface.")
        return "ms" if model_id.strip().lower().startswith("iic/") else "hf"

    def _ensure_dependencies_available(self) -> None:
        """Validates optional backend dependencies and raises actionable errors."""
        missing: list[str] = []
        for module_name in ("torch", "funasr", "modelscope"):
            if importlib.util.find_spec(module_name) is None:
                missing.append(module_name)
        if missing:
            modules = ", ".join(missing)
            raise RuntimeError(
                "Emotion2Vec backend requires optional dependencies that are not "
                f"installed: {modules}. Install accurate-research dependencies and retry."
            )

    def _read_funasr_embed_dim(self, model: _FunASRAutoModel) -> int:
        """Reads embedding dimension from loaded FunASR Emotion2Vec model config."""
        cfg = getattr(getattr(model, "model", None), "cfg", None)
        embed_dim_obj: object | None = None
        cfg_get = getattr(cfg, "get", None)
        if callable(cfg_get):
            embed_dim_obj = cfg_get("embed_dim")
        if embed_dim_obj is None and cfg is not None:
            embed_dim_obj = getattr(cfg, "embed_dim", None)
        if not isinstance(embed_dim_obj, int) or embed_dim_obj <= 0:
            raise RuntimeError(
                "Emotion2Vec model configuration is missing a valid positive embed_dim."
            )
        return embed_dim_obj

    def _no_grad_context(self) -> AbstractContextManager[object]:
        """Returns a torch.no_grad context when torch is available."""
        try:
            torch_module = importlib.import_module("torch")
        except ModuleNotFoundError:
            return nullcontext()
        no_grad = getattr(torch_module, "no_grad", None)
        if callable(no_grad):
            return cast(AbstractContextManager[object], no_grad())
        return nullcontext()

    def _normalize_hidden_state(self, hidden_state: object) -> NDArray[np.float32]:
        """Normalizes model hidden state output into `(frames, hidden)` matrix."""
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
                    "Emotion2Vec backend expects batch size 1 per chunk during encoding."
                )
            embeddings = embeddings[0]
        if embeddings.ndim != 2:
            raise RuntimeError(
                "Emotion2Vec backend produced invalid hidden-state rank; expected 2D "
                "or 3D output."
            )
        if embeddings.shape[0] <= 0:
            raise RuntimeError("Emotion2Vec backend produced zero frame embeddings.")
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
            raise RuntimeError(
                "Emotion2Vec backend cannot map timestamps for zero frames."
            )
        if not np.isfinite(chunk_duration_seconds) or chunk_duration_seconds <= 0.0:
            raise RuntimeError(
                "Emotion2Vec backend received non-positive chunk duration."
            )

        frame_duration = chunk_duration_seconds / float(frame_count)
        starts = chunk_start_seconds + (
            np.arange(frame_count, dtype=np.float64) * frame_duration
        )
        ends = starts + frame_duration
        ends[-1] = chunk_start_seconds + chunk_duration_seconds
        return starts, ends
