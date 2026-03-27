"""Contract tests for feature runtime selector and cache-encoding helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from ser.data.embedding_cache import EmbeddingCacheEntry
from ser.models import feature_runtime_encoding as helpers
from ser.repr import EncodedSequence, PoolingWindow


class _RuntimePolicyStub:
    def __init__(self, *, device: str, dtype: str, reason: str) -> None:
        self.device = device
        self.dtype = dtype
        self.reason = reason


def _encoded_sequence(backend_id: str) -> EncodedSequence:
    return EncodedSequence(
        embeddings=np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
        frame_start_seconds=np.asarray([0.0, 0.5], dtype=np.float64),
        frame_end_seconds=np.asarray([0.5, 1.0], dtype=np.float64),
        backend_id=backend_id,
    )


def test_resolve_profile_runtime_selectors_uses_injected_policy() -> None:
    """Runtime selector helper should honor injected policy resolver output."""
    settings = cast(
        Any,
        SimpleNamespace(
            torch_runtime=SimpleNamespace(device="auto", dtype="auto"),
            feature_runtime_policy=SimpleNamespace(
                for_backend=lambda _backend_id: SimpleNamespace(
                    device="cuda",
                    dtype="float16",
                )
            ),
        ),
    )

    def _resolve_policy(**_kwargs: object) -> _RuntimePolicyStub:
        return _RuntimePolicyStub(
            device="cuda",
            dtype="float16",
            reason="unit_test",
        )

    device, dtype = helpers.resolve_profile_runtime_selectors(
        backend_id="hf_whisper",
        settings=settings,
        logger=logging.getLogger("tests.feature_runtime_encoding"),
        resolve_policy=_resolve_policy,
    )

    assert device == "cuda"
    assert dtype == "float16"


def test_resolve_accurate_runtime_config_rejects_unknown_backend() -> None:
    """Unknown accurate backend ids should fail fast."""
    settings = cast(
        Any,
        SimpleNamespace(
            accurate_runtime=SimpleNamespace(
                pool_window_size_seconds=2.0,
                pool_window_stride_seconds=0.5,
            ),
            accurate_research_runtime=SimpleNamespace(
                pool_window_size_seconds=2.5,
                pool_window_stride_seconds=0.75,
            ),
        ),
    )

    with pytest.raises(ValueError, match="Unknown accurate backend id"):
        _ = helpers.resolve_accurate_runtime_config(
            settings=settings,
            backend_id="unknown-backend",
            accurate_backend_id="hf_whisper",
            accurate_research_backend_id="emotion2vec",
        )


def test_encode_sequence_with_cache_uses_compute_callback() -> None:
    """Cache encoder should compute sequence and return cache entry payload."""
    captured: dict[str, object] = {}

    class _BackendStub:
        backend_id = "hf_xlsr"
        feature_dim = 2

        def encode_sequence(
            self,
            audio: np.ndarray[Any, np.dtype[np.float32]],
            sample_rate: int,
        ) -> EncodedSequence:
            captured["audio_dtype"] = audio.dtype
            captured["sample_rate"] = sample_rate
            return _encoded_sequence("hf_xlsr")

        def pool(
            self,
            encoded: EncodedSequence,
            windows: list[PoolingWindow],
        ) -> np.ndarray[Any, np.dtype[np.float64]]:
            del encoded, windows
            return np.asarray([[0.1, 0.2]], dtype=np.float64)

    class _CacheStub:
        def get_or_compute(self, **kwargs: object) -> EmbeddingCacheEntry:
            captured["cache_kwargs"] = kwargs
            compute = cast(Any, kwargs.get("compute"))
            assert callable(compute)
            encoded = cast(EncodedSequence, compute())
            return EmbeddingCacheEntry(
                encoded=encoded,
                cache_key="cache-key",
                cache_path=Path("cache-file.npz"),
                cache_hit=False,
            )

    def _read_audio(
        file_path: str,
        *,
        start_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], int]:
        captured["audio_path"] = file_path
        captured["start_seconds"] = start_seconds
        captured["duration_seconds"] = duration_seconds
        return np.asarray([0.1, 0.2], dtype=np.float32), 16_000

    encoded = helpers.encode_sequence_with_cache(
        audio_path="sample.wav",
        start_seconds=0.5,
        duration_seconds=1.0,
        backend=cast(Any, _BackendStub()),
        cache=cast(Any, _CacheStub()),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        log_prefix="Medium",
        logger=logging.getLogger("tests.feature_runtime_encoding"),
        read_audio=_read_audio,
    )

    assert encoded.backend_id == "hf_xlsr"
    assert captured["audio_path"] == "sample.wav"
    assert captured["start_seconds"] == 0.5
    assert captured["duration_seconds"] == 1.0
    assert captured["sample_rate"] == 16_000
    assert captured["audio_dtype"] == np.float32
