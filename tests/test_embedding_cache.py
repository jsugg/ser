"""Tests for medium embedding cache keying and invalidation behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ser.data.embedding_cache import EmbeddingCache
from ser.repr import EncodedSequence


def _encoded_sequence(scale: float = 1.0) -> EncodedSequence:
    """Builds a deterministic encoded sequence fixture."""
    return EncodedSequence(
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32) * scale,
        frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
        frame_end_seconds=np.asarray([1.0, 2.0], dtype=np.float64),
        backend_id="hf_xlsr",
    )


def test_embedding_cache_get_or_compute_uses_deterministic_key_and_hit(
    tmp_path: Path,
) -> None:
    """Second identical lookup should hit cache without recomputation."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"deterministic-audio")
    cache = EmbeddingCache(tmp_path / "cache")
    calls = {"count": 0}

    def compute() -> EncodedSequence:
        calls["count"] += 1
        return _encoded_sequence()

    first = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        compute=compute,
    )
    second = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        compute=compute,
    )

    assert calls["count"] == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.cache_key == second.cache_key
    assert first.cache_path == second.cache_path
    np.testing.assert_allclose(
        second.encoded.embeddings, _encoded_sequence().embeddings
    )


def test_embedding_cache_invalidates_corrupted_entry_and_recomputes(
    tmp_path: Path,
) -> None:
    """Corrupted cache payloads should be removed and recomputed safely."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"deterministic-audio")
    cache = EmbeddingCache(tmp_path / "cache")
    calls = {"count": 0}

    def compute() -> EncodedSequence:
        calls["count"] += 1
        return _encoded_sequence(scale=float(calls["count"]))

    first = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        compute=compute,
    )
    first.cache_path.write_text("corrupted-cache-entry", encoding="utf-8")

    second = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        compute=compute,
    )

    assert calls["count"] == 2
    assert second.cache_hit is False
    assert second.invalidated_stale_entry is True
    np.testing.assert_allclose(
        second.encoded.embeddings, _encoded_sequence(scale=2.0).embeddings
    )


def test_embedding_cache_key_changes_when_framing_changes(tmp_path: Path) -> None:
    """Framing config should be part of cache-key derivation."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"deterministic-audio")
    cache = EmbeddingCache(tmp_path / "cache")

    base = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        compute=_encoded_sequence,
    )
    shifted = cache.get_or_compute(
        audio_path=str(audio_path),
        backend_id="hf_xlsr",
        model_id="facebook/wav2vec2-xls-r-300m",
        frame_size_seconds=2.0,
        frame_stride_seconds=1.0,
        compute=_encoded_sequence,
    )

    assert base.cache_key != shifted.cache_key
    assert base.cache_path != shifted.cache_path
