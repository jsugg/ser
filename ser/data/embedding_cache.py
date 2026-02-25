"""Disk-backed cache for medium-profile encoded embedding sequences."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ser.repr import EncodedSequence
from ser.utils.logger import get_logger

logger: Logger = get_logger(__name__)

type EncodeSequenceCallable = Callable[[], EncodedSequence]


@dataclass(frozen=True)
class EmbeddingCacheEntry:
    """Cache lookup result for one audio/backend/model tuple."""

    encoded: EncodedSequence
    cache_key: str
    cache_path: Path
    cache_hit: bool
    invalidated_stale_entry: bool = False


class EmbeddingCache:
    """Caches encoded embedding sequences on disk for deterministic reuse."""

    def __init__(self, cache_dir: Path) -> None:
        """Initializes cache location.

        Args:
            cache_dir: Root directory used for cache entries.
        """
        self._cache_dir: Path = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        *,
        audio_path: str,
        backend_id: str,
        model_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
        compute: EncodeSequenceCallable,
    ) -> EmbeddingCacheEntry:
        """Returns cached encoded sequence or computes and stores it.

        Args:
            audio_path: Source audio file path.
            backend_id: Backend identifier (for example, ``hf_xlsr``).
            model_id: Model identifier/revision string.
            frame_size_seconds: Framing size used by caller.
            frame_stride_seconds: Framing stride used by caller.
            compute: Callback used to compute encoded sequence on cache miss.

        Returns:
            Cache result containing encoded sequence and cache-hit metadata.

        Raises:
            FileNotFoundError: If ``audio_path`` does not exist.
            ValueError: If key parameters are invalid or computed backend id
                does not match ``backend_id``.
        """
        if not backend_id.strip():
            raise ValueError("backend_id must be a non-empty string.")
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if frame_size_seconds <= 0.0:
            raise ValueError("frame_size_seconds must be greater than zero.")
        if frame_stride_seconds <= 0.0:
            raise ValueError("frame_stride_seconds must be greater than zero.")

        source_path: Path = Path(audio_path).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(
                f"Audio file not found for embedding cache: {audio_path}"
            )

        cache_key: str = self._cache_key(
            source_path=source_path,
            backend_id=backend_id,
            model_id=model_id,
            frame_size_seconds=frame_size_seconds,
            frame_stride_seconds=frame_stride_seconds,
        )
        cache_path: Path = self._cache_path(
            cache_key=cache_key, backend_id=backend_id, model_id=model_id
        )

        invalidated_stale_entry = False
        if cache_path.exists():
            try:
                encoded: EncodedSequence = self._read_cache_entry(cache_path)
            except Exception as err:
                invalidated_stale_entry = True
                logger.warning(
                    "Invalid embedding cache entry at %s; recomputing. Error: %s",
                    cache_path,
                    err,
                )
                cache_path.unlink(missing_ok=True)
            else:
                if encoded.backend_id == backend_id:
                    return EmbeddingCacheEntry(
                        encoded=encoded,
                        cache_key=cache_key,
                        cache_path=cache_path,
                        cache_hit=True,
                        invalidated_stale_entry=invalidated_stale_entry,
                    )
                invalidated_stale_entry = True
                logger.warning(
                    "Embedding cache backend mismatch for %s (expected=%s, got=%s); recomputing.",
                    cache_path,
                    backend_id,
                    encoded.backend_id,
                )
                cache_path.unlink(missing_ok=True)

        encoded = compute()
        if encoded.backend_id != backend_id:
            raise ValueError(
                "Computed embedding backend does not match cache key backend. "
                f"Expected {backend_id!r}, got {encoded.backend_id!r}."
            )
        self._write_cache_entry(cache_path, encoded)
        return EmbeddingCacheEntry(
            encoded=encoded,
            cache_key=cache_key,
            cache_path=cache_path,
            cache_hit=False,
            invalidated_stale_entry=invalidated_stale_entry,
        )

    def _cache_key(
        self,
        *,
        source_path: Path,
        backend_id: str,
        model_id: str,
        frame_size_seconds: float,
        frame_stride_seconds: float,
    ) -> str:
        """Builds deterministic key from source hash + backend/model/frame config."""
        payload: dict[str, Any] = {
            "audio_sha256": self._hash_file(source_path),
            "backend_id": backend_id,
            "model_id": model_id,
            "frame_size_seconds": round(frame_size_seconds, 6),
            "frame_stride_seconds": round(frame_stride_seconds, 6),
        }
        serialized: str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_path(self, *, cache_key: str, backend_id: str, model_id: str) -> Path:
        """Returns deterministic cache file path for one key."""
        model_segment: str = model_id.replace("/", "__").replace(" ", "_")
        return self._cache_dir / backend_id / model_segment / f"{cache_key}.npz"

    def _read_cache_entry(self, path: Path) -> EncodedSequence:
        """Reads and validates one cache entry."""
        with np.load(path, allow_pickle=False) as payload:
            embeddings: NDArray[np.float32] = np.asarray(
                payload["embeddings"], dtype=np.float32
            )
            frame_start_seconds: NDArray[np.float64] = np.asarray(
                payload["frame_start_seconds"], dtype=np.float64
            )
            frame_end_seconds: NDArray[np.float64] = np.asarray(
                payload["frame_end_seconds"], dtype=np.float64
            )
            raw_backend = np.asarray(payload["backend_id"])
        if raw_backend.ndim != 0:
            raise ValueError("Embedding cache backend_id payload is invalid.")
        backend_id = str(raw_backend.item())
        return EncodedSequence(
            embeddings=embeddings,
            frame_start_seconds=frame_start_seconds,
            frame_end_seconds=frame_end_seconds,
            backend_id=backend_id,
        )

    def _write_cache_entry(self, path: Path, encoded: EncodedSequence) -> None:
        """Atomically writes one cache entry to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path = path.with_suffix(".tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    embeddings=np.asarray(encoded.embeddings, dtype=np.float32),
                    frame_start_seconds=np.asarray(
                        encoded.frame_start_seconds,
                        dtype=np.float64,
                    ),
                    frame_end_seconds=np.asarray(
                        encoded.frame_end_seconds, dtype=np.float64
                    ),
                    backend_id=np.asarray(encoded.backend_id, dtype=np.str_),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def _hash_file(self, path: Path) -> str:
        """Returns sha256 for file content."""
        digest: hashlib._Hash = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
