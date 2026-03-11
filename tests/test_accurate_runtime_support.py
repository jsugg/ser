"""Tests for accurate runtime support adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

import ser.config as config
from ser.repr import EncodedSequence, PoolingWindow
from ser.runtime.accurate_runtime_support import (
    build_cpu_settings_snapshot,
    build_process_settings_snapshot,
    encode_accurate_sequence,
    prepare_accurate_backend_runtime,
)


class _BackendStub:
    """Structural feature-backend stub for support helper tests."""

    def __init__(self, *, encode_error: str | None = None) -> None:
        self._encode_error = encode_error
        self.prepared = False

    @property
    def backend_id(self) -> str:
        return "hf_whisper"

    @property
    def feature_dim(self) -> int:
        return 4

    def encode_sequence(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> EncodedSequence:
        del audio, sample_rate
        if self._encode_error is not None:
            raise RuntimeError(self._encode_error)
        return EncodedSequence(
            embeddings=np.ones((2, 4), dtype=np.float32),
            frame_start_seconds=np.asarray([0.0, 1.0], dtype=np.float64),
            frame_end_seconds=np.asarray([1.0, 2.0], dtype=np.float64),
            backend_id=self.backend_id,
        )

    def pool(
        self,
        encoded: EncodedSequence,
        windows: Sequence[PoolingWindow],
    ) -> NDArray[np.float64]:
        del encoded, windows
        return np.ones((1, 4), dtype=np.float64)

    def prepare_runtime(self) -> None:
        self.prepared = True


class _PrepareErrorBackendStub(_BackendStub):
    """Backend stub whose runtime preparation raises a configured error."""

    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    def prepare_runtime(self) -> None:
        raise RuntimeError(self._message)


def test_build_process_settings_snapshot_clones_emotions_mapping() -> None:
    """Support helper should return a process-safe settings snapshot."""
    settings = config.reload_settings()

    snapshot = build_process_settings_snapshot(settings)

    assert snapshot is not settings
    assert snapshot.emotions == settings.emotions
    assert snapshot.emotions is not settings.emotions
    assert snapshot.torch_runtime.device == settings.torch_runtime.device


def test_build_cpu_settings_snapshot_pins_torch_device_to_cpu() -> None:
    """CPU fallback snapshot helper should pin device selectors to CPU."""
    settings = config.reload_settings()

    snapshot = build_cpu_settings_snapshot(settings)

    assert snapshot is not settings
    assert snapshot.emotions == settings.emotions
    assert snapshot.emotions is not settings.emotions
    assert snapshot.torch_runtime.device == "cpu"


def test_encode_accurate_sequence_maps_runtime_errors() -> None:
    """Support helper should map dependency and transient runtime failures."""
    with pytest.raises(RuntimeError, match="requires optional dependencies"):
        encode_accurate_sequence(
            backend=_BackendStub(encode_error="backend requires optional dependencies"),
            audio=np.ones(4, dtype=np.float32),
            sample_rate=16_000,
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )

    with pytest.raises(ValueError, match="backend exploded"):
        encode_accurate_sequence(
            backend=_BackendStub(encode_error="backend exploded"),
            audio=np.ones(4, dtype=np.float32),
            sample_rate=16_000,
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )


def test_prepare_accurate_backend_runtime_handles_runtime_contracts() -> None:
    """Support helper should warm runtime and map setup-time failures."""
    backend = _BackendStub()

    prepare_accurate_backend_runtime(
        backend,
        dependency_error_factory=RuntimeError,
        transient_error_factory=ValueError,
    )

    assert backend.prepared is True

    with pytest.raises(RuntimeError, match="not installed"):
        prepare_accurate_backend_runtime(
            _PrepareErrorBackendStub("dependency not installed"),
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )

    with pytest.raises(ValueError, match="backend exploded"):
        prepare_accurate_backend_runtime(
            _PrepareErrorBackendStub("backend exploded"),
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )
