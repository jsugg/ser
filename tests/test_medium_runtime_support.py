"""Tests for medium runtime support adapters."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import ser.config as config
from ser.config import AppConfig
from ser.repr import EncodedSequence, XLSRBackend
from ser.runtime.medium_runtime_support import (
    build_cpu_medium_backend_for_settings,
    build_cpu_settings_snapshot,
    build_medium_backend_for_settings,
    build_runtime_settings_snapshot,
    encode_medium_sequence,
    validate_medium_loaded_model_runtime_contract,
)


@dataclass(frozen=True, slots=True)
class _LoadedModelStub:
    artifact_metadata: dict[str, object] | None


def _settings_stub() -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            models=SimpleNamespace(huggingface_cache_root=Path("/tmp/hf-cache")),
        ),
    )


def test_build_medium_backend_for_settings_uses_explicit_runtime_selectors() -> None:
    """Support helper should wire model id, cache root, and runtime selectors."""
    captured: dict[str, object] = {}

    class _BackendStub:
        def __init__(
            self,
            *,
            model_id: str,
            cache_dir: Path,
            device: str,
            dtype: str,
        ) -> None:
            captured["model_id"] = model_id
            captured["cache_dir"] = cache_dir
            captured["device"] = device
            captured["dtype"] = dtype

    backend = build_medium_backend_for_settings(
        settings=_settings_stub(),
        expected_backend_model_id="unit-test/xlsr-v1",
        runtime_device="mps",
        runtime_dtype="float16",
        backend_factory=cast(Callable[..., XLSRBackend], _BackendStub),
    )

    typed_backend = cast(_BackendStub, backend)

    assert isinstance(typed_backend, _BackendStub)
    assert captured == {
        "model_id": "unit-test/xlsr-v1",
        "cache_dir": Path("/tmp/hf-cache"),
        "device": "mps",
        "dtype": "float16",
    }


def test_build_runtime_settings_snapshot_clones_emotions_and_pins_selectors() -> None:
    """Support helper should clone settings and apply explicit runtime selectors."""
    settings = config.reload_settings()

    snapshot = build_runtime_settings_snapshot(
        settings,
        runtime_device="mps",
        runtime_dtype="float16",
    )

    assert snapshot is not settings
    assert snapshot.emotions == settings.emotions
    assert snapshot.emotions is not settings.emotions
    assert snapshot.torch_runtime.device == "mps"
    assert snapshot.torch_runtime.dtype == "float16"


def test_build_cpu_settings_snapshot_pins_cpu_float32_runtime() -> None:
    """CPU fallback helper should normalize runtime selectors for safe retries."""
    settings = config.reload_settings()

    snapshot = build_cpu_settings_snapshot(settings)

    assert snapshot is not settings
    assert snapshot.emotions == settings.emotions
    assert snapshot.emotions is not settings.emotions
    assert snapshot.torch_runtime.device == "cpu"
    assert snapshot.torch_runtime.dtype == "float32"


def test_build_cpu_medium_backend_for_settings_uses_cpu_defaults() -> None:
    """CPU fallback backend helper should force cpu/float32 selectors."""
    settings = config.reload_settings()
    captured: dict[str, object] = {}

    class _BackendStub:
        def __init__(
            self,
            *,
            model_id: str,
            cache_dir: Path,
            device: str,
            dtype: str,
        ) -> None:
            captured["model_id"] = model_id
            captured["cache_dir"] = cache_dir
            captured["device"] = device
            captured["dtype"] = dtype

    backend = build_cpu_medium_backend_for_settings(
        settings=settings,
        expected_backend_model_id="unit-test/xlsr-v1",
        backend_factory=cast(Callable[..., XLSRBackend], _BackendStub),
    )

    typed_backend = cast(_BackendStub, backend)

    assert isinstance(typed_backend, _BackendStub)
    assert captured == {
        "model_id": "unit-test/xlsr-v1",
        "cache_dir": settings.models.huggingface_cache_root,
        "device": "cpu",
        "dtype": "float32",
    }


def test_validate_medium_loaded_model_runtime_contract_warns_on_selector_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Support helper should validate metadata and emit selector-mismatch warnings."""
    loaded_model = _LoadedModelStub(
        artifact_metadata={
            "backend_id": "hf_xlsr",
            "profile": "medium",
            "backend_model_id": "unit-test/xlsr-v1",
            "torch_device": "cpu",
            "torch_dtype": "float32",
        }
    )

    with caplog.at_level(logging.WARNING):
        validate_medium_loaded_model_runtime_contract(
            loaded_model,
            expected_backend_model_id="unit-test/xlsr-v1",
            profile="medium",
            runtime_device="mps",
            runtime_dtype="float16",
            unavailable_error_factory=RuntimeError,
            logger=logging.getLogger("ser.tests.medium_runtime_support"),
        )

    assert "Artifact torch runtime selectors differ from current settings" in caplog.text


def test_encode_medium_sequence_maps_runtime_errors() -> None:
    """Support helper should map dependency and transient runtime failures."""

    class _BackendStub:
        def __init__(self, message: str) -> None:
            self._message = message

        def encode_sequence(
            self,
            audio: NDArray[np.float32],
            sample_rate: int,
        ) -> EncodedSequence:
            del audio, sample_rate
            raise RuntimeError(self._message)

    with pytest.raises(RuntimeError, match="dependency"):
        encode_medium_sequence(
            backend=_BackendStub("dependency missing"),
            audio=np.ones(4, dtype=np.float32),
            sample_rate=16_000,
            is_dependency_error=lambda err: "dependency" in str(err),
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )

    with pytest.raises(ValueError, match="backend exploded"):
        encode_medium_sequence(
            backend=_BackendStub("backend exploded"),
            audio=np.ones(4, dtype=np.float32),
            sample_rate=16_000,
            is_dependency_error=lambda err: "dependency" in str(err),
            dependency_error_factory=RuntimeError,
            transient_error_factory=ValueError,
        )
