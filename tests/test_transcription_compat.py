"""Unit tests for transcription compatibility helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ser.utils.transcription_compat as compat


def test_resolve_stable_whisper_fallback_model_name_maps_distil_to_turbo() -> None:
    """Distil model ids should map to a stable-whisper compatible fallback."""
    assert (
        compat.resolve_stable_whisper_fallback_model_name("distil-large-v3") == "turbo"
    )


def test_resolve_stable_whisper_fallback_model_name_keeps_supported_model() -> None:
    """Already-supported stable-whisper model ids should pass through unchanged."""
    assert compat.resolve_stable_whisper_fallback_model_name("large-v3") == "large-v3"


def test_has_known_faster_whisper_openmp_runtime_conflict_detects_dual_libiomp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Conflict detection should trigger when ctranslate2 and torch OpenMP runtimes coexist."""
    ctranslate2_origin = tmp_path / "ctranslate2" / "__init__.py"
    torch_origin = tmp_path / "torchpkg" / "torch" / "__init__.py"
    faster_whisper_origin = tmp_path / "faster_whisper" / "__init__.py"
    ctranslate2_openmp = tmp_path / "ctranslate2" / ".dylibs" / "libiomp5.dylib"
    functorch_openmp = (
        tmp_path / "torchpkg" / "functorch" / ".dylibs" / "libiomp5.dylib"
    )
    ctranslate2_openmp.parent.mkdir(parents=True, exist_ok=True)
    functorch_openmp.parent.mkdir(parents=True, exist_ok=True)
    ctranslate2_origin.parent.mkdir(parents=True, exist_ok=True)
    torch_origin.parent.mkdir(parents=True, exist_ok=True)
    faster_whisper_origin.parent.mkdir(parents=True, exist_ok=True)
    ctranslate2_origin.write_text("", encoding="utf-8")
    torch_origin.write_text("", encoding="utf-8")
    faster_whisper_origin.write_text("", encoding="utf-8")
    ctranslate2_openmp.write_text("", encoding="utf-8")
    functorch_openmp.write_text("", encoding="utf-8")

    def _fake_find_spec(name: str) -> object | None:
        if name == "faster_whisper":
            return SimpleNamespace(origin=str(faster_whisper_origin))
        if name == "ctranslate2":
            return SimpleNamespace(origin=str(ctranslate2_origin))
        if name == "torch":
            return SimpleNamespace(origin=str(torch_origin))
        return None

    monkeypatch.setattr(compat.sys, "platform", "darwin")
    monkeypatch.setattr(compat.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(compat.importlib.util, "find_spec", _fake_find_spec)
    monkeypatch.delitem(compat.sys.modules, "faster_whisper", raising=False)

    assert compat.has_known_faster_whisper_openmp_runtime_conflict() is True


def test_stable_whisper_sparse_mps_incompatibility_detects_sparse_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SparseMPS NotImplementedError should mark stable-whisper MPS as incompatible."""

    class _FakeSparseTensor:
        def to(self, *, device: str) -> object:
            del device
            raise NotImplementedError(
                "Could not run 'aten::empty.memory_format' with arguments from the "
                "'SparseMPS' backend."
            )

    class _FakeDenseTensor:
        def to_sparse(self) -> _FakeSparseTensor:
            return _FakeSparseTensor()

    fake_torch = SimpleNamespace(
        float32=object(),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: True,
                is_built=lambda: True,
            )
        ),
        ones=lambda _shape, dtype=None: _FakeDenseTensor(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    compat.has_known_stable_whisper_sparse_mps_incompatibility.cache_clear()

    assert compat.has_known_stable_whisper_sparse_mps_incompatibility() is True

    compat.has_known_stable_whisper_sparse_mps_incompatibility.cache_clear()


def test_stable_whisper_sparse_mps_incompatibility_returns_false_when_mps_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MPS-unavailable environments should not report sparse-MPS incompatibility."""
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: False,
                is_built=lambda: True,
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    compat.has_known_stable_whisper_sparse_mps_incompatibility.cache_clear()

    assert compat.has_known_stable_whisper_sparse_mps_incompatibility() is False

    compat.has_known_stable_whisper_sparse_mps_incompatibility.cache_clear()
