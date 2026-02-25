"""Contract tests for Emotion2Vec backend behavior."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from ser.repr import Emotion2VecBackend
from ser.utils.logger import (
    DependencyLogFilter,
    DependencyLogPolicy,
    is_dependency_log_record,
    scoped_dependency_log_policy,
)


@dataclass(frozen=True)
class _FakeModelConfig:
    """Minimal model config stub exposing hidden-size metadata."""

    hidden_size: int


@dataclass(frozen=True)
class _FakeModelOutput:
    """Minimal model output stub exposing hidden-state payload."""

    last_hidden_state: NDArray[np.float32]


class _FakeFeatureExtractor:
    """Deterministic extractor stub that returns chunk input for model calls."""

    def __call__(
        self,
        audio: NDArray[np.float32],
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, object]:
        del sampling_rate, return_tensors, padding
        return {"input_values": np.asarray(audio, dtype=np.float32)}


class _FakeModel:
    """Deterministic model stub producing chunk-size dependent frame outputs."""

    def __init__(self, hidden_size: int) -> None:
        self.config = _FakeModelConfig(hidden_size=hidden_size)
        self.call_sizes: list[int] = []

    def eval(self) -> None:
        """No-op eval mode for protocol compatibility."""

    def __call__(self, **kwargs: object) -> _FakeModelOutput:
        input_values = np.asarray(kwargs["input_values"], dtype=np.float32)
        self.call_sizes.append(int(input_values.size))
        frame_count = max(1, int(np.ceil(input_values.size / 4.0)))
        base = np.arange(
            frame_count * int(self.config.hidden_size),
            dtype=np.float32,
        ).reshape(frame_count, int(self.config.hidden_size))
        return _FakeModelOutput(last_hidden_state=np.expand_dims(base, axis=0))


def test_emotion2vec_backend_feature_dim_is_resolved_from_model_config() -> None:
    """feature_dim should be read from model hidden-size metadata."""
    backend = Emotion2VecBackend(
        feature_extractor=_FakeFeatureExtractor(),
        model=_FakeModel(hidden_size=9),
    )
    assert backend.backend_id == "emotion2vec"
    assert backend.feature_dim == 9


def test_emotion2vec_backend_encode_sequence_preserves_chunk_timestamps() -> None:
    """Encoding should concatenate chunk outputs with monotonic timestamps."""
    model = _FakeModel(hidden_size=3)
    backend = Emotion2VecBackend(
        max_chunk_seconds=1.5,
        feature_extractor=_FakeFeatureExtractor(),
        model=model,
    )
    audio = np.arange(12, dtype=np.float32)  # 3.0s at 4 Hz

    encoded = backend.encode_sequence(audio, sample_rate=4)

    assert encoded.backend_id == "emotion2vec"
    assert encoded.embeddings.shape == (4, 3)
    np.testing.assert_allclose(
        encoded.frame_start_seconds,
        np.asarray([0.0, 0.75, 1.5, 2.25], dtype=np.float64),
    )
    np.testing.assert_allclose(
        encoded.frame_end_seconds,
        np.asarray([0.75, 1.5, 2.25, 3.0], dtype=np.float64),
    )
    assert model.call_sizes == [6, 6]


def test_emotion2vec_backend_missing_dependency_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing runtime deps should fail with explicit dependency message."""
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(module_name: str, package: str | None = None) -> object | None:
        if module_name in {"torch", "funasr", "modelscope"}:
            return None
        return original_find_spec(module_name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    backend = Emotion2VecBackend(model_id="unit-test/e2v")

    with pytest.raises(RuntimeError, match="optional dependencies"):
        _ = backend.feature_dim


def test_emotion2vec_backend_configures_modelscope_cache_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ModelScope hub mode should map MODELSCOPE_CACHE to configured cache root."""
    cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    backend = Emotion2VecBackend(
        model_id="iic/emotion2vec_plus_large",
        modelscope_cache_root=cache_root,
    )
    monkeypatch.setenv("MODELSCOPE_CACHE", "/tmp/legacy-cache")

    backend._configure_model_cache_environment()

    assert backend._resolve_hub(model_id="iic/emotion2vec_plus_large", hub=None) == "ms"
    assert os.environ["MODELSCOPE_CACHE"] == str(cache_root)
    assert cache_root.is_dir()


def test_emotion2vec_backend_configures_huggingface_cache_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """HuggingFace hub mode should map HF cache variables to configured roots."""
    hf_root = tmp_path / "model-cache" / "huggingface"
    backend = Emotion2VecBackend(
        model_id="unit-test/emotion2vec-hf",
        huggingface_cache_root=hf_root,
    )
    monkeypatch.setenv("HF_HOME", "/tmp/hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/hf-hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/tmp/hf-hub-compat")

    backend._configure_model_cache_environment()

    assert backend._resolve_hub(model_id="unit-test/emotion2vec-hf", hub=None) == "hf"
    assert os.environ["HF_HOME"] == str(hf_root)
    assert os.environ["HF_HUB_CACHE"] == str(hf_root / "hub")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(hf_root / "hub")
    assert hf_root.is_dir()
    assert (hf_root / "hub").is_dir()


def test_emotion2vec_backend_resolves_modelscope_cached_snapshot(
    tmp_path: Path,
) -> None:
    """ModelScope cache snapshot should be preferred when fully available."""
    cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    model_dir = cache_root / "models" / "iic" / "emotion2vec_plus_large"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("model: UnitModel\n", encoding="utf-8")
    backend = Emotion2VecBackend(
        model_id="iic/emotion2vec_plus_large",
        modelscope_cache_root=cache_root,
    )

    source, is_local = backend._resolve_funasr_model_source()

    assert source == str(model_dir)
    assert is_local is True


def test_emotion2vec_backend_resolves_modelscope_snapshot_from_non_hub_root(
    tmp_path: Path,
) -> None:
    """ModelScope snapshot resolution should support roots above the `hub` dir."""
    cache_root = tmp_path / "model-cache" / "modelscope"
    model_dir = cache_root / "hub" / "models" / "iic" / "emotion2vec_plus_large"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    (model_dir / "configuration.json").write_text("{}", encoding="utf-8")
    backend = Emotion2VecBackend(
        model_id="iic/emotion2vec_plus_large",
        modelscope_cache_root=cache_root,
    )

    source, is_local = backend._resolve_funasr_model_source()

    assert source == str(model_dir)
    assert is_local is True


def test_emotion2vec_backend_uses_local_snapshot_without_hub_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local cache snapshots should be loaded directly without hub parameter."""
    cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    model_dir = cache_root / "models" / "iic" / "emotion2vec_plus_large"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("model: UnitModel\n", encoding="utf-8")

    original_find_spec = importlib.util.find_spec
    original_import_module = importlib.import_module

    def fake_find_spec(module_name: str, package: str | None = None) -> object | None:
        if module_name in {"torch", "funasr", "modelscope"}:
            return object()
        return original_find_spec(module_name, package)

    captured_calls: list[dict[str, object]] = []

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            captured_calls.append(dict(kwargs))
            self.model = SimpleNamespace(cfg={"embed_dim": 768})

    def fake_import_module(module_name: str) -> object:
        if module_name == "funasr.auto.auto_model":
            return SimpleNamespace(AutoModel=FakeAutoModel)
        return original_import_module(module_name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    backend = Emotion2VecBackend(
        model_id="iic/emotion2vec_plus_large",
        modelscope_cache_root=cache_root,
    )
    backend._ensure_funasr_model()

    assert captured_calls
    assert captured_calls[0]["model"] == str(model_dir)
    assert "hub" not in captured_calls[0]


def test_emotion2vec_backend_falls_back_to_hub_when_local_snapshot_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backend should fall back to hub lookup if local snapshot init fails."""
    cache_root = tmp_path / "model-cache" / "modelscope" / "hub"
    model_dir = cache_root / "models" / "iic" / "emotion2vec_plus_large"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"checkpoint")
    (model_dir / "config.yaml").write_text("model: UnitModel\n", encoding="utf-8")

    original_find_spec = importlib.util.find_spec
    original_import_module = importlib.import_module

    def fake_find_spec(module_name: str, package: str | None = None) -> object | None:
        if module_name in {"torch", "funasr", "modelscope"}:
            return object()
        return original_find_spec(module_name, package)

    captured_calls: list[dict[str, object]] = []

    class FakeAutoModel:
        def __init__(self, **kwargs: object) -> None:
            captured_calls.append(dict(kwargs))
            if kwargs.get("model") == str(model_dir):
                raise RuntimeError("corrupted snapshot")
            self.model = SimpleNamespace(cfg={"embed_dim": 768})

    def fake_import_module(module_name: str) -> object:
        if module_name == "funasr.auto.auto_model":
            return SimpleNamespace(AutoModel=FakeAutoModel)
        return original_import_module(module_name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    backend = Emotion2VecBackend(
        model_id="iic/emotion2vec_plus_large",
        modelscope_cache_root=cache_root,
    )
    backend._ensure_funasr_model()

    assert len(captured_calls) == 2
    assert captured_calls[0]["model"] == str(model_dir)
    assert captured_calls[1]["model"] == "iic/emotion2vec_plus_large"
    assert captured_calls[1]["hub"] == "ms"


def test_emotion2vec_backend_suppresses_dependency_noise_outside_debug(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Known dependency INFO lines should be demoted at INFO-level runs."""
    backend = Emotion2VecBackend(model_id="iic/emotion2vec_plus_large")
    module_logger = logging.getLogger("ser.repr.emotion2vec")
    original_level = module_logger.level
    module_logger.setLevel(logging.INFO)
    try:
        with caplog.at_level(logging.INFO):
            with backend._suppress_third_party_info_logs():
                logging.getLogger("funasr.auto.auto_model").info(
                    "download models from model hub: ms"
                )
                logging.getLogger("funasr.auto.auto_model").warning(
                    "trust_remote_code: False"
                )
                logging.getLogger("other.module").info("unrelated info")
                print("funasr version: 1.3.1.")
    finally:
        module_logger.setLevel(original_level)

    assert "download models from model hub: ms" not in caplog.text
    assert "trust_remote_code: False" in caplog.text
    assert "unrelated info" in caplog.text
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_emotion2vec_backend_keeps_dependency_noise_in_debug(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """DEBUG-level runs should keep third-party diagnostics visible."""
    backend = Emotion2VecBackend(model_id="iic/emotion2vec_plus_large")
    module_logger = logging.getLogger("ser.repr.emotion2vec")
    original_level = module_logger.level
    module_logger.setLevel(logging.DEBUG)
    try:
        with caplog.at_level(logging.INFO):
            with backend._suppress_third_party_info_logs():
                logging.getLogger().info("download models from model hub: ms")
                print("funasr version: 1.3.1.")
    finally:
        module_logger.setLevel(original_level)

    assert "download models from model hub: ms" in caplog.text
    captured = capsys.readouterr()
    assert "funasr version: 1.3.1." in captured.out


def test_dependency_info_filter_demotes_root_info_from_dependency_paths() -> None:
    """Root records emitted from dependency files should be demoted to DEBUG."""
    policy = DependencyLogPolicy(
        logger_prefixes=frozenset({"funasr", "modelscope"}),
        root_path_markers=frozenset({"/site-packages/funasr/"}),
    )
    record = logging.LogRecord(
        name="root",
        level=logging.INFO,
        pathname="/tmp/site-packages/funasr/auto/auto_model.py",
        lineno=1,
        msg="download models from model hub: ms",
        args=(),
        exc_info=None,
    )

    assert is_dependency_log_record(record, policy=policy) is True
    keep = DependencyLogFilter(policy=policy).filter(record)

    assert keep is True
    assert record.levelno == logging.DEBUG


def test_dependency_info_filter_leaves_non_dependency_root_records_unchanged() -> None:
    """Root records outside dependency paths should retain original level."""
    policy = DependencyLogPolicy(
        logger_prefixes=frozenset({"funasr", "modelscope"}),
        root_path_markers=frozenset({"/site-packages/funasr/"}),
    )
    record = logging.LogRecord(
        name="root",
        level=logging.INFO,
        pathname="/tmp/myapp/runtime.py",
        lineno=1,
        msg="application info",
        args=(),
        exc_info=None,
    )

    assert is_dependency_log_record(record, policy=policy) is False
    keep = DependencyLogFilter(policy=policy).filter(record)

    assert keep is True
    assert record.levelno == logging.INFO


def test_scoped_dependency_log_policy_only_applies_within_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Scoped dependency policy should not mutate logging behavior outside context."""
    policy = DependencyLogPolicy(logger_prefixes=frozenset({"funasr"}))
    dependency_logger = logging.getLogger("funasr.auto.auto_model")

    with caplog.at_level(logging.INFO):
        dependency_logger.info("outside-before")
        with scoped_dependency_log_policy(policy=policy, keep_demoted=False):
            dependency_logger.info("inside-context")
        dependency_logger.info("outside-after")

    assert "outside-before" in caplog.text
    assert "inside-context" not in caplog.text
    assert "outside-after" in caplog.text
