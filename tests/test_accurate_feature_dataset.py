"""Contracts for accurate feature-dataset extraction helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ser.models.accurate_feature_dataset as afd
from ser.config import AppConfig
from ser.data import EmbeddingCache
from ser.repr import EncodedSequence, FeatureBackend


@dataclass(frozen=True)
class _Utterance:
    sample_id: str
    corpus: str
    audio_path: Path
    label: str
    language: str | None = None
    start_seconds: float | None = None
    duration_seconds: float | None = None


def _settings_stub() -> AppConfig:
    return cast(
        AppConfig,
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


def test_encode_accurate_sequence_uses_profile_specific_default_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accurate encoder should resolve backend-specific default model id."""
    settings = _settings_stub()
    captured: dict[str, object] = {}
    expected_encoded = cast(EncodedSequence, object())

    def _fake_encode_sequence_with_cache(**kwargs: object) -> EncodedSequence:
        captured.update(kwargs)
        return expected_encoded

    monkeypatch.setattr(
        afd, "encode_sequence_with_cache", _fake_encode_sequence_with_cache
    )
    monkeypatch.setattr(
        afd, "resolve_accurate_model_id", lambda _settings: "resolved/whisper"
    )
    monkeypatch.setattr(
        afd,
        "resolve_accurate_research_model_id",
        lambda _settings: "resolved/emotion2vec",
    )

    encoded = afd.encode_accurate_sequence(
        audio_path="sample.wav",
        backend=cast(FeatureBackend, object()),
        cache=cast(EmbeddingCache, object()),
        model_id=None,
        backend_id="hf_whisper",
        settings=settings,
        accurate_backend_id="hf_whisper",
        accurate_research_backend_id="emotion2vec",
        logger=logging.getLogger("ser.tests.accurate_feature_dataset"),
    )

    assert encoded is expected_encoded
    assert captured["model_id"] == "resolved/whisper"
    assert captured["frame_size_seconds"] == 2.0
    assert captured["frame_stride_seconds"] == 0.5


def test_build_accurate_feature_dataset_wires_encode_with_runtime_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset builder should call encode with stable runtime wiring."""
    settings = _settings_stub()
    utterances = [
        _Utterance(
            sample_id="sample-1",
            corpus="ravdess",
            audio_path=Path("clip.wav"),
            label="happy",
            language="en",
            start_seconds=0.25,
            duration_seconds=1.5,
        )
    ]
    captured: dict[str, object] = {}

    def _fake_encode_sequence(**kwargs: object) -> EncodedSequence:
        captured["encode_kwargs"] = kwargs
        return cast(EncodedSequence, object())

    def _fake_build_prepared(
        **kwargs: object,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        captured["window_size_seconds"] = kwargs["window_size_seconds"]
        captured["window_stride_seconds"] = kwargs["window_stride_seconds"]
        encode_callable = kwargs["encode_sequence"]
        assert callable(encode_callable)
        encode_callable(utterances[0])
        window_meta_factory = kwargs["window_meta_factory"]
        assert callable(window_meta_factory)
        meta = cast(list[str], [window_meta_factory("sample-1", "ravdess", "en")])
        return np.asarray([[0.1, 0.2]], dtype=np.float64), ["happy"], meta

    monkeypatch.setattr(
        afd,
        "_build_prepared_accurate_feature_dataset",
        _fake_build_prepared,
    )
    monkeypatch.setattr(
        afd, "resolve_accurate_model_id", lambda _settings: "resolved/whisper"
    )
    monkeypatch.setattr(
        afd,
        "resolve_accurate_research_model_id",
        lambda _settings: "resolved/emotion2vec",
    )

    matrix, labels, meta = afd.build_accurate_feature_dataset(
        utterances=utterances,
        backend=cast(FeatureBackend, object()),
        cache=cast(EmbeddingCache, object()),
        model_id="configured/model",
        backend_id="emotion2vec",
        settings=settings,
        accurate_backend_id="hf_whisper",
        accurate_research_backend_id="emotion2vec",
        logger=logging.getLogger("ser.tests.accurate_feature_dataset"),
        window_meta_factory=lambda sample_id, corpus, language: (
            f"{sample_id}:{corpus}:{language}"
        ),
        encode_sequence=_fake_encode_sequence,
    )

    assert matrix.shape == (1, 2)
    assert labels == ["happy"]
    assert meta == ["sample-1:ravdess:en"]
    assert captured["window_size_seconds"] == 2.5
    assert captured["window_stride_seconds"] == 0.75
    encode_kwargs = captured["encode_kwargs"]
    assert isinstance(encode_kwargs, dict)
    assert encode_kwargs["model_id"] == "configured/model"
    assert encode_kwargs["backend_id"] == "emotion2vec"
    assert encode_kwargs["audio_path"] == "clip.wav"
