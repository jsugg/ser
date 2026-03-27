"""Contracts for medium feature-dataset extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import ser.models.emotion_model as emotion_model
import ser.models.medium_feature_dataset as mfd
from ser.config import AppConfig, AudioReadConfig
from ser.data import EmbeddingCache
from ser.models.medium_noise_controls import (
    MediumNoiseControlStats,
)
from ser.repr import EncodedSequence, XLSRBackend


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
            audio_read=AudioReadConfig(),
            medium_runtime=SimpleNamespace(
                pool_window_size_seconds=1.25,
                pool_window_stride_seconds=0.5,
            ),
        ),
    )


def test_build_medium_feature_dataset_wires_encode_with_runtime_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset builder should wire medium encode/runtime settings deterministically."""
    settings = _settings_stub()
    utterances = [
        _Utterance(
            sample_id="sample-1",
            corpus="ravdess",
            audio_path=Path("clip.wav"),
            label="happy",
            language="en",
            start_seconds=0.1,
            duration_seconds=1.0,
        )
    ]
    captured: dict[str, object] = {}

    def _fake_encode_sequence(**kwargs: object) -> EncodedSequence:
        captured["encode_kwargs"] = kwargs
        return cast(EncodedSequence, object())

    def _apply_noise_controls(
        pooled_features: np.ndarray,
    ) -> tuple[np.ndarray, MediumNoiseControlStats]:
        return pooled_features, MediumNoiseControlStats(total_windows=1, kept_windows=1)

    def _merge_noise_stats(
        base: MediumNoiseControlStats,
        incoming: MediumNoiseControlStats,
    ) -> MediumNoiseControlStats:
        return MediumNoiseControlStats(
            total_windows=base.total_windows + incoming.total_windows,
            kept_windows=base.kept_windows + incoming.kept_windows,
            dropped_low_std_windows=(
                base.dropped_low_std_windows + incoming.dropped_low_std_windows
            ),
            dropped_cap_windows=base.dropped_cap_windows + incoming.dropped_cap_windows,
            forced_keep_windows=base.forced_keep_windows + incoming.forced_keep_windows,
        )

    def _fake_build_prepared(
        **kwargs: object,
    ) -> tuple[np.ndarray, list[str], list[str], MediumNoiseControlStats]:
        captured["window_size_seconds"] = kwargs["window_size_seconds"]
        captured["window_stride_seconds"] = kwargs["window_stride_seconds"]
        assert kwargs["apply_noise_controls"] is _apply_noise_controls
        assert kwargs["merge_noise_stats"] is _merge_noise_stats
        assert kwargs["initial_noise_stats"] == MediumNoiseControlStats()
        encode_callable = kwargs["encode_sequence"]
        assert callable(encode_callable)
        encode_callable(utterances[0])
        window_meta_factory = kwargs["window_meta_factory"]
        assert callable(window_meta_factory)
        meta = cast(list[str], [window_meta_factory("sample-1", "ravdess", "en")])
        return (
            np.asarray([[0.1, 0.2]], dtype=np.float64),
            ["happy"],
            meta,
            MediumNoiseControlStats(total_windows=1, kept_windows=1),
        )

    monkeypatch.setattr(
        mfd,
        "_build_prepared_medium_feature_dataset",
        _fake_build_prepared,
    )

    matrix, labels, meta, stats = mfd.build_medium_feature_dataset(
        utterances=utterances,
        backend=cast(XLSRBackend, object()),
        cache=cast(EmbeddingCache, object()),
        model_id="facebook/wav2vec2-xls-r-300m",
        settings=settings,
        encode_sequence=_fake_encode_sequence,
        apply_noise_controls=_apply_noise_controls,
        merge_noise_stats=_merge_noise_stats,
        window_meta_factory=lambda sample_id, corpus, language: (
            f"{sample_id}:{corpus}:{language}"
        ),
    )

    assert matrix.shape == (1, 2)
    assert labels == ["happy"]
    assert meta == ["sample-1:ravdess:en"]
    assert stats.total_windows == 1
    assert stats.kept_windows == 1
    assert captured["window_size_seconds"] == 1.25
    assert captured["window_stride_seconds"] == 0.5
    encode_kwargs = captured["encode_kwargs"]
    assert isinstance(encode_kwargs, dict)
    assert encode_kwargs["audio_path"] == "clip.wav"
    assert encode_kwargs["model_id"] == "facebook/wav2vec2-xls-r-300m"
    assert encode_kwargs["start_seconds"] == 0.1
    assert encode_kwargs["duration_seconds"] == 1.0


def test_encode_medium_sequence_uses_profile_specific_default_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium encoder should resolve the settings-derived default model id."""
    settings = _settings_stub()
    captured: dict[str, object] = {}
    expected_encoded = cast(EncodedSequence, object())

    def _fake_encode_sequence_with_cache(**kwargs: object) -> EncodedSequence:
        captured.update(kwargs)
        return expected_encoded

    monkeypatch.setattr(mfd, "encode_sequence_with_cache", _fake_encode_sequence_with_cache)
    monkeypatch.setattr(mfd, "resolve_medium_model_id", lambda _settings: "resolved/xlsr")

    encoded = mfd.encode_medium_sequence(
        audio_path="clip.wav",
        start_seconds=0.1,
        duration_seconds=1.0,
        backend=cast(XLSRBackend, object()),
        cache=cast(EmbeddingCache, object()),
        model_id=None,
        settings=settings,
        backend_id=emotion_model.MEDIUM_BACKEND_ID,
        logger=emotion_model.logger,
    )

    assert encoded is expected_encoded
    assert captured["model_id"] == "resolved/xlsr"
    assert captured["backend_id"] == emotion_model.MEDIUM_BACKEND_ID
    assert captured["frame_size_seconds"] == 1.25
    assert captured["frame_stride_seconds"] == 0.5
