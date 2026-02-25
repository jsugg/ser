"""Tests for grouped-evaluation metadata in medium training split workflow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ser.models import emotion_model as em


def _sample_path(actor_id: int, emotion_code: str) -> str:
    """Builds deterministic RAVDESS-style sample paths."""
    return (
        "ser/dataset/ravdess/"
        f"Actor_{actor_id:02d}/03-01-{emotion_code}-01-01-01-{actor_id:02d}.wav"
    )


def test_medium_split_prefers_grouped_strategy_when_speaker_ids_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Medium file-level split should use grouped speakers when metadata is complete."""
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(
            training=SimpleNamespace(
                test_size=0.25,
                random_state=7,
                stratify_split=True,
            )
        ),
    )
    samples = [
        (_sample_path(1, "03"), "happy"),
        (_sample_path(1, "04"), "sad"),
        (_sample_path(2, "03"), "happy"),
        (_sample_path(2, "04"), "sad"),
        (_sample_path(3, "03"), "happy"),
        (_sample_path(3, "04"), "sad"),
        (_sample_path(4, "03"), "happy"),
        (_sample_path(4, "04"), "sad"),
    ]

    train_samples, test_samples, split_metadata = em._split_labeled_audio_samples(samples)

    assert train_samples
    assert test_samples
    assert split_metadata.split_strategy == "group_shuffle_split"
    assert split_metadata.speaker_grouped is True
    assert split_metadata.speaker_id_coverage == pytest.approx(1.0)
    assert split_metadata.speaker_overlap_count == 0


def test_medium_split_falls_back_when_speaker_metadata_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing speaker IDs should trigger non-grouped fallback with diagnostics."""
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(
            training=SimpleNamespace(
                test_size=0.25,
                random_state=7,
                stratify_split=True,
            )
        ),
    )
    samples = [
        ("ser/dataset/ravdess/invalid_01.wav", "happy"),
        ("ser/dataset/ravdess/invalid_02.wav", "sad"),
        (_sample_path(3, "03"), "happy"),
        (_sample_path(4, "04"), "sad"),
    ]

    _, _, split_metadata = em._split_labeled_audio_samples(samples)

    assert split_metadata.split_strategy == "stratified_shuffle_split_fallback"
    assert split_metadata.speaker_grouped is False
    assert split_metadata.speaker_id_coverage == pytest.approx(0.5)
