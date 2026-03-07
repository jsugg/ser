"""Tests for grouped-evaluation metadata in medium training split workflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from ser.config import AppConfig
from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.models import emotion_model as em


def _sample_utterance(
    actor_id: int,
    emotion_label: str,
    *,
    speaker_id: str | None = "auto",
    split: Literal["train", "dev", "test"] | None = None,
    corpus: str = "ravdess",
) -> Utterance:
    """Builds deterministic RAVDESS-like utterances."""
    path = Path(
        f"ser/dataset/ravdess/Actor_{actor_id:02d}/03-01-03-01-01-01-{actor_id:02d}.wav"
    )
    resolved_speaker = speaker_id
    if speaker_id == "auto":
        resolved_speaker = f"ravdess:{actor_id:02d}"
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=f"{corpus}:{path.as_posix()}:{emotion_label}",
        corpus=corpus,
        audio_path=path,
        label=emotion_label,
        speaker_id=resolved_speaker,
        split=split,
        language="en",
    )


def test_medium_split_prefers_grouped_strategy_when_speaker_ids_are_available() -> None:
    """Medium file-level split should use grouped speakers when metadata is complete."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            training=SimpleNamespace(
                test_size=0.25,
                random_state=7,
                stratify_split=True,
            )
        ),
    )
    samples = [
        _sample_utterance(1, "happy"),
        _sample_utterance(1, "sad"),
        _sample_utterance(2, "happy"),
        _sample_utterance(2, "sad"),
        _sample_utterance(3, "happy"),
        _sample_utterance(3, "sad"),
        _sample_utterance(4, "happy"),
        _sample_utterance(4, "sad"),
    ]

    train_samples, test_samples, split_metadata = em._split_utterances(
        samples,
        settings=settings,
    )

    assert train_samples
    assert test_samples
    assert split_metadata.split_strategy == "group_shuffle_split"
    assert split_metadata.speaker_grouped is True
    assert split_metadata.speaker_id_coverage == pytest.approx(1.0)
    assert split_metadata.speaker_overlap_count == 0


def test_medium_split_falls_back_when_speaker_metadata_is_incomplete() -> None:
    """Missing speaker IDs should trigger non-grouped fallback with diagnostics."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            training=SimpleNamespace(
                test_size=0.25,
                random_state=7,
                stratify_split=True,
            )
        ),
    )
    samples = [
        _sample_utterance(1, "happy", speaker_id=None, corpus="custom"),
        _sample_utterance(2, "sad", speaker_id=None, corpus="custom"),
        _sample_utterance(3, "happy"),
        _sample_utterance(4, "sad"),
    ]

    _, _, split_metadata = em._split_utterances(samples, settings=settings)

    assert split_metadata.split_strategy == "hash_stratified_split"
    assert split_metadata.speaker_grouped is False
    assert split_metadata.speaker_id_coverage == pytest.approx(0.5)


def test_medium_split_prefers_explicit_manifest_split() -> None:
    """Explicit manifest split tags should take priority over speaker grouping."""
    settings = cast(
        AppConfig,
        SimpleNamespace(
            training=SimpleNamespace(
                test_size=0.25,
                random_state=7,
                stratify_split=True,
            )
        ),
    )
    samples = [
        _sample_utterance(1, "happy", split="train"),
        _sample_utterance(1, "sad", split="train"),
        _sample_utterance(2, "happy", split="test"),
        _sample_utterance(2, "sad", split="test"),
    ]

    train_samples, test_samples, split_metadata = em._split_utterances(
        samples,
        settings=settings,
    )

    assert len(train_samples) == 2
    assert len(test_samples) == 2
    assert split_metadata.split_strategy == "manifest_split"
    assert split_metadata.speaker_grouped is False
