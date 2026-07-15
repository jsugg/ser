"""Regressions for deterministic utterance splitting, including O(n) speaker accounting."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ser._internal.config.schema import DatasetConfig, TrainingConfig
from ser._internal.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser._internal.models.dataset_splitting import (
    _partition_speakers,
    split_utterances_three_way,
)
from ser.config import AppConfig

logger = logging.getLogger(__name__)


def _settings(tmp_path: Path, *, test_size: float = 0.25, dev_size: float = 0.20) -> AppConfig:
    return AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=test_size, dev_size=dev_size),
    )


def _utterance(index: int, label: str, *, speaker: str | None = None) -> Utterance:
    """Builds an utterance with no manifest split so the hash-stratified path runs."""
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=f"fixture:{index}",
        corpus="fixture",
        audio_path=Path(f"{index}.wav"),
        label=label,
        speaker_id=speaker,
        split=None,
    )


def test_hash_stratified_three_way_split_is_complete_disjoint_and_deterministic(
    tmp_path: Path,
) -> None:
    """Partial speaker coverage forces the hash path; partitions must stay well-formed."""
    settings = _settings(tmp_path)
    # Half the samples carry a speaker id -> coverage < 100% -> hash-stratified path.
    samples = [
        _utterance(
            index,
            "calm" if index % 2 == 0 else "happy",
            speaker=f"spk-{index % 5}" if index % 2 == 0 else None,
        )
        for index in range(60)
    ]

    train, dev, test, metadata = split_utterances_three_way(
        samples=list(samples), settings=settings, logger=logger
    )

    assert metadata.split_strategy == "hash_stratified_split+dev"
    # Complete and disjoint partition over the original identities.
    partition_ids = [item.sample_id for item in (*train, *dev, *test)]
    assert sorted(partition_ids) == sorted(item.sample_id for item in samples)
    assert len(partition_ids) == len(set(partition_ids))
    assert train and dev and test

    # Deterministic across repeated invocations.
    train2, dev2, test2, metadata2 = split_utterances_three_way(
        samples=list(samples), settings=settings, logger=logger
    )
    assert [u.sample_id for u in train2] == [u.sample_id for u in train]
    assert [u.sample_id for u in dev2] == [u.sample_id for u in dev]
    assert [u.sample_id for u in test2] == [u.sample_id for u in test]
    assert metadata2 == metadata


def test_partition_speakers_matches_naive_membership() -> None:
    """The set-based helper must equal the original O(n**2) list-membership result."""
    samples = [
        _utterance(0, "calm", speaker="a"),
        _utterance(1, "happy", speaker="b"),
        _utterance(2, "calm", speaker="a"),  # duplicate speaker
        _utterance(3, "happy", speaker=None),  # no speaker -> excluded
        _utterance(4, "calm", speaker="c"),
    ]
    speaker_ids = [item.speaker_id for item in samples]
    partition = [samples[0], samples[2], samples[3]]  # speakers a, a, None

    expected = {
        speaker
        for utterance, speaker in zip(samples, speaker_ids, strict=False)
        if utterance in partition and speaker is not None
    }
    result = _partition_speakers(partition, samples=samples, speaker_ids=speaker_ids)

    assert result == expected == {"a"}


def test_three_way_split_scales_linearly_on_large_inventory(tmp_path: Path) -> None:
    """Guards against reintroducing O(n**2) membership: 20k rows must finish quickly.

    The prior list-membership implementation took minutes at this size (~n**2 equality
    checks); the linear implementation completes in well under a second. The 30s bound
    is a wide margin that only a quadratic regression can exceed.
    """
    settings = _settings(tmp_path)
    samples = [_utterance(index, "calm" if index % 2 == 0 else "happy") for index in range(20_000)]

    started = time.perf_counter()
    train, dev, test, metadata = split_utterances_three_way(
        samples=list(samples), settings=settings, logger=logger
    )
    elapsed = time.perf_counter() - started

    assert metadata.split_strategy == "hash_stratified_split+dev"
    assert len(train) + len(dev) + len(test) == len(samples)
    assert elapsed < 30.0, f"three-way split took {elapsed:.1f}s; expected linear-time completion"
