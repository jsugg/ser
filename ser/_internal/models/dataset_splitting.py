"""Deterministic dataset split helpers for profile training workflows."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from hashlib import sha1

import numpy as np
from sklearn.model_selection import train_test_split

from ser.config import AppConfig
from ser.data import LabeledAudioSample, Utterance
from ser.data.data_loader import extract_ravdess_speaker_id_from_path
from ser.train.eval import grouped_train_test_split


@dataclass(frozen=True)
class MediumSplitMetadata:
    """Split diagnostics persisted for grouped-evaluation traceability."""

    split_strategy: str
    speaker_grouped: bool
    speaker_id_coverage: float
    train_unique_speakers: int
    test_unique_speakers: int
    speaker_overlap_count: int


def split_labeled_audio_samples(
    *,
    samples: list[LabeledAudioSample],
    settings: AppConfig,
    logger: logging.Logger,
) -> tuple[list[LabeledAudioSample], list[LabeledAudioSample], MediumSplitMetadata]:
    """Splits labeled files with grouped-speaker preference and traceable metadata."""
    if len(samples) < 2:
        raise RuntimeError("Medium training requires at least two labeled audio files.")

    indices: np.ndarray = np.arange(len(samples), dtype=np.int64)
    labels: list[str] = [label for _, label in samples]
    raw_speaker_ids: list[str | None] = [
        extract_ravdess_speaker_id_from_path(audio_path) for audio_path, _ in samples
    ]
    resolved_speaker_ids = [item for item in raw_speaker_ids if item is not None]
    speaker_coverage = float(len(resolved_speaker_ids)) / float(len(samples))

    split_strategy = "stratified_shuffle_split"
    train_idx = np.empty(0, dtype=np.int64)
    test_idx = np.empty(0, dtype=np.int64)
    can_group_by_speaker = (
        len(resolved_speaker_ids) == len(samples) and len(set(resolved_speaker_ids)) >= 2
    )
    if can_group_by_speaker:
        grouped_features = np.zeros((len(samples), 1), dtype=np.float64)
        try:
            grouped_split = grouped_train_test_split(
                grouped_features,
                labels,
                [str(item) for item in resolved_speaker_ids],
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
            )
            train_idx = grouped_split.train_indices
            test_idx = grouped_split.test_indices
            split_strategy = "group_shuffle_split"
        except ValueError as err:
            logger.warning(
                "Medium grouped split failed (%s); falling back to stratified split.",
                err,
            )
            can_group_by_speaker = False

    if not can_group_by_speaker:
        split_strategy = "stratified_shuffle_split_fallback"
        stratify_labels: list[str] | None = labels if settings.training.stratify_split else None
        try:
            train_idx_raw, test_idx_raw = train_test_split(
                indices,
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
                stratify=stratify_labels,
            )
            train_idx = np.asarray(train_idx_raw, dtype=np.int64)
            test_idx = np.asarray(test_idx_raw, dtype=np.int64)
        except ValueError as err:
            logger.warning(
                "Medium stratified split failed (%s); falling back to non-stratified split.",
                err,
            )
            train_idx_raw, test_idx_raw = train_test_split(
                indices,
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
                stratify=None,
            )
            train_idx = np.asarray(train_idx_raw, dtype=np.int64)
            test_idx = np.asarray(test_idx_raw, dtype=np.int64)

    train_samples: list[LabeledAudioSample] = [samples[int(index)] for index in train_idx]
    test_samples: list[LabeledAudioSample] = [samples[int(index)] for index in test_idx]
    if train_idx.size == 0 or test_idx.size == 0:
        raise RuntimeError(
            "Medium training split failed to produce deterministic index partitions."
        )
    if not train_samples or not test_samples:
        raise RuntimeError("Medium training split produced an empty partition; adjust test_size.")

    train_speakers = {
        raw_speaker_ids[int(index)]
        for index in train_idx.tolist()
        if raw_speaker_ids[int(index)] is not None
    }
    test_speakers = {
        raw_speaker_ids[int(index)]
        for index in test_idx.tolist()
        if raw_speaker_ids[int(index)] is not None
    }
    speaker_overlap_count = len(train_speakers.intersection(test_speakers))
    if split_strategy == "group_shuffle_split" and speaker_overlap_count > 0:
        raise RuntimeError("Grouped medium split produced overlapping speakers in train/test.")

    return (
        train_samples,
        test_samples,
        MediumSplitMetadata(
            split_strategy=split_strategy,
            speaker_grouped=split_strategy == "group_shuffle_split",
            speaker_id_coverage=speaker_coverage,
            train_unique_speakers=len(train_speakers),
            test_unique_speakers=len(test_speakers),
            speaker_overlap_count=speaker_overlap_count,
        ),
    )


def resolve_corpus_scoped_speaker_id(utterance: Utterance) -> str | None:
    """Returns speaker id with fallback extraction for known RAVDESS layouts."""
    if utterance.speaker_id is not None:
        return utterance.speaker_id
    if utterance.corpus != "ravdess":
        return None
    speaker_raw = extract_ravdess_speaker_id_from_path(str(utterance.audio_path))
    if speaker_raw is None:
        return None
    return f"{utterance.corpus}:{speaker_raw}"


def hash_for_split(sample_id: str, *, salt: str) -> int:
    """Returns deterministic hash token used to order utterances for split."""
    digest = sha1(f"{salt}|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def hash_stratified_split(
    *,
    samples: list[Utterance],
    test_size: float,
    salt: str,
) -> tuple[list[Utterance], list[Utterance]]:
    """Builds deterministic per-label train/test split ordered by salted hash."""
    by_label: dict[str, list[Utterance]] = {}
    for utterance in samples:
        by_label.setdefault(utterance.label, []).append(utterance)

    train: list[Utterance] = []
    test: list[Utterance] = []
    for _, group in sorted(by_label.items(), key=lambda item: item[0]):
        group_sorted = sorted(
            group,
            key=lambda utterance: hash_for_split(utterance.sample_id, salt=salt),
        )
        if len(group_sorted) < 2:
            train.extend(group_sorted)
            continue
        n_test = int(round(test_size * len(group_sorted)))
        if n_test <= 0:
            n_test = 1
        if n_test >= len(group_sorted):
            n_test = len(group_sorted) - 1
        test.extend(group_sorted[:n_test])
        train.extend(group_sorted[n_test:])

    if not test and train:
        train_sorted = sorted(
            train,
            key=lambda utterance: hash_for_split(utterance.sample_id, salt=salt),
        )
        test.append(train_sorted.pop(0))
        train = train_sorted
    if not train and test:
        test_sorted = sorted(
            test,
            key=lambda utterance: hash_for_split(utterance.sample_id, salt=salt),
        )
        train.append(test_sorted.pop(0))
        test = test_sorted
    return train, test


def split_utterances(
    *,
    samples: list[Utterance],
    settings: AppConfig,
    logger: logging.Logger,
) -> tuple[list[Utterance], list[Utterance], MediumSplitMetadata]:
    """Splits utterances deterministically with manifest/speaker/hash policy."""
    if len(samples) < 2:
        raise RuntimeError("Training requires at least two labeled audio files.")

    labels: list[str] = [utterance.label for utterance in samples]
    speaker_ids: list[str | None] = [
        resolve_corpus_scoped_speaker_id(utterance) for utterance in samples
    ]
    resolved_speaker_ids = [item for item in speaker_ids if item is not None]
    speaker_coverage = float(len(resolved_speaker_ids)) / float(len(samples))

    has_manifest_split = all(utterance.split is not None for utterance in samples)
    if has_manifest_split:
        train_split = [utterance for utterance in samples if utterance.split in {"train", "dev"}]
        test_split = [utterance for utterance in samples if utterance.split == "test"]
        if train_split and test_split:
            train_speakers = {
                speaker
                for utterance, speaker in zip(samples, speaker_ids, strict=False)
                if utterance in train_split and speaker is not None
            }
            test_speakers = {
                speaker
                for utterance, speaker in zip(samples, speaker_ids, strict=False)
                if utterance in test_split and speaker is not None
            }
            return (
                train_split,
                test_split,
                MediumSplitMetadata(
                    split_strategy="manifest_split",
                    speaker_grouped=False,
                    speaker_id_coverage=speaker_coverage,
                    train_unique_speakers=len(train_speakers),
                    test_unique_speakers=len(test_speakers),
                    speaker_overlap_count=len(train_speakers.intersection(test_speakers)),
                ),
            )

    can_group_by_speaker = (
        len(resolved_speaker_ids) == len(samples) and len(set(resolved_speaker_ids)) >= 2
    )
    if can_group_by_speaker:
        grouped_features = np.zeros((len(samples), 1), dtype=np.float64)
        try:
            grouped_split = grouped_train_test_split(
                grouped_features,
                labels,
                [str(item) for item in resolved_speaker_ids],
                test_size=settings.training.test_size,
                random_state=settings.training.random_state,
            )
            train_idx = grouped_split.train_indices
            test_idx = grouped_split.test_indices
            train_split = [samples[int(index)] for index in train_idx]
            test_split = [samples[int(index)] for index in test_idx]
            train_speakers = {
                speaker
                for index in train_idx.tolist()
                if (speaker := speaker_ids[int(index)]) is not None
            }
            test_speakers = {
                speaker
                for index in test_idx.tolist()
                if (speaker := speaker_ids[int(index)]) is not None
            }
            overlap = len(train_speakers.intersection(test_speakers))
            if overlap > 0:
                raise RuntimeError("Grouped split produced overlapping speakers in train/test.")
            return (
                train_split,
                test_split,
                MediumSplitMetadata(
                    split_strategy="group_shuffle_split",
                    speaker_grouped=True,
                    speaker_id_coverage=speaker_coverage,
                    train_unique_speakers=len(train_speakers),
                    test_unique_speakers=len(test_speakers),
                    speaker_overlap_count=overlap,
                ),
            )
        except ValueError as err:
            logger.warning(
                "Grouped split failed (%s); falling back to deterministic hash split.",
                err,
            )

    salt = os.getenv("SER_SPLIT_SALT", f"ser:{settings.training.random_state}").strip()
    train_split, test_split = hash_stratified_split(
        samples=samples,
        test_size=settings.training.test_size,
        salt=salt,
    )
    if not train_split or not test_split:
        raise RuntimeError("Deterministic split produced an empty partition; adjust test_size.")
    train_speakers = {
        speaker
        for utterance, speaker in zip(samples, speaker_ids, strict=False)
        if utterance in train_split and speaker is not None
    }
    test_speakers = {
        speaker
        for utterance, speaker in zip(samples, speaker_ids, strict=False)
        if utterance in test_split and speaker is not None
    }
    return (
        train_split,
        test_split,
        MediumSplitMetadata(
            split_strategy="hash_stratified_split",
            speaker_grouped=False,
            speaker_id_coverage=speaker_coverage,
            train_unique_speakers=len(train_speakers),
            test_unique_speakers=len(test_speakers),
            speaker_overlap_count=len(train_speakers.intersection(test_speakers)),
        ),
    )
