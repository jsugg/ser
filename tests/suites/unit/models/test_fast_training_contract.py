"""Canonical fast features and speaker-level training evidence."""

import hashlib
import json
import logging
from dataclasses import replace
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sklearn.neural_network import MLPClassifier

from ser._internal.analysis.audio_input import DecodedAudio, canonicalize, decode_file
from ser._internal.analysis.fast import FastBackend
from ser._internal.data import Utterance
from ser._internal.data.data_loader import load_checked_fast_data
from ser._internal.data.manifest import SplitName
from ser._internal.features import feature_extractor
from ser._internal.models import dataset_splitting, fast_training, fast_training_entrypoints
from ser._internal.models.artifact_envelope import build_model_artifact
from ser._internal.models.artifact_persistence import persist_pickle_artifact
from ser._internal.repr import HandcraftedBackend
from ser.config import AppConfig


def test_training_rows_equal_analysis_windows(tmp_path: Path) -> None:
    settings = AppConfig(emotions={}, tmp_folder=tmp_path)
    path = tmp_path / "quiet.wav"
    t = np.arange(48_000 * 4 + 1) / 48_000
    sf.write(path, np.column_stack((0.08 * np.sin(500 * t**2), 0.02 * np.cos(900 * t))), 48_000)
    sample = Utterance(1, "clip", "fixture", path, "calm", speaker_id="actor")
    decoded = decode_file(path, max_duration_s=10)
    assert isinstance(decoded, DecodedAudio)
    backend = FastBackend(
        MLPClassifier(), (), HandcraftedBackend(feature_flags=settings.feature_flags)
    )
    expected = backend.extract(canonicalize(decoded), deadline_monotonic_s=float("inf"))
    checked = load_checked_fast_data(
        settings=settings,
        utterances=[
            replace(sample, split="test"),
            replace(sample, sample_id="a", speaker_id="a", split="train", duration_seconds=1.0),
            replace(
                sample,
                sample_id="b",
                speaker_id="b",
                split="train",
                label="sad",
                start_seconds=1.0,
                duration_seconds=1.0,
            ),
        ],
    )
    assert checked is not None
    train_rows, rows, _, _ = checked.matrices
    assert checked.test_sample_ids == [sample.sample_id] * len(expected.spans)
    np.testing.assert_array_equal(rows, expected.vectors)
    assert checked.train_sample_ids == ["a", "b"]
    first_second = canonicalize(DecodedAudio(decoded.samples[:48_000], 48_000))
    next_second = canonicalize(DecodedAudio(decoded.samples[48_000:96_000], 48_000))
    expected_train = np.vstack(
        [
            backend.extract(segment, deadline_monotonic_s=float("inf")).vectors
            for segment in (first_second, next_second)
        ]
    )
    np.testing.assert_array_equal(train_rows, expected_train)
    reduced = feature_extractor.fast_training_vectors(
        sample,
        feature_flags=replace(settings.feature_flags, mel=False),
        cache_dir=tmp_path / "fast_features",
    )
    assert reduced.shape == (len(rows), rows.shape[1] - 128)


def test_fast_split_evidence_is_order_independent_and_speaker_disjoint() -> None:
    settings = AppConfig(emotions={})
    samples = [
        Utterance(1, f"{i}-{j}", "fixture", Path("unused"), label, speaker_id=str(i))
        for i in range(24)
        for j, label in enumerate(("calm", "happy"))
    ]

    def split(rows: list[Utterance]) -> str:
        train, dev, test, _ = dataset_splitting.split_utterances_three_way(
            samples=rows, settings=settings, logger=logging.getLogger(__name__)
        )
        partitions = {"train": train, "dev": dev, "test": test}
        digest = dataset_splitting.fast_split_digest(partitions)
        pairs = sorted((u.sample_id, name) for name, group in partitions.items() for u in group)
        assert (
            digest == hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
        )
        speakers = [{u.speaker_id for u in group} for group in partitions.values()]
        assert not (
            speakers[0] & speakers[1] or speakers[0] & speakers[2] or speakers[1] & speakers[2]
        )
        assert all({u.label for u in group} == {"happy", "calm"} for group in partitions.values())
        return digest

    assert split(samples) == split(samples[::-1])
    with pytest.raises(ValueError, match="speaker identity"):
        split([replace(samples[0], speaker_id=None), *samples[1:]])
    with pytest.raises(ValueError, match="speaker.*overlap"):
        dataset_splitting.fast_split_digest({"train": samples[:1], "test": samples[1:]})


def test_feature_contract_rejects_stale_revision_and_changed_flags() -> None:
    settings = AppConfig(emotions={})
    revision = feature_extractor.fast_feature_contract(settings.feature_flags)
    fast_training.validate_fast_feature_contract(
        {"model_revision": revision}, settings.feature_flags
    )
    for artifact_metadata in ({}, {"model_revision": revision.replace("16000", "48000")}):
        with pytest.raises(ValueError, match="feature contract"):
            fast_training.validate_fast_feature_contract(artifact_metadata, settings.feature_flags)
    with pytest.raises(ValueError, match="feature contract"):
        fast_training.validate_fast_feature_contract(
            {"model_revision": revision}, replace(settings.feature_flags, mel=False)
        )


def test_evaluation_load_refuses_native_rate_model(tmp_path: Path) -> None:
    settings = AppConfig(emotions={})
    settings = replace(settings, models=replace(settings.models, folder=tmp_path))
    artifact = build_model_artifact(
        model=MLPClassifier(),
        feature_vector_size=193,
        labels=["happy", "sad"],
        training_samples=2,
        model_revision="handcrafted-v0-48000hz",
    )
    persist_pickle_artifact(settings.models.model_file, artifact)
    with pytest.raises(ValueError, match="feature contract"):
        fast_training_entrypoints.load_fast_model_for_evaluation(settings)


def test_fast_readiness_split_refuses_unknown_speakers_and_leaking_native_dev() -> None:
    settings = AppConfig(emotions={})
    names: tuple[SplitName, ...] = ("train", "dev", "test")
    rows = [
        Utterance(1, str(i), "fixture", Path("unused"), "calm", speaker_id=str(i), split=part)
        for i, part in enumerate(names)
    ]
    for bad, reason in [
        (replace(rows[1], speaker_id=None), "speaker identity"),
        (replace(rows[1], speaker_id="0"), "speaker overlap"),
    ]:
        with pytest.raises(ValueError, match=reason):
            dataset_splitting.split_utterances_three_way(
                samples=[rows[0], bad, rows[2]],
                settings=settings,
                logger=logging.getLogger(__name__),
            )


def test_fast_partitions_keep_explicit_speaker_identity_across_corpora() -> None:
    samples = [
        Utterance(1, f"{corpus}-{i}", corpus, Path("unused"), "calm", speaker_id=str(i))
        for corpus in ("first", "second")
        for i in range(12)
    ]
    train, dev, test, _ = dataset_splitting.split_utterances_three_way(
        samples=samples, settings=AppConfig(emotions={}), logger=logging.getLogger(__name__)
    )
    speakers = [{s.speaker_id for s in part} for part in (train, dev, test)]
    assert sum(map(len, speakers)) == 12
    with pytest.raises(ValueError, match="speaker overlap"):
        dataset_splitting.fast_split_digest({"train": samples[:12], "test": samples[12:]})


def test_fast_cache_recovers_corrupt_entry_and_reports_activity(tmp_path: Path) -> None:
    from ser._internal.models.training_orchestration import training_operation_scope
    from ser._internal.models.training_readiness import TrainingOperation

    path = tmp_path / "tone.wav"
    sf.write(path, np.sin(np.arange(16_000) * 0.1), 16_000)
    sample = Utterance(1, "clip", "fixture", path, "calm", speaker_id="actor")
    cache = tmp_path / "features"
    flags = replace(AppConfig(emotions={}).feature_flags, tonnetz=False)
    with training_operation_scope(TrainingOperation()) as state:
        expected = feature_extractor.fast_training_vectors(
            sample, feature_flags=flags, cache_dir=cache
        )
        next(cache.glob("*.npy")).write_bytes(b"incomplete")
        for _ in range(2):
            actual = feature_extractor.fast_training_vectors(
                sample, feature_flags=flags, cache_dir=cache
            )
            np.testing.assert_array_equal(actual, expected)
        assert (state.cache_hits, state.cache_misses, state.recomputed_cache_entries) == (1, 2, 1)


@pytest.mark.parametrize("changed_input", ["contract", "ser", "librosa", "numpy", "scipy"])
def test_fast_cache_misses_when_contract_or_version_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_input: str
) -> None:
    path = tmp_path / "tone.wav"
    sf.write(path, np.sin(np.arange(16_000) * 0.1), 16_000)
    sample = Utterance(1, "clip", "fixture", path, "calm", speaker_id="actor")
    flags = replace(AppConfig(emotions={}).feature_flags, tonnetz=False)
    cache = tmp_path / "features"
    expected = feature_extractor.fast_training_vectors(sample, feature_flags=flags, cache_dir=cache)
    old_rows = np.full_like(expected, 7.0)
    np.save(next(cache.glob("*.npy")), old_rows, allow_pickle=False)
    np.testing.assert_array_equal(
        feature_extractor.fast_training_vectors(sample, feature_flags=flags, cache_dir=cache),
        old_rows,
    )

    if changed_input == "contract":
        old_contract = feature_extractor.fast_feature_contract(flags)
        monkeypatch.setattr(
            feature_extractor, "fast_feature_contract", lambda _flags: old_contract + "-revised"
        )
    else:
        installed_version = metadata.version
        monkeypatch.setattr(
            metadata,
            "version",
            lambda name: "999.0" if name == changed_input else installed_version(name),
        )
    np.testing.assert_array_equal(
        feature_extractor.fast_training_vectors(sample, feature_flags=flags, cache_dir=cache),
        expected,
    )
    assert len(list(cache.glob("*.npy"))) == 2


def test_fast_cache_separates_blas_thread_counts(tmp_path: Path) -> None:
    from threadpoolctl import threadpool_info, threadpool_limits

    path = tmp_path / "tone.wav"
    sf.write(path, np.sin(np.arange(16_000) * 0.1), 16_000)
    sample = Utterance(1, "clip", "fixture", path, "calm", speaker_id="actor")
    flags = AppConfig(emotions={}).feature_flags
    cache = tmp_path / "features"
    for threads in (1, 2):
        with threadpool_limits(threads, user_api="blas"):
            fresh = feature_extractor.fast_training_vectors(sample, feature_flags=flags)
            if not any(
                p["user_api"] == "blas" and p["num_threads"] == threads for p in threadpool_info()
            ):
                pytest.skip("BLAS thread limits are unavailable on this build")
            cached = feature_extractor.fast_training_vectors(
                sample, feature_flags=flags, cache_dir=cache
            )
            np.testing.assert_array_equal(cached, fresh)
    assert len(list(cache.glob("*.npy"))) == 2
