"""Integrated evidence for resumable preparation and deterministic resource cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

import ser._internal.models.training_readiness as readiness_module  # noqa: TID251
from ser._internal.config.schema import DatasetConfig, ModelsConfig, TrainingConfig  # noqa: TID251
from ser._internal.data.embedding_cache import (  # noqa: TID251
    EmbeddingCache,
    EmbeddingCacheEntry,
)
from ser._internal.data.manifest import Utterance  # noqa: TID251
from ser._internal.models import dataset_splitting  # noqa: TID251
from ser._internal.models import training_orchestration as orchestration  # noqa: TID251
from ser._internal.models.training_orchestration import training_operation_scope  # noqa: TID251
from ser._internal.models.training_readiness import (  # noqa: TID251
    ReadinessReport,
    TrainingMode,
    TrainingOperation,
)
from ser._internal.repr import EncodedSequence, FeatureBackend  # noqa: TID251
from ser.config import AppConfig


def _settings(tmp_path: Path) -> AppConfig:
    return AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        models=ModelsConfig(folder=tmp_path / "models", model_cache_dir=tmp_path / "cache"),
    )


def _inventory(tmp_path: Path) -> tuple[Utterance, ...]:
    samples: list[Utterance] = []
    for index in range(8):
        path = tmp_path / f"{index}.wav"
        path.write_bytes(f"audio-{index}".encode())
        samples.append(
            Utterance(
                1,
                f"sample-{index}",
                "fixture",
                path,
                "calm" if index % 2 == 0 else "happy",
            )
        )
    return tuple(samples)


def _readiness(samples: tuple[Utterance, ...]) -> ReadinessReport:
    return ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
        effective_sample_ids=tuple(item.sample_id for item in samples),
    )


def test_interrupted_prepare_has_no_ready_plan_and_rerun_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    readiness = _readiness(samples)
    cache = EmbeddingCache(settings.tmp_folder / "fast_features")
    compute_calls = 0

    def _compute() -> EncodedSequence:
        nonlocal compute_calls
        compute_calls += 1
        return EncodedSequence(
            embeddings=np.asarray([[0.1, 0.2]], dtype=np.float32),
            frame_start_seconds=np.asarray([0.0], dtype=np.float64),
            frame_end_seconds=np.asarray([0.1], dtype=np.float64),
            backend_id="handcrafted",
        )

    def _cache_lookup() -> EmbeddingCacheEntry:
        return cache.get_or_compute(
            audio_path=str(samples[0].audio_path),
            backend_id="handcrafted",
            model_id="builtin",
            frame_size_seconds=1.0,
            frame_stride_seconds=1.0,
            compute=_compute,
        )

    first = _cache_lookup()
    assert first.cache_hit is False

    original_replace = readiness_module.os.replace
    interrupted = False

    def _interrupt_plan(source: str | Path, destination: str | Path) -> None:
        nonlocal interrupted
        if Path(destination).name == "prepared-training-fast.json" and not interrupted:
            interrupted = True
            raise OSError("simulated plan publication interruption")
        original_replace(source, destination)

    monkeypatch.setattr(readiness_module.os, "replace", _interrupt_plan)
    initial_train, initial_test, _ = dataset_splitting.split_utterances(
        samples=list(samples), settings=settings, logger=orchestration.logger
    )
    x_train = np.ones((len(initial_train), 2), dtype=np.float64)
    x_test = np.ones((len(initial_test), 2), dtype=np.float64)

    def _publish() -> None:
        orchestration.publish_prepared_features(
            settings=settings,
            backend_id="handcrafted",
            model_id="builtin",
            device="cpu",
            dtype="float64",
            utterances=samples,
            x_train=x_train,
            x_test=x_test,
            y_train=[item.require_label() for item in initial_train],
            y_test=[item.require_label() for item in initial_test],
            metadata={"split_metadata": {}},
            cache_namespace="fast_features",
            windowing_policy={"strategy": "fixture"},
            noise_statistics={},
        )

    plan_path = settings.tmp_folder / "prepared-training-fast.json"
    with training_operation_scope(TrainingOperation(mode=TrainingMode.PREPARE_ONLY)) as state:
        state.readiness = readiness
        state.utterances = samples
        with pytest.raises(OSError, match="publication interruption"):
            _publish()
        assert not plan_path.exists()
        assert not list(plan_path.parent.glob(f".{plan_path.name}.*"))

        second = _cache_lookup()
        assert second.cache_hit is True
        assert compute_calls == 1
        _publish()

    assert plan_path.is_file()
    assert orchestration.load_prepared_plan(plan_path).payload_path


@pytest.mark.parametrize("failure_phase", ["prepare", "publish", "unexpected"])
def test_checked_backend_closes_once_for_every_failure_path(failure_phase: str) -> None:
    closed = 0

    class _Backend:
        backend_id = "fixture"
        feature_dim = 2

        def close(self) -> None:
            nonlocal closed
            closed += 1

    with pytest.raises(RuntimeError, match=failure_phase):
        with training_operation_scope(TrainingOperation()):
            orchestration.reuse_checked_backend(
                backend_id="fixture",
                model_id="model@revision",
                device="cpu",
                dtype="float32",
                build=lambda: cast(FeatureBackend, _Backend()),
            )
            raise RuntimeError(failure_phase)
    assert closed == 1
