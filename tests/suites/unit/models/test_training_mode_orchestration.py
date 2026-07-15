"""Train-mode CLI and mandatory-check orchestration regressions."""

from __future__ import annotations

import errno
import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

import ser._internal.data.data_loader as data_loader  # noqa: TID251
import ser._internal.diagnostics.service as diagnostics_service  # noqa: TID251
import ser._internal.models.dataset_splitting as dataset_splitting  # noqa: TID251
import ser._internal.models.fast_training as fast_training  # noqa: TID251
import ser._internal.models.profile_training_preparation as profile_preparation  # noqa: TID251
import ser._internal.models.training_entrypoints as training_entrypoints  # noqa: TID251
import ser._internal.models.training_orchestration as orchestration  # noqa: TID251
from ser.__main__ import (
    _build_main_parser,
    _training_operation_from_args,
    _validate_training_mode_args,
)
from ser._internal.config.schema import (  # noqa: TID251
    DataLoaderConfig,
    DatasetConfig,
    ModelsConfig,
    TrainingConfig,
)
from ser._internal.data.dataset_registry import DatasetRegistryEntry  # noqa: TID251
from ser._internal.data.manifest import Utterance  # noqa: TID251
from ser._internal.models.medium_noise_controls import (  # noqa: TID251
    MediumNoiseControlStats,
    merge_medium_noise_stats,
)
from ser._internal.models.training_orchestration import (  # noqa: TID251
    training_operation_scope,
)
from ser._internal.models.training_readiness import (  # noqa: TID251
    PreparedPlanError,
    QuarantineBudgetExceeded,
    ReadinessReport,
    TrainingMode,
    TrainingOperation,
    TrainingReadinessError,
)
from ser._internal.models.training_support import WindowMeta  # noqa: TID251
from ser._internal.repr import FeatureBackend  # noqa: TID251
from ser._internal.repr import EncodedSequence, PoolingWindow  # noqa: TID251
from ser._internal.runtime.commands import classify_training_exception  # noqa: TID251
from ser._internal.utils.audio_utils import AudioDecodeError  # noqa: TID251
from ser.config import AppConfig
from ser.diagnostics.domain import DiagnosticFinding


def _settings() -> AppConfig:
    return AppConfig(emotions={"01": "calm", "02": "happy"})


def test_cli_parses_each_training_mode_and_prepared_plan(tmp_path: Path) -> None:
    parser = _build_main_parser(_settings())
    dry_args = parser.parse_args(["--train", "--dry-run"])
    prepare_args = parser.parse_args(["--train", "--prepare-only", "--repair"])
    plan_args = parser.parse_args(["--train", "--prepared-plan", str(tmp_path / "plan.json")])
    _validate_training_mode_args(parser, dry_args)
    _validate_training_mode_args(parser, prepare_args)
    _validate_training_mode_args(parser, plan_args)
    assert _training_operation_from_args(dry_args).mode is TrainingMode.DRY_RUN
    assert _training_operation_from_args(prepare_args) == TrainingOperation(
        mode=TrainingMode.PREPARE_ONLY,
        repair=True,
    )
    assert _training_operation_from_args(plan_args).prepared_plan == tmp_path / "plan.json"


@pytest.mark.parametrize(
    "argv",
    [
        ["--dry-run"],
        ["--train", "--repair"],
        ["--train", "--dry-run", "--prepared-plan", "plan.json"],
    ],
)
def test_cli_rejects_train_mode_conflicts_before_execution(argv: list[str]) -> None:
    parser = _build_main_parser(_settings())
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as raised:
        _validate_training_mode_args(parser, args)
    assert raised.value.code == 2


def test_argparse_rejects_dry_run_and_prepare_only_together() -> None:
    parser = _build_main_parser(_settings())
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--train", "--dry-run", "--prepare-only"])
    assert raised.value.code == 2


def test_dry_run_returns_after_shared_checker_before_profile_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        training_entrypoints._training_orchestration,
        "ensure_entrypoint_readiness",
        lambda **_kwargs: calls.append("checked"),
    )
    monkeypatch.setattr(
        training_entrypoints._fast_training_entrypoints,
        "train_model",
        lambda **_kwargs: calls.append("trained"),
    )
    with training_operation_scope(TrainingOperation(mode=TrainingMode.DRY_RUN)):
        training_entrypoints.train_model(settings=_settings())
    assert calls == ["checked"]


def test_real_training_invokes_shared_checker_before_profile_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        training_entrypoints._training_orchestration,
        "ensure_entrypoint_readiness",
        lambda **_kwargs: calls.append("checked"),
    )
    monkeypatch.setattr(
        training_entrypoints._fast_training_entrypoints,
        "train_model",
        lambda **_kwargs: calls.append("trained"),
    )
    with training_operation_scope(TrainingOperation()):
        training_entrypoints.train_model(settings=_settings())
    assert calls == ["checked", "trained"]


def test_readiness_loader_never_rebuilds_missing_manifest_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    entry = DatasetRegistryEntry(
        dataset_id="fixture",
        dataset_root=tmp_path / "dataset",
        manifest_path=tmp_path / "missing.jsonl",
        options={},
    )
    monkeypatch.setattr(data_loader, "load_dataset_registry", lambda **_kwargs: {"fixture": entry})
    monkeypatch.setattr(data_loader, "validate_registered_dataset_integrity", lambda _entry: None)
    prepare_called = False

    def _prepare(**_kwargs: object) -> tuple[Path, ...]:
        nonlocal prepare_called
        prepare_called = True
        return ()

    monkeypatch.setattr(data_loader, "prepare_from_registry_entry", _prepare)
    with pytest.raises(RuntimeError, match="missing its manifest"):
        data_loader.load_utterances(settings=settings, allow_prepare=False)
    assert prepare_called is False


def test_backend_smoke_has_a_hard_wall_clock_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SER_TRAINING_SMOKE_TIMEOUT_SECONDS", "0.01")

    def _slow_smoke(**_kwargs: object) -> object:
        time.sleep(1.0)
        raise AssertionError("deadline failed")

    monkeypatch.setattr(orchestration, "_run_selected_backend_smoke_unbounded", _slow_smoke)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="hard wall-clock timeout"):
        orchestration.run_selected_backend_smoke(
            settings=_settings(),
            samples=(),
            probe_cache_dir=tmp_path,
        )
    assert time.perf_counter() - started < 0.5


def test_backend_smoke_rejects_worker_without_hard_deadline(tmp_path: Path) -> None:
    failures: list[BaseException] = []

    def _run() -> None:
        try:
            orchestration.run_selected_backend_smoke(
                settings=_settings(),
                samples=(),
                probe_cache_dir=tmp_path,
            )
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=_run)
    worker.start()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "hard-deadline" in str(failures[0])


def test_model_revision_must_come_from_checked_backend() -> None:
    class _ResolvedBackend:
        backend_id = "hf_xlsr"
        feature_dim = 3
        model_revision = "commit-a"

    with training_operation_scope(TrainingOperation()) as state:
        state.checked_backend = cast(FeatureBackend, _ResolvedBackend())
        state.checked_backend_id = "hf_xlsr"
        state.checked_model_id = "repo/model"
        assert (
            orchestration._resolved_model_revision(backend_id="hf_xlsr", model_id="repo/model")
            == "commit-a"
        )
        _ResolvedBackend.model_revision = "repo/model"
        with pytest.raises(PreparedPlanError, match="unpinned"):
            orchestration._resolved_model_revision(backend_id="hf_xlsr", model_id="repo/model")


def test_transient_local_io_retry_is_bounded_and_accounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    monkeypatch.setattr(orchestration.time, "sleep", lambda _seconds: None)

    def _operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise InterruptedError("transient read")
        return "ok"

    with training_operation_scope(TrainingOperation()) as state:
        assert orchestration.bounded_retry_local_io(_operation, identity="sample") == "ok"
        assert state.bounded_retries == 2
    attempts = 0
    with training_operation_scope(TrainingOperation()):
        with pytest.raises(InterruptedError):
            orchestration.bounded_retry_local_io(
                _operation,
                identity="sample",
                max_retries=1,
            )


def test_transient_errno_is_retried_but_resource_exhaustion_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestration.time, "sleep", lambda _seconds: None)
    attempts = 0

    def _transient() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError(errno.EAGAIN, "temporarily unavailable")
        return "ok"

    with training_operation_scope(TrainingOperation()) as state:
        assert orchestration.bounded_retry_local_io(_transient, identity="sample") == "ok"
        assert state.bounded_retries == 1
        assert state.containment_counts == {"sample:media_decode_failed:bounded_retry": 1}
        with pytest.raises(OSError) as error:
            orchestration.bounded_retry_local_io(
                lambda: (_ for _ in ()).throw(OSError(errno.ENOSPC, "disk full")),
                identity="sample",
            )
    assert error.value.errno == errno.ENOSPC


def test_encoding_quarantine_rechecks_canonical_dev_support(tmp_path: Path) -> None:
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        data_loader=DataLoaderConfig(
            max_failed_files=12,
            max_failed_file_ratio=1.0,
            max_failed_file_ratio_per_corpus=1.0,
            max_failed_file_ratio_per_class=1.0,
            max_failures_per_reason=12,
            min_remaining_per_class_split=1,
        ),
    )
    samples = tuple(
        Utterance(
            1,
            f"sample-{index}",
            "fixture",
            tmp_path / f"{index}.wav",
            "calm" if index % 2 == 0 else "happy",
        )
        for index in range(12)
    )
    train, _, _, _ = dataset_splitting.split_utterances_three_way(
        samples=list(samples),
        settings=settings,
        logger=orchestration.logger,
    )
    train_support = {
        label: [item for item in train if item.label == label] for label in ("calm", "happy")
    }
    safe_candidate = next(items[0] for items in train_support.values() if len(items) > 1)

    def _encode(item: Utterance) -> EncodedSequence:
        if item.sample_id == safe_candidate.sample_id:
            raise AudioDecodeError("isolated decoder failure")
        return cast(EncodedSequence, object())

    def _windows(
        _encoded: EncodedSequence,
        _window_size: float,
        _window_stride: float,
    ) -> list[PoolingWindow]:
        return []

    def _pool(
        _encoded: EncodedSequence,
        _windows: Sequence[PoolingWindow],
    ) -> np.ndarray:
        return np.asarray([[0.1, 0.2]], dtype=np.float64)

    def _apply(features: np.ndarray) -> tuple[np.ndarray, MediumNoiseControlStats]:
        return features, MediumNoiseControlStats(total_windows=1, kept_windows=1)

    def _prepare() -> tuple[np.ndarray, list[str], list[WindowMeta], MediumNoiseControlStats]:
        return profile_preparation.build_medium_feature_dataset(
            utterances=orchestration.current_training_state().utterances,
            encode_sequence=_encode,
            window_size_seconds=1.0,
            window_stride_seconds=1.0,
            build_pooling_windows=_windows,
            pool_features=_pool,
            apply_noise_controls=_apply,
            merge_noise_stats=merge_medium_noise_stats,
            initial_noise_stats=MediumNoiseControlStats(),
            window_meta_factory=WindowMeta,
            handle_sample_failure=lambda item, error: orchestration.handle_sample_encoding_failure(
                settings=settings,
                sample=item,
                error=error,
            ),
        )

    with training_operation_scope(TrainingOperation()) as state:
        state.utterances = samples
        matrix, labels, metadata, stats = orchestration.prepare_until_quarantine_stable(
            settings=settings,
            prepare=_prepare,
        )
        assert safe_candidate.sample_id not in {item.sample_id for item in state.utterances}
        assert matrix.shape[0] == len(samples) - 1
        assert len(labels) == len(metadata) == len(samples) - 1
        assert stats.kept_windows == len(samples) - 1
        fit_calls = 0

        class _FitProbe:
            def fit(self, rows: np.ndarray, targets: list[str]) -> None:
                nonlocal fit_calls
                assert rows.shape[0] == len(targets) == len(samples) - 1
                fit_calls += 1

        _FitProbe().fit(matrix, labels)
        assert fit_calls == 1

    with training_operation_scope(TrainingOperation()) as state:
        minimal_samples = samples[:6]
        state.utterances = minimal_samples
        failing_sample = minimal_samples[0]
        blocked_fit_calls = 0

        def _fail_minimal(item: Utterance) -> EncodedSequence:
            if item.sample_id == failing_sample.sample_id:
                raise AudioDecodeError("isolated decoder failure")
            return cast(EncodedSequence, object())

        with pytest.raises(QuarantineBudgetExceeded, match="Projected (dev split|canonical split)"):
            profile_preparation.build_medium_feature_dataset(
                utterances=state.utterances,
                encode_sequence=_fail_minimal,
                window_size_seconds=1.0,
                window_stride_seconds=1.0,
                build_pooling_windows=_windows,
                pool_features=_pool,
                apply_noise_controls=_apply,
                merge_noise_stats=merge_medium_noise_stats,
                initial_noise_stats=MediumNoiseControlStats(),
                window_meta_factory=WindowMeta,
                handle_sample_failure=lambda item, error: (
                    orchestration.handle_sample_encoding_failure(
                        settings=settings,
                        sample=item,
                        error=error,
                    )
                ),
            )
        assert blocked_fit_calls == 0

    with training_operation_scope(TrainingOperation()) as state:
        state.utterances = samples

        def _systematic_failure(_item: Utterance) -> EncodedSequence:
            raise RuntimeError("backend contract failure")

        with pytest.raises(RuntimeError, match="backend contract failure"):
            profile_preparation.build_medium_feature_dataset(
                utterances=state.utterances,
                encode_sequence=_systematic_failure,
                window_size_seconds=1.0,
                window_stride_seconds=1.0,
                build_pooling_windows=_windows,
                pool_features=_pool,
                apply_noise_controls=_apply,
                merge_noise_stats=merge_medium_noise_stats,
                initial_noise_stats=MediumNoiseControlStats(),
                window_meta_factory=WindowMeta,
                handle_sample_failure=lambda item, error: (
                    orchestration.handle_sample_encoding_failure(
                        settings=settings,
                        sample=item,
                        error=error,
                    )
                ),
            )
        assert blocked_fit_calls == 0


def test_checked_backend_is_reused_once_and_closed_at_scope_exit() -> None:
    closed = 0
    builds = 0

    class _Backend:
        backend_id = "fixture"
        feature_dim = 3

        def close(self) -> None:
            nonlocal closed
            closed += 1

    backend = cast(FeatureBackend, _Backend())

    def _build() -> FeatureBackend:
        nonlocal builds
        builds += 1
        return backend

    with training_operation_scope(TrainingOperation()):
        first = orchestration.reuse_checked_backend(
            backend_id="fixture",
            model_id="model@revision",
            device="cpu",
            dtype="float32",
            build=_build,
        )
        second = orchestration.reuse_checked_backend(
            backend_id="fixture",
            model_id="model@revision",
            device="cpu",
            dtype="float32",
            build=_build,
        )
        assert first is second
    assert builds == 1
    assert closed == 1


def test_fast_preparation_consumes_exactly_the_checked_utterance_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    checked = [
        Utterance(1, "calm-1", "fixture", tmp_path / "one.wav", "calm"),
        Utterance(1, "calm-2", "fixture", tmp_path / "two.wav", "calm"),
        Utterance(1, "happy-1", "fixture", tmp_path / "three.wav", "happy"),
        Utterance(1, "happy-2", "fixture", tmp_path / "four.wav", "happy"),
    ]
    reads: list[str] = []

    def _read(path: str, **_kwargs: object) -> tuple[np.ndarray, int]:
        reads.append(path)
        return np.ones(80, dtype=np.float32), 8000

    class _Backend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def extract_vector(self, *, audio: np.ndarray, sample_rate: int) -> np.ndarray:
            assert sample_rate == 8000
            return np.asarray([float(audio.size)], dtype=np.float64)

    monkeypatch.setattr(data_loader, "read_audio_file", _read)
    monkeypatch.setattr(data_loader, "HandcraftedBackend", _Backend)
    split = data_loader.load_checked_fast_data(utterances=checked, settings=settings)
    assert split is not None
    x_train, x_test, y_train, y_test = split
    assert x_train.shape[0] + x_test.shape[0] == len(checked)
    assert sorted(reads) == sorted(str(item.audio_path) for item in checked)
    assert sorted([*y_train, *y_test]) == ["calm", "calm", "happy", "happy"]


def test_operation_level_plan_validation_recomputes_trusted_expectations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        models=ModelsConfig(folder=tmp_path / "models", model_cache_dir=tmp_path / "cache"),
    )
    utterances = tuple(
        Utterance(
            1,
            f"sample-{index}",
            "fixture",
            tmp_path / f"{index}.wav",
            "calm" if index % 2 == 0 else "happy",
        )
        for index in range(8)
    )
    readiness = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
        effective_sample_ids=tuple(item.sample_id for item in utterances),
    )
    with training_operation_scope(TrainingOperation(mode=TrainingMode.PREPARE_ONLY)) as state:
        state.readiness = readiness
        state.utterances = utterances
        orchestration.publish_prepared_features(
            settings=settings,
            backend_id="handcrafted",
            model_id="builtin",
            device="cpu",
            dtype="float64",
            utterances=utterances,
            x_train=np.ones((6, 3), dtype=np.float64),
            x_test=np.ones((2, 3), dtype=np.float64),
            y_train=["calm", "happy", "calm", "happy", "calm", "happy"],
            y_test=["calm", "happy"],
            metadata={"split_metadata": {}},
            cache_namespace="fast_features",
            windowing_policy={"strategy": "utterance"},
            noise_statistics={},
        )
    plan_path = settings.tmp_folder / "prepared-training-fast.json"
    plan = orchestration.load_prepared_plan(plan_path)
    assert plan.effective_counts["partition_samples"] == {"train": 4, "test": 2, "dev": 2}
    assert plan.disposition_counts == {
        "included_samples": 8,
        "quarantined_samples": 0,
        "included_windows": 8,
        "dropped_windows": 0,
    }
    for namespace in ("../escape", str(tmp_path / "absolute")):
        forged = replace(plan, cache_namespace=namespace)
        monkeypatch.setattr(
            orchestration,
            "prepared_plan_for_operation",
            lambda _settings, candidate=forged: candidate,
        )
        with training_operation_scope(TrainingOperation()) as state:
            state.readiness = readiness
            state.utterances = utterances
            with pytest.raises(PreparedPlanError, match="namespace"):
                orchestration.validate_operation_plan(
                    settings=settings,
                    backend_id="handcrafted",
                    model_id="builtin",
                    device="cpu",
                    dtype="float64",
                )
    forged_version = replace(plan, cache_version="attacker-controlled")
    monkeypatch.setattr(
        orchestration,
        "prepared_plan_for_operation",
        lambda _settings: forged_version,
    )
    with training_operation_scope(TrainingOperation()) as state:
        state.readiness = readiness
        state.utterances = utterances
        with pytest.raises(PreparedPlanError, match="cache_version"):
            orchestration.validate_operation_plan(
                settings=settings,
                backend_id="handcrafted",
                model_id="builtin",
                device="cpu",
                dtype="float64",
            )

    monkeypatch.setattr(
        orchestration,
        "prepared_plan_for_operation",
        lambda _settings: orchestration.load_prepared_plan(plan_path),
    )
    added_cache = settings.tmp_folder / "fast_features" / "unexpected.npz"
    added_cache.parent.mkdir(parents=True)
    added_cache.write_bytes(b"new cache key")
    with training_operation_scope(TrainingOperation(prepared_plan=plan_path)) as state:
        state.readiness = readiness
        state.utterances = utterances
        with pytest.raises(PreparedPlanError, match="cache_keys"):
            orchestration.validate_operation_plan(
                settings=settings,
                backend_id="handcrafted",
                model_id="builtin",
                device="cpu",
                dtype="float64",
            )


def test_fast_prepare_only_publishes_three_way_plan_without_classifier(
    tmp_path: Path,
) -> None:
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        models=ModelsConfig(folder=tmp_path / "models", model_cache_dir=tmp_path / "cache"),
    )
    utterances = tuple(
        Utterance(
            1,
            f"sample-{index}",
            "fixture",
            tmp_path / f"{index}.wav",
            "calm" if index % 2 == 0 else "happy",
        )
        for index in range(8)
    )
    readiness = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
        effective_sample_ids=tuple(item.sample_id for item in utterances),
    )
    initial_train, initial_test, _ = dataset_splitting.split_utterances(
        samples=list(utterances), settings=settings, logger=orchestration.logger
    )
    split = (
        np.arange(len(initial_train) * 3, dtype=np.float64).reshape(-1, 3),
        np.arange(len(initial_test) * 3, dtype=np.float64).reshape(-1, 3),
        [item.require_label() for item in initial_train],
        [item.require_label() for item in initial_test],
    )
    classifier_calls = 0

    def _create_classifier() -> MLPClassifier:
        nonlocal classifier_calls
        classifier_calls += 1
        return MLPClassifier()

    def _unexpected_evaluation(
        *, y_true: list[str], y_pred: list[str]
    ) -> fast_training.TrainingEvaluationLike:
        raise AssertionError("prepare-only evaluated a classifier")

    def _unexpected_persistence(
        *, model: fast_training.EmotionClassifier, artifact: dict[str, object]
    ) -> fast_training.PersistedArtifactsLike:
        raise AssertionError("prepare-only persisted a model")

    def _ignore_training_report(report: dict[str, object], path: Path) -> None:
        del report, path

    hooks = fast_training.FastTrainingHooks(
        logger=orchestration.logger,
        settings=settings,
        load_utterances=lambda: utterances,
        ensure_dataset_consents_for_training=lambda *, utterances: None,
        load_data=lambda *, test_size: split,
        load_checked_data=lambda _utterances: split,
        create_classifier=_create_classifier,
        evaluate_training_predictions=_unexpected_evaluation,
        build_provenance_metadata=lambda **_kwargs: {},
        build_model_artifact=lambda **_kwargs: {},
        extract_artifact_metadata=lambda _artifact: {},
        persist_model_artifacts=_unexpected_persistence,
        build_training_report=lambda **_kwargs: {},
        persist_training_report=_ignore_training_report,
        default_backend_id="handcrafted",
        default_profile_id="fast",
    )
    with training_operation_scope(TrainingOperation(mode=TrainingMode.PREPARE_ONLY)) as state:
        state.readiness = readiness
        state.utterances = utterances
        fast_training.train_fast_model(hooks=hooks)

    plan = orchestration.load_prepared_plan(settings.tmp_folder / "prepared-training-fast.json")
    with training_operation_scope(
        TrainingOperation(prepared_plan=Path(plan.payload_path))
    ) as state:
        state.readiness = readiness
        state.utterances = utterances
        payload = orchestration.read_prepared_feature_payload(plan)
    assert classifier_calls == 0
    assert (payload.x_train.shape[0], payload.x_dev.shape[0], payload.x_test.shape[0]) == (
        4,
        2,
        2,
    )

    class _Evaluation:
        accuracy = 1.0
        macro_f1 = 1.0
        uar = 1.0
        ser_metrics: dict[str, object] = {}

    class _Persisted:
        pickle_path = tmp_path / "model.pkl"
        secure_path: Path | None = None

    captures: list[tuple[np.ndarray, list[str]]] = []

    class _Classifier:
        def fit(self, x_rows: np.ndarray, labels: list[str]) -> None:
            captures.append((np.asarray(x_rows), list(labels)))

        def predict(self, x_rows: np.ndarray) -> np.ndarray:
            return np.asarray(["calm"] * len(x_rows))

    def _training_hooks() -> fast_training.FastTrainingHooks:
        return replace(
            hooks,
            create_classifier=lambda: cast(
                fast_training.EmotionClassifier,
                _Classifier(),
            ),
            evaluate_training_predictions=lambda *, y_true, y_pred: _Evaluation(),
            build_provenance_metadata=lambda **_kwargs: {},
            build_model_artifact=lambda **_kwargs: {},
            extract_artifact_metadata=lambda _artifact: {},
            persist_model_artifacts=lambda **_kwargs: _Persisted(),
            build_training_report=lambda **_kwargs: {},
        )

    with training_operation_scope(TrainingOperation()) as state:
        state.readiness = readiness
        state.utterances = utterances
        fast_training.train_fast_model(hooks=_training_hooks())
    plan_path = settings.tmp_folder / "prepared-training-fast.json"
    with training_operation_scope(TrainingOperation(prepared_plan=plan_path)) as state:
        state.readiness = readiness
        state.utterances = utterances
        fast_training.train_fast_model(hooks=_training_hooks())

    assert len(captures) == 2
    np.testing.assert_array_equal(captures[0][0], captures[1][0])
    assert captures[0][1] == captures[1][1]


def test_prepared_plan_matches_full_fresh_partition_and_noise_provenance(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        models=ModelsConfig(folder=tmp_path / "models", model_cache_dir=tmp_path / "cache"),
    )
    utterances = tuple(
        Utterance(
            1,
            f"sample-{index}",
            "fixture",
            tmp_path / f"{index}.wav",
            "calm" if index % 2 == 0 else "happy",
        )
        for index in range(8)
    )
    readiness = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
        effective_sample_ids=tuple(item.sample_id for item in utterances),
    )
    initial_train, initial_test, split_metadata = dataset_splitting.split_utterances(
        samples=list(utterances), settings=settings, logger=orchestration.logger
    )
    x_train = np.arange(len(initial_train) * 3, dtype=np.float64).reshape(-1, 3)
    x_test = np.arange(len(initial_test) * 3, dtype=np.float64).reshape(-1, 3)
    y_train = [item.require_label() for item in initial_train]
    y_test = [item.require_label() for item in initial_test]
    train_meta = [
        {"sample_id": item.sample_id, "corpus": item.corpus, "language": "en"}
        for item in initial_train
    ]
    with training_operation_scope(TrainingOperation()) as state:
        state.utterances = utterances
        fresh_x, fresh_y, fresh_meta, fresh_utterances = orchestration.canonical_train_partition(
            settings=settings,
            x_train=x_train,
            y_train=y_train,
            train_metadata=train_meta,
            sample_id=lambda item: item["sample_id"],
        )
    fresh_noise = MediumNoiseControlStats(
        total_windows=len(fresh_meta) + 2,
        kept_windows=len(fresh_meta),
        dropped_low_std_windows=1,
        dropped_cap_windows=1,
    )
    test_noise = MediumNoiseControlStats(
        total_windows=len(initial_test),
        kept_windows=len(initial_test),
    )
    caplog.set_level(logging.INFO, logger=orchestration.logger.name)
    with training_operation_scope(TrainingOperation(mode=TrainingMode.PREPARE_ONLY)) as state:
        state.readiness = readiness
        state.utterances = utterances
        orchestration.publish_prepared_features(
            settings=settings,
            backend_id="handcrafted",
            model_id="builtin",
            device="cpu",
            dtype="float64",
            utterances=utterances,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            metadata={
                "split_metadata": split_metadata,
                "train_meta": train_meta,
                "train_noise_stats": fresh_noise,
                "test_noise_stats": test_noise,
            },
            cache_namespace="fast_features",
            windowing_policy={"strategy": "fixture"},
            noise_statistics={"train": fresh_noise, "test": test_noise},
        )
    loaded_plan = orchestration.load_prepared_plan(
        settings.tmp_folder / "prepared-training-fast.json"
    )
    with training_operation_scope(
        TrainingOperation(prepared_plan=settings.tmp_folder / "prepared-training-fast.json")
    ) as state:
        state.readiness = readiness
        state.utterances = utterances
        payload = orchestration.read_prepared_feature_payload(loaded_plan)

    np.testing.assert_array_equal(payload.x_train, fresh_x)
    assert payload.y_train == fresh_y
    assert payload.metadata["train_meta"] == fresh_meta
    assert payload.metadata["train_sample_ids"] == [item.sample_id for item in fresh_utterances]
    expected_noise = orchestration.serialize_metadata(fresh_noise)
    assert payload.metadata["train_noise_stats"] == expected_noise
    assert loaded_plan.noise_statistics["train"] == expected_noise
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("PREPARED_PUBLISH_START") for message in messages)
    assert any(message.startswith("PREPARED_PUBLISH_DONE") for message in messages)


def test_prepared_payload_validation_rejects_semantic_shape_tampering(tmp_path: Path) -> None:
    settings = AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        training=TrainingConfig(test_size=0.25, dev_size=0.25),
        models=ModelsConfig(folder=tmp_path / "models", model_cache_dir=tmp_path / "cache"),
    )
    utterances = tuple(
        Utterance(
            1,
            f"sample-{index}",
            "fixture",
            tmp_path / f"{index}.wav",
            "calm" if index % 2 == 0 else "happy",
        )
        for index in range(8)
    )
    readiness = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
        effective_sample_ids=tuple(item.sample_id for item in utterances),
    )
    with training_operation_scope(TrainingOperation(mode=TrainingMode.PREPARE_ONLY)) as state:
        state.readiness = readiness
        state.utterances = utterances
        plan = orchestration.publish_prepared_features(
            settings=settings,
            backend_id="handcrafted",
            model_id="builtin",
            device="cpu",
            dtype="float64",
            utterances=utterances,
            x_train=np.ones((6, 3), dtype=np.float64),
            x_test=np.ones((2, 3), dtype=np.float64),
            y_train=["calm", "happy", "calm", "happy", "calm", "happy"],
            y_test=["calm", "happy"],
            metadata={"split_metadata": {}},
            cache_namespace="fast_features",
            windowing_policy={"strategy": "utterance"},
            noise_statistics={},
        )
        payload_path = Path(plan.payload_path)
        with payload_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                x_train=np.ones((2, 4), dtype=np.float64),
                x_dev=np.ones((2, 4), dtype=np.float64),
                x_test=np.ones((2, 4), dtype=np.float64),
                y_train=np.asarray(["calm", "happy"]),
                y_dev=np.asarray(["calm", "happy"]),
                y_test=np.asarray(["calm", "happy"]),
                metadata_json=np.asarray("{}"),
            )
        tampered = replace(plan, payload_digest=orchestration.hash_file(payload_path))
        tampered = replace(
            tampered,
            overall_digest=orchestration.digest_payload(tampered.unsigned_dict()),
        )
        with pytest.raises(PreparedPlanError, match="planned shape"):
            orchestration.read_prepared_feature_payload(tampered)


def test_doctor_command_path_includes_the_complete_training_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics_service,
        "_check_complete_training_readiness",
        lambda **_kwargs: (
            DiagnosticFinding(
                code="training_readiness_fixture",
                severity="info",
                message="checked",
            ),
        ),
    )
    report = diagnostics_service.run_doctor_diagnostics(
        settings=_settings(),
        include_transcription_checks=False,
        include_noise_findings=False,
        include_training_readiness=True,
    )
    assert any(item.code == "training_readiness_fixture" for item in report.findings)


def test_training_exit_codes_distinguish_validation_from_internal_failure() -> None:
    validation_report = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
    )
    validation = classify_training_exception(TrainingReadinessError(validation_report))
    internal = classify_training_exception(RuntimeError("unexpected invariant defect"))
    assert validation.exit_code == 2
    assert validation.include_traceback is False
    assert internal.exit_code == 1
    assert internal.include_traceback is True


def test_robustness_statistics_are_exposed_for_artifact_and_report_provenance() -> None:
    readiness = ReadinessReport(
        1,
        "2026-01-01T00:00:00+00:00",
        "fast",
        "settings",
        "registry",
        "manifest",
        "media",
        (),
        (),
    )
    with training_operation_scope(TrainingOperation()) as state:
        state.readiness = readiness
        state.cache_hits = 3
        state.cache_misses = 2
        state.recomputed_cache_entries = 1
        state.dropped_windows = 4
        provenance = orchestration.build_training_robustness_provenance()
    assert provenance["readiness"] == {
        "schema_version": 1,
        "settings_digest": "settings",
        "registry_digest": "registry",
        "manifest_digest": "manifest",
        "media_digest": "media",
    }
    assert provenance["statistics"] == {
        "quarantined_samples": 0,
        "dropped_windows": 4,
        "cache_hits": 3,
        "cache_misses": 2,
        "recomputed_cache_entries": 1,
        "bounded_retries": 0,
        "containment": {},
    }
