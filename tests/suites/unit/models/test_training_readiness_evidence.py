"""Executable evidence that readiness failures cannot reach model fitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given
from hypothesis import strategies as st

from ser._internal.config.schema import (  # noqa: TID251
    DataLoaderConfig,
    DatasetConfig,
    ModelsConfig,
)
from ser._internal.data.manifest import SplitName, Utterance  # noqa: TID251
from ser._internal.models import training_readiness as readiness  # noqa: TID251
from ser._internal.models.training_readiness import (  # noqa: TID251
    CacheEntryCorruptError,
    FailureDisposition,
    FailureReasonCode,
    FailureScope,
    OptionalArtifactError,
    QuarantineBudgetExceeded,
    QuarantinePolicy,
    SmokeResult,
    TrainingReadinessError,
    WindowContainmentError,
    build_quarantine_record,
    classify_failure,
    enforce_quarantine_budget,
    ensure_training_ready,
    normalized_pcm_digest,
)
from ser._internal.utils.audio_utils import AudioDecodeError  # noqa: TID251
from ser.config import AppConfig


def _settings(tmp_path: Path) -> AppConfig:
    """Builds an isolated readiness configuration with explicit budgets."""
    return AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        data_loader=DataLoaderConfig(
            max_workers=1,
            max_failed_file_ratio=0.5,
            max_failed_files=4,
            max_failed_file_ratio_per_corpus=0.5,
            max_failed_file_ratio_per_class=0.5,
            max_failures_per_reason=2,
            min_remaining_per_class_split=1,
        ),
        models=ModelsConfig(
            folder=tmp_path / "models",
            model_cache_dir=tmp_path / "cache",
        ),
    )


def _utterance(
    tmp_path: Path,
    sample_id: str,
    label: str,
    *,
    value: float,
    split: SplitName | None = None,
) -> Utterance:
    """Writes one deterministic tiny WAV and returns its manifest row."""
    path = tmp_path / f"{sample_id}.wav"
    sf.write(path, np.full(80, value, dtype=np.float32), 8_000, subtype="PCM_16")
    return Utterance(
        schema_version=1,
        sample_id=sample_id,
        corpus="fixture",
        audio_path=path,
        label=label,
        language="en",
        split=split,
    )


def _inventory(tmp_path: Path) -> list[Utterance]:
    """Returns a minimal feasible unsplit two-class inventory."""
    return [
        _utterance(tmp_path, f"calm-{index}", "calm", value=0.10 + index / 100)
        for index in range(4)
    ] + [
        _utterance(tmp_path, f"happy-{index}", "happy", value=0.20 + index / 100)
        for index in range(4)
    ]


def _native_inventory(tmp_path: Path) -> list[Utterance]:
    """Returns complete train/dev/test partitions with both classes represented."""
    rows: list[Utterance] = []
    for partition_index, split in enumerate(("train", "dev", "test")):
        typed_split = cast(SplitName, split)
        rows.extend(
            (
                _utterance(
                    tmp_path,
                    f"{split}-calm",
                    "calm",
                    value=0.10 + partition_index / 100,
                    split=typed_split,
                ),
                _utterance(
                    tmp_path,
                    f"{split}-happy",
                    "happy",
                    value=0.20 + partition_index / 100,
                    split=typed_split,
                ),
            )
        )
    return rows


class _ClassifierTrap:
    """Records forbidden classifier construction and fit calls."""

    def __init__(self, calls: list[str]) -> None:
        calls.append("classifier")
        self._calls = calls

    def fit(self, features: object, labels: object) -> None:
        """Records a forbidden fit boundary."""
        del features, labels
        self._calls.append("fit")


def _assert_blocked_before_fit(
    *,
    settings: AppConfig,
    load_utterances: Callable[[], list[Utterance] | None],
    expected_reason: FailureReasonCode,
    smoke_error: Exception | None = None,
    expected_smoke_calls: int = 0,
) -> None:
    """Asserts mandatory readiness raises before classifier construction or fit."""
    smoke_calls: list[tuple[str, ...]] = []
    training_calls: list[str] = []

    def _smoke(
        *,
        settings: AppConfig,
        samples: Sequence[Utterance],
        probe_cache_dir: Path,
    ) -> SmokeResult:
        del settings, probe_cache_dir
        smoke_calls.append(tuple(item.sample_id for item in samples))
        if smoke_error is not None:
            raise smoke_error
        return SmokeResult(
            attempted=len(samples),
            succeeded=len(samples),
            feature_dim=4,
            cache_round_trip=True,
            backend_id="stub",
            model_id="stub-model",
            device="cpu",
            dtype="float32",
        )

    with pytest.raises(TrainingReadinessError) as captured:
        ensure_training_ready(
            settings=settings,
            load_utterances=load_utterances,
            smoke_runner=_smoke,
        )
        classifier = _ClassifierTrap(training_calls)
        classifier.fit(object(), object())

    assert training_calls == []
    assert len(smoke_calls) == expected_smoke_calls
    assert any(
        finding.blocking and finding.reason_code is expected_reason
        for finding in captured.value.report.findings
    )


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        pytest.param("backend", FailureReasonCode.BACKEND_UNAVAILABLE, id="missing-backend"),
        pytest.param("model", FailureReasonCode.BACKEND_UNAVAILABLE, id="missing-model"),
        pytest.param(
            "numeric", FailureReasonCode.INVALID_CONFIGURATION, id="invalid-numeric-config"
        ),
        pytest.param("manifest", FailureReasonCode.MANIFEST_INVALID, id="invalid-manifest"),
        pytest.param("media", FailureReasonCode.MEDIA_NOT_REGULAR, id="invalid-media"),
        pytest.param("output", FailureReasonCode.OUTPUT_UNWRITABLE, id="unwritable-output"),
        pytest.param("disk", FailureReasonCode.DISK_SPACE_LOW, id="low-disk"),
    ],
)
def test_preflight_failures_block_before_classifier_and_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected_reason: FailureReasonCode,
) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)

    def _load_samples() -> list[Utterance]:
        return samples

    loader: Callable[[], list[Utterance] | None] = _load_samples
    smoke_error: Exception | None = None
    expected_smoke_calls = 0

    if case == "backend":
        smoke_error = ImportError("feature backend is not installed")
        expected_smoke_calls = 1
    elif case == "model":
        smoke_error = RuntimeError("model load failed: pinned snapshot is missing")
        expected_smoke_calls = 1
    elif case == "numeric":
        settings = replace(
            settings,
            training=replace(settings.training, test_size=float("nan")),
        )
    elif case == "manifest":

        def _invalid_manifest() -> list[Utterance]:
            raise ValueError("manifest record lacks sample_id")

        loader = _invalid_manifest
    elif case == "media":
        media_directory = tmp_path / "not-a-file"
        media_directory.mkdir()
        samples[0] = replace(samples[0], audio_path=media_directory)
    elif case == "output":

        def _deny_output(_path: Path) -> None:
            raise PermissionError("output is read-only")

        monkeypatch.setattr(readiness, "_probe_directory", _deny_output)  # noqa: SLF001
    elif case == "disk":
        monkeypatch.setattr(
            readiness.shutil,
            "disk_usage",
            lambda _path: SimpleNamespace(total=1, used=1, free=0),
        )
    else:
        raise AssertionError(f"Unhandled test case: {case}")

    _assert_blocked_before_fit(
        settings=settings,
        load_utterances=loader,
        expected_reason=expected_reason,
        smoke_error=smoke_error,
        expected_smoke_calls=expected_smoke_calls,
    )


@pytest.mark.parametrize("identity", ["content", "speaker", "session"])
def test_protected_identity_leakage_blocks_before_classifier_and_fit(
    tmp_path: Path,
    identity: str,
) -> None:
    settings = _settings(tmp_path)
    samples = _native_inventory(tmp_path)

    if identity == "content":
        duplicated_pcm = np.full(80, 0.10, dtype=np.float32)
        sf.write(samples[2].audio_path, duplicated_pcm, 8_000, subtype="PCM_16")
        digest = normalized_pcm_digest(samples[0].audio_path)
        samples[0] = replace(samples[0], normalized_audio_sha256=digest)
        samples[2] = replace(samples[2], normalized_audio_sha256=digest)
    elif identity == "speaker":
        samples[0] = replace(samples[0], speaker_id="fixture:speaker-leak")
        samples[2] = replace(samples[2], speaker_id="fixture:speaker-leak")
    elif identity == "session":
        samples[0] = replace(samples[0], session_id="fixture:session-leak")
        samples[2] = replace(samples[2], session_id="fixture:session-leak")
    else:
        raise AssertionError(f"Unhandled identity: {identity}")

    _assert_blocked_before_fit(
        settings=settings,
        load_utterances=lambda: samples,
        expected_reason=(
            FailureReasonCode.DUPLICATE_CONTENT
            if identity == "content"
            else FailureReasonCode.SPLIT_LEAKAGE
        ),
    )


@pytest.mark.parametrize("case", ["class", "split"])
def test_insufficient_class_or_split_support_blocks_before_fit(
    tmp_path: Path,
    case: str,
) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)[:4]
    if case == "split":
        samples = _native_inventory(tmp_path)
        settings = replace(
            settings,
            data_loader=replace(
                settings.data_loader,
                min_remaining_per_class_split=2,
            ),
        )

    _assert_blocked_before_fit(
        settings=settings,
        load_utterances=lambda: samples,
        expected_reason=FailureReasonCode.INSUFFICIENT_CLASS_SUPPORT,
    )


@given(
    scope=st.sampled_from(list(FailureScope)),
    failure_kind=st.sampled_from(("unknown", "window", "cache", "optional", "decode", "timeout")),
)
def test_failure_dispositions_are_confined_to_exact_typed_scopes(
    scope: FailureScope,
    failure_kind: str,
) -> None:
    error_by_kind: dict[str, Exception] = {
        "unknown": RuntimeError("systemic"),
        "window": WindowContainmentError("low variance"),
        "cache": CacheEntryCorruptError("checksum mismatch"),
        "optional": OptionalArtifactError("optional export failed"),
        "decode": AudioDecodeError("isolated decode failure"),
        "timeout": TimeoutError("transient local read"),
    }
    expected_by_kind: dict[str, tuple[FailureScope, FailureDisposition]] = {
        "window": (FailureScope.WINDOW, FailureDisposition.CONTINUE),
        "cache": (FailureScope.CACHE, FailureDisposition.RECOMPUTE),
        "optional": (FailureScope.OPTIONAL_ARTIFACT, FailureDisposition.CONTINUE),
        "decode": (FailureScope.SAMPLE, FailureDisposition.QUARANTINE),
        "timeout": (FailureScope.SAMPLE, FailureDisposition.BOUNDED_RETRY),
    }

    classified = classify_failure(error_by_kind[failure_kind], scope=scope)
    expected = expected_by_kind.get(failure_kind)
    if expected is not None and scope is expected[0]:
        assert classified.disposition is expected[1]
    else:
        assert classified.disposition is FailureDisposition.ABORT
    if classified.disposition is FailureDisposition.QUARANTINE:
        assert classified.scope is FailureScope.SAMPLE


@given(
    total=st.integers(min_value=1, max_value=50),
    raw_existing=st.integers(min_value=0, max_value=100),
    max_absolute=st.integers(min_value=0, max_value=50),
    global_ratio=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    corpus_ratio=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    class_ratio=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    max_per_reason=st.integers(min_value=0, max_value=50),
    min_remaining=st.integers(min_value=0, max_value=50),
    strict=st.booleans(),
)
def test_quarantine_budget_accepts_exactly_the_conjunction_of_all_limits(
    total: int,
    raw_existing: int,
    max_absolute: int,
    global_ratio: float,
    corpus_ratio: float,
    class_ratio: float,
    max_per_reason: int,
    min_remaining: int,
    strict: bool,
) -> None:
    samples = [
        Utterance(1, f"sample-{index}", "fixture", Path(f"{index}.wav"), "calm")
        for index in range(total)
    ]
    existing_count = raw_existing % total
    classification = classify_failure(
        AudioDecodeError("isolated decode failure"),
        scope=FailureScope.SAMPLE,
    )
    records = [
        build_quarantine_record(
            sample=sample,
            classification=classification,
            occurred_at="2026-01-01T00:00:00+00:00",
            retry_count=0,
        )
        for sample in samples[:existing_count]
    ]
    policy = QuarantinePolicy(
        max_absolute=max_absolute,
        max_global_ratio=global_ratio,
        max_corpus_ratio=corpus_ratio,
        max_class_ratio=class_ratio,
        max_per_reason=max_per_reason,
        min_remaining_per_class_split=min_remaining,
        strict=strict,
    )
    projected = existing_count + 1
    projected_ratio = projected / total
    should_accept = (
        not strict
        and projected <= max_absolute
        and projected_ratio <= global_ratio
        and projected_ratio <= corpus_ratio
        and projected_ratio <= class_ratio
        and projected <= max_per_reason
        and total - projected >= min_remaining
    )

    def _enforce() -> None:
        enforce_quarantine_budget(
            policy=policy,
            all_samples=samples,
            existing_records=records,
            candidate=samples[existing_count],
            classification=classification,
        )

    if should_accept:
        _enforce()
    else:
        with pytest.raises(QuarantineBudgetExceeded):
            _enforce()
