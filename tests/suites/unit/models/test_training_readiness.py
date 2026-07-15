"""Regression contracts for robust training readiness and prepared plans."""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, ModuleType
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
    RuntimeFlags,
)
from ser._internal.data.manifest import SplitName, Utterance  # noqa: TID251
from ser._internal.models.training_readiness import (  # noqa: TID251
    CacheEntryCorruptError,
    FailureDisposition,
    FailureReasonCode,
    FailureScope,
    OptionalArtifactError,
    PreparedPlan,
    PreparedPlanError,
    QuarantineBudgetExceeded,
    QuarantinePolicy,
    ReadinessReport,
    RepairRecord,
    SmokeResult,
    TrainingMode,
    TrainingOperation,
    WindowContainmentError,
    atomic_write_json,
    build_prepared_plan,
    build_quarantine_record,
    canonical_json_bytes,
    classify_failure,
    digest_payload,
    enforce_quarantine_budget,
    load_prepared_plan,
    quarantine_ledger_digest,
    run_training_readiness,
    select_smoke_samples,
    settings_digest,
    validate_prepared_plan,
    write_prepared_plan,
    write_quarantine_ledger,
)
from ser._internal.utils.audio_utils import (  # noqa: TID251
    AudioDecodeError,
    AudioIntegrityError,
)
from ser.config import AppConfig


def _settings(tmp_path: Path, *, ratio: float = 0.5) -> AppConfig:
    return AppConfig(
        emotions={"01": "calm", "02": "happy"},
        tmp_folder=tmp_path / "tmp",
        dataset=DatasetConfig(folder=tmp_path),
        data_loader=DataLoaderConfig(
            max_workers=1,
            max_failed_file_ratio=ratio,
            max_failed_files=4,
            max_failed_file_ratio_per_corpus=ratio,
            max_failed_file_ratio_per_class=ratio,
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
    corpus: str = "fixture",
    split: str | None = None,
    suffix: str = ".wav",
    value: float = 0.1,
) -> Utterance:
    path = tmp_path / f"{sample_id}{suffix}"
    sf.write(path, np.full(800, value, dtype=np.float32), 8000)
    return Utterance(
        schema_version=1,
        sample_id=sample_id,
        corpus=corpus,
        audio_path=path,
        label=label,
        language="en",
        split=cast(SplitName | None, split),
    )


def _inventory(tmp_path: Path) -> list[Utterance]:
    return [
        _utterance(tmp_path, "calm-1", "calm", value=0.10),
        _utterance(tmp_path, "calm-2", "calm", value=0.11),
        _utterance(tmp_path, "calm-3", "calm", value=0.12),
        _utterance(tmp_path, "calm-4", "calm", value=0.13),
        _utterance(tmp_path, "happy-1", "happy", value=0.20),
        _utterance(tmp_path, "happy-2", "happy", value=0.21),
        _utterance(tmp_path, "happy-3", "happy", value=0.22),
        _utterance(tmp_path, "happy-4", "happy", value=0.23),
    ]


def test_training_operation_rejects_unsafe_flag_combinations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repair"):
        TrainingOperation(mode=TrainingMode.TRAIN, repair=True).validate()
    with pytest.raises(ValueError, match="prepared-plan"):
        TrainingOperation(
            mode=TrainingMode.PREPARE_ONLY,
            prepared_plan=tmp_path / "plan.json",
        ).validate()


def test_smoke_selection_is_bounded_deterministic_and_stratified(tmp_path: Path) -> None:
    samples = _inventory(tmp_path)
    segmented = replace(
        samples[0],
        sample_id="calm-segment",
        corpus="other",
        start_seconds=0.0,
        duration_seconds=0.05,
    )
    first = select_smoke_samples([*samples, segmented], cap=3)
    second = select_smoke_samples([segmented, *reversed(samples)], cap=3)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert len(first) == 3
    assert {item.corpus for item in first} == {"fixture", "other"}


def test_valid_tiny_audio_inventory_passes_and_writes_atomic_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    report_path = tmp_path / "reports" / "readiness.json"

    report, resolved = run_training_readiness(
        settings=settings,
        load_utterances=lambda: samples,
        report_path=report_path,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert report.ready is True
    assert resolved == tuple(samples)
    assert report_path.is_file()
    assert not list(report_path.parent.glob(f".{report_path.name}.*"))


def test_readiness_logs_start_phase_and_done_events(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dry-run readiness should emit visible progress before long work returns."""
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    caplog.set_level(logging.INFO, logger="ser._internal.models.training_readiness")

    report, _ = run_training_readiness(settings=settings, load_utterances=lambda: samples)

    messages = [record.getMessage() for record in caplog.records]
    assert report.ready is True
    assert any(message.startswith("READINESS_START") for message in messages)
    assert any("READINESS_PHASE_START phase=manifest_load" in message for message in messages)
    assert any(message.startswith("DATASET_MEDIA_ROOTS_START") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_ROOTS_DONE") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_CHECK_START") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_PROGRESS") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_CHECK_DONE") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_CONTAINMENT_START") for message in messages)
    assert any(message.startswith("DATASET_MEDIA_CONTAINMENT_DONE") for message in messages)
    assert any("READINESS_PHASE_DONE phase=dataset_media" in message for message in messages)
    assert any(message.startswith("INVENTORY_DIGEST_START") for message in messages)
    assert any(message.startswith("INVENTORY_DIGEST_PROGRESS") for message in messages)
    assert any(message.startswith("INVENTORY_DIGEST_DONE") for message in messages)
    assert any(message.startswith("READINESS_DONE") for message in messages)


def test_missing_media_is_contained_before_backend_smoke(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    missing = replace(samples[0], audio_path=tmp_path / "missing.wav")
    smoke_called = False

    def _smoke(**_kwargs: object) -> SmokeResult:
        nonlocal smoke_called
        smoke_called = True
        sample_count = len(cast(tuple[Utterance, ...], _kwargs["samples"]))
        return SmokeResult(
            sample_count,
            sample_count,
            3,
            True,
            "handcrafted",
            "builtin",
            "cpu",
            "float64",
        )

    report, _ = run_training_readiness(
        settings=settings,
        load_utterances=lambda: [missing, *samples[1:]],
        smoke_runner=_smoke,
    )

    assert report.ready is True
    assert smoke_called is True
    assert FailureReasonCode.MEDIA_MISSING in {item.reason_code for item in report.findings}


def test_one_isolated_corrupt_sample_is_quarantined_and_revalidated(tmp_path: Path) -> None:
    settings = _settings(tmp_path, ratio=0.5)
    samples = [
        *_inventory(tmp_path),
        _utterance(tmp_path, "calm-5", "calm", value=0.14),
        _utterance(tmp_path, "happy-5", "happy", value=0.24),
    ]
    samples[0].audio_path.write_bytes(b"not an audio container")

    report, effective = run_training_readiness(
        settings=settings,
        load_utterances=lambda: samples,
    )

    assert report.ready is True
    assert [record.sample_id for record in report.quarantine] == [samples[0].sample_id]
    assert samples[0].sample_id not in {item.sample_id for item in effective}
    assert any(
        finding.reason_code is FailureReasonCode.MEDIA_DECODE_FAILED and not finding.blocking
        for finding in report.findings
    )


def test_dry_run_readiness_keeps_quarantine_only_in_atomic_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    samples[0].audio_path.write_bytes(b"not an audio container")
    report_path = tmp_path / "dry-run-readiness.json"
    report, _ = run_training_readiness(
        settings=settings,
        load_utterances=lambda: samples,
        persist_quarantine_ledger=False,
        report_path=report_path,
    )
    assert report.quarantine
    assert report_path.is_file()
    assert not (settings.tmp_folder / "quarantine-fast.jsonl").exists()


def test_permission_and_resource_os_errors_never_quarantine() -> None:
    for error in (PermissionError("denied"), OSError(28, "disk full")):
        classification = classify_failure(error, scope=FailureScope.SAMPLE)
        assert classification.disposition is FailureDisposition.ABORT


def test_duplicate_pcm_is_detected_across_audio_containers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    pcm = np.arange(-400, 400, dtype=np.int16)
    sf.write(samples[0].audio_path, pcm, 8000, subtype="PCM_16")
    flac_path = tmp_path / "duplicate.flac"
    sf.write(flac_path, pcm, 8000, format="FLAC", subtype="PCM_16")
    samples[1] = replace(samples[1], audio_path=flac_path)

    report, _ = run_training_readiness(settings=settings, load_utterances=lambda: samples)

    duplicates = [
        finding
        for finding in report.findings
        if finding.reason_code is FailureReasonCode.DUPLICATE_CONTENT
    ]
    assert len(duplicates) == 1
    assert duplicates[0].blocking is True


def test_media_path_escape_blocks_before_metadata_or_hash_access(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outside_path = tmp_path.parent / f"{tmp_path.name}-outside.wav"
    sf.write(outside_path, np.ones(80, dtype=np.float32), 8000)
    outside = replace(_utterance(tmp_path, "outside", "calm", value=0.4), audio_path=outside_path)
    samples = [outside, *_inventory(tmp_path)]
    report, _ = run_training_readiness(settings=settings, load_utterances=lambda: samples)
    assert any(
        finding.reason_code is FailureReasonCode.MANIFEST_INVALID and finding.sample_id == "outside"
        for finding in report.findings
    )


def test_smoke_selection_rejects_an_unsatisfiable_strata_cap(tmp_path: Path) -> None:
    samples = [
        _utterance(tmp_path, "a", "calm", corpus="one", value=0.1),
        _utterance(tmp_path, "b", "happy", corpus="two", value=0.2),
    ]
    with pytest.raises(ValueError, match="cannot cover"):
        select_smoke_samples(samples, cap=1)


def test_repair_cleans_owned_staging_and_corrupt_cache_then_revalidates(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    settings.tmp_folder.mkdir(parents=True)
    staging = settings.tmp_folder / ".ser-write-probe-abandoned"
    staging.write_text("probe", encoding="utf-8")
    corrupt_cache = settings.tmp_folder / "medium_embeddings" / "broken.npz"
    corrupt_cache.parent.mkdir(parents=True)
    corrupt_cache.write_bytes(b"broken")
    user_npz = settings.tmp_folder / "user-data" / "important.npz"
    user_npz.parent.mkdir(parents=True)
    user_npz.write_bytes(b"not an application cache")
    caplog.set_level(logging.INFO, logger="ser._internal.models.training_readiness")

    report, _ = run_training_readiness(
        settings=settings,
        load_utterances=lambda: samples,
        repair=True,
    )

    assert report.ready is True
    assert not staging.exists()
    assert not corrupt_cache.exists()
    assert user_npz.read_bytes() == b"not an application cache"
    actions = {record.action for record in report.repairs}
    assert {
        "create_application_directory",
        "clean_application_staging",
        "invalidate_derived_cache",
    } <= actions
    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("REPAIR_START") for message in messages)
    assert any("REPAIR_ACTION action=clean_application_staging" in message for message in messages)
    assert any(message.startswith("REPAIR_DONE") for message in messages)


def test_backend_repair_is_followed_by_complete_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path)
    samples = _inventory(tmp_path)
    attempts = 0

    def _smoke(**_kwargs: object) -> SmokeResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ImportError("backend dependency unavailable")
        sample_count = len(cast(tuple[Utterance, ...], _kwargs["samples"]))
        return SmokeResult(
            sample_count,
            sample_count,
            3,
            True,
            "handcrafted",
            "builtin",
            "cpu",
            "float64",
        )

    monkeypatch.setattr(
        "ser._internal.models.training_readiness._repair_pinned_model",
        lambda _settings: RepairRecord("redownload_pinned_model", "fixture", True, "repaired"),
    )
    report, _ = run_training_readiness(
        settings=settings,
        load_utterances=lambda: samples,
        smoke_runner=_smoke,
        repair=True,
    )
    assert attempts == 2
    assert report.ready is True
    assert any(record.action == "redownload_pinned_model" for record in report.repairs)


def test_pinned_model_repair_rolls_back_atomic_cache_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ser._internal.models.training_readiness as readiness_module  # noqa: TID251

    base = _settings(tmp_path)
    settings = replace(
        base,
        models=replace(base.models, medium_model_id="owner/model@deadbeef"),
        runtime_flags=RuntimeFlags(profile_pipeline=True, medium_profile=True),
    )
    live_cache = settings.models.huggingface_cache_root
    live_cache.mkdir(parents=True)
    sentinel = live_cache / "valid.bin"
    sentinel.write_bytes(b"preserve")
    hub = ModuleType("huggingface_hub")
    errors = ModuleType("huggingface_hub.errors")

    class _HubError(Exception):
        pass

    def _snapshot_download(**kwargs: object) -> str:
        cache_dir = Path(cast(Path, kwargs["cache_dir"]))
        snapshot = cache_dir / "snapshots" / "deadbeef"
        snapshot.mkdir(parents=True)
        (snapshot / "weights.bin").write_bytes(b"new")
        return str(snapshot)

    hub.__dict__["snapshot_download"] = _snapshot_download
    errors.__dict__["HfHubHTTPError"] = _HubError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    real_replace = readiness_module.os.replace

    def _fail_publish(source: object, destination: object) -> None:
        if cast(Path, source).name == "cache":
            raise OSError("publish interruption")
        real_replace(cast(Path, source), cast(Path, destination))

    monkeypatch.setattr(readiness_module.os, "replace", _fail_publish)
    monkeypatch.setenv("SER_TRAINING_REPAIR_ALLOW_NETWORK", "1")
    result = readiness_module._repair_pinned_model(settings)
    assert result.succeeded is False
    assert sentinel.read_bytes() == b"preserve"
    assert not list(live_cache.parent.glob(f".{live_cache.name}.rollback-*"))


def test_git_lfs_pointer_is_systematic_abort() -> None:
    classified = classify_failure(
        AudioIntegrityError("Audio file is an unmaterialized Git LFS pointer"),
        scope=FailureScope.SAMPLE,
    )
    assert classified.scope is FailureScope.CORPUS
    assert classified.reason_code is FailureReasonCode.GIT_LFS_POINTER
    assert classified.disposition is FailureDisposition.ABORT


def test_missing_file_requires_exact_contained_sample_provenance(tmp_path: Path) -> None:
    sample = _utterance(tmp_path, "missing", "calm", value=0.1)
    sample.audio_path.unlink()
    unproven = classify_failure(FileNotFoundError("cache missing"), scope=FailureScope.SAMPLE)
    wrong_path = classify_failure(
        FileNotFoundError(2, "missing", str(tmp_path / "cache.bin")),
        scope=FailureScope.SAMPLE,
        sample=sample,
        allowed_roots=(tmp_path,),
    )
    proven = classify_failure(
        FileNotFoundError(2, "missing", str(sample.audio_path)),
        scope=FailureScope.SAMPLE,
        sample=sample,
        allowed_roots=(tmp_path,),
    )
    assert unproven.disposition is FailureDisposition.ABORT
    assert wrong_path.disposition is FailureDisposition.ABORT
    assert proven.disposition is FailureDisposition.QUARANTINE


def test_quarantine_identity_ignores_audit_timestamps(tmp_path: Path) -> None:
    sample = _inventory(tmp_path)[0]
    classification = classify_failure(
        AudioDecodeError("isolated decoder failure"), scope=FailureScope.SAMPLE
    )
    first = build_quarantine_record(
        sample=sample,
        classification=classification,
        occurred_at="2026-01-01T00:00:00+00:00",
        retry_count=0,
    )
    second = replace(
        first,
        first_occurrence="2026-02-01T00:00:00+00:00",
        last_occurrence="2026-02-01T00:00:01+00:00",
    )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    assert quarantine_ledger_digest([first]) == quarantine_ledger_digest([second])
    assert write_quarantine_ledger(first_path, [first]) == write_quarantine_ledger(
        second_path, [second]
    )
    assert first_path.read_text() != second_path.read_text()


def test_unknown_sample_exception_aborts_with_original_diagnostic() -> None:
    classified = classify_failure(
        ValueError("non-finite backend output"), scope=FailureScope.SAMPLE
    )
    assert classified.disposition is FailureDisposition.ABORT
    assert classified.diagnostic == "non-finite backend output"


def test_typed_non_sample_containment_actions_are_allowlisted_only() -> None:
    assert (
        classify_failure(
            WindowContainmentError("low variance"), scope=FailureScope.WINDOW
        ).disposition
        is FailureDisposition.CONTINUE
    )
    assert (
        classify_failure(
            CacheEntryCorruptError("derived cache"), scope=FailureScope.CACHE
        ).disposition
        is FailureDisposition.RECOMPUTE
    )
    assert (
        classify_failure(
            OptionalArtifactError("advisory export"), scope=FailureScope.OPTIONAL_ARTIFACT
        ).disposition
        is FailureDisposition.CONTINUE
    )
    for scope in (
        FailureScope.WINDOW,
        FailureScope.CACHE,
        FailureScope.OPTIONAL_ARTIFACT,
    ):
        assert classify_failure(RuntimeError("unknown"), scope=scope).disposition is (
            FailureDisposition.ABORT
        )


def test_shared_readiness_denies_restricted_backend_before_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ser._internal.license_check as license_check  # noqa: TID251

    settings = replace(
        _settings(tmp_path),
        runtime_flags=RuntimeFlags(
            profile_pipeline=True,
            accurate_research_profile=True,
            restricted_backends=False,
        ),
    )
    monkeypatch.setattr(license_check, "parse_allowed_restricted_backends_env", frozenset)
    monkeypatch.setattr(license_check, "load_persisted_backend_consents", lambda **_kwargs: {})
    smoke_called = False

    def _smoke(**_kwargs: object) -> SmokeResult:
        nonlocal smoke_called
        smoke_called = True
        raise AssertionError("denied backend must not be constructed")

    report, _ = run_training_readiness(
        settings=settings,
        load_utterances=lambda: _inventory(tmp_path),
        smoke_runner=_smoke,
    )
    assert smoke_called is False
    assert any(finding.check == "restricted_backend_access" for finding in report.findings)


@pytest.mark.parametrize(
    ("policy_override", "message"),
    [
        ({"max_absolute": 0}, "Absolute"),
        ({"max_global_ratio": 0.124}, "Global"),
        ({"max_corpus_ratio": 0.124}, "corpus"),
        ({"max_class_ratio": 0.249}, "class"),
        ({"max_per_reason": 0}, "reason"),
        ({"strict": True}, "Strict"),
        ({"min_remaining_per_class_split": 4}, "remaining"),
    ],
)
def test_every_quarantine_budget_boundary_aborts(
    tmp_path: Path,
    policy_override: dict[str, object],
    message: str,
) -> None:
    samples = _inventory(tmp_path)
    base = QuarantinePolicy(
        max_absolute=4,
        max_global_ratio=0.5,
        max_corpus_ratio=0.5,
        max_class_ratio=0.5,
        max_per_reason=2,
        min_remaining_per_class_split=1,
    )
    policy = QuarantinePolicy(
        max_absolute=cast(int, policy_override.get("max_absolute", base.max_absolute)),
        max_global_ratio=cast(
            float, policy_override.get("max_global_ratio", base.max_global_ratio)
        ),
        max_corpus_ratio=cast(
            float, policy_override.get("max_corpus_ratio", base.max_corpus_ratio)
        ),
        max_class_ratio=cast(float, policy_override.get("max_class_ratio", base.max_class_ratio)),
        max_per_reason=cast(int, policy_override.get("max_per_reason", base.max_per_reason)),
        min_remaining_per_class_split=cast(
            int,
            policy_override.get(
                "min_remaining_per_class_split", base.min_remaining_per_class_split
            ),
        ),
        strict=cast(bool, policy_override.get("strict", base.strict)),
    )
    classification = classify_failure(
        AudioDecodeError("isolated decode"), scope=FailureScope.SAMPLE
    )
    with pytest.raises(QuarantineBudgetExceeded, match=message):
        enforce_quarantine_budget(
            policy=policy,
            all_samples=samples,
            existing_records=[],
            candidate=samples[0],
            classification=classification,
        )


def test_allowlisted_sample_failure_at_exact_budget_is_accepted(tmp_path: Path) -> None:
    samples = _inventory(tmp_path)
    classification = classify_failure(
        AudioDecodeError("isolated decode"), scope=FailureScope.SAMPLE
    )
    enforce_quarantine_budget(
        policy=QuarantinePolicy(
            max_absolute=1,
            max_global_ratio=0.25,
            max_corpus_ratio=0.25,
            max_class_ratio=0.5,
            max_per_reason=1,
            min_remaining_per_class_split=1,
        ),
        all_samples=samples,
        existing_records=[],
        candidate=samples[0],
        classification=classification,
    )
    record = build_quarantine_record(
        sample=samples[0],
        classification=classification,
        occurred_at="2026-01-01T00:00:00+00:00",
        retry_count=0,
    )
    assert record.path_digest != str(samples[0].audio_path)
    assert record.disposition is FailureDisposition.QUARANTINE


@given(
    total=st.integers(min_value=2, max_value=200),
    ratio=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_global_budget_arithmetic_matches_exact_projected_ratio(
    total: int,
    ratio: float,
) -> None:
    samples = [
        Utterance(1, f"sample-{index}", "fixture", Path(f"{index}.wav"), "calm")
        for index in range(total)
    ]
    classification = classify_failure(
        AudioDecodeError("isolated decoder failure"),
        scope=FailureScope.SAMPLE,
    )
    policy = QuarantinePolicy(total, ratio, 1.0, 1.0, total, 0)
    if 1 / total <= ratio:
        enforce_quarantine_budget(
            policy=policy,
            all_samples=samples,
            existing_records=[],
            candidate=samples[0],
            classification=classification,
        )
    else:
        with pytest.raises(QuarantineBudgetExceeded, match="Global"):
            enforce_quarantine_budget(
                policy=policy,
                all_samples=samples,
                existing_records=[],
                candidate=samples[0],
                classification=classification,
            )


@given(scope=st.sampled_from(list(FailureScope)))
def test_unknown_and_unproven_missing_failures_never_become_exclusions(
    scope: FailureScope,
) -> None:
    unknown = classify_failure(RuntimeError("systemic"), scope=scope)
    missing = classify_failure(FileNotFoundError("unproven"), scope=scope)
    assert unknown.disposition is FailureDisposition.ABORT
    assert missing.disposition is FailureDisposition.ABORT


def test_smoke_selection_remains_linear_and_capped() -> None:
    samples = [
        Utterance(1, f"sample-{index:05d}", "fixture", Path(f"{index}.wav"), "calm")
        for index in range(10_000)
    ]
    started = datetime.now(UTC)
    selected = select_smoke_samples(samples, cap=16)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    assert len(selected) == 16
    assert elapsed < 1.0


def _readiness(settings: AppConfig) -> ReadinessReport:
    return ReadinessReport(
        schema_version=1,
        created_at="2026-01-01T00:00:00+00:00",
        profile="fast",
        settings_digest=digest_payload(settings),
        registry_digest="registry",
        manifest_digest="manifest",
        media_digest="media",
        selected_sample_ids=("calm-1",),
        findings=(),
    )


def _build_test_plan(
    settings: AppConfig,
    payload: Path,
    *,
    cache_keys: tuple[str, ...] = ("key",),
    created_at: str | None = None,
) -> PreparedPlan:
    return build_prepared_plan(
        settings=settings,
        readiness=_readiness(settings),
        backend_id="handcrafted",
        model_id="builtin",
        model_revision="builtin",
        device="cpu",
        dtype="float64",
        recipe_digest="recipe",
        split_ledger_digest="split",
        quarantine_ledger_digest="quarantine",
        cache_namespace="features",
        cache_version="v1",
        cache_keys=cache_keys,
        effective_counts={"class": {"calm": 2, "happy": 2}},
        feature_shape=(4, 3),
        feature_dtype="float64",
        windowing_policy={"strategy": "utterance"},
        noise_statistics={},
        payload_path=payload,
        package_version="1.0.0",
        created_at=created_at,
    )


def test_prepared_plan_is_deterministic_atomic_and_digest_verified(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = tmp_path / "features.npz"
    payload.write_bytes(b"safe-features")
    first = _build_test_plan(
        settings,
        payload,
        cache_keys=("b", "a"),
        created_at="2026-01-01T00:00:00+00:00",
    )
    second = _build_test_plan(
        settings,
        payload,
        cache_keys=("b", "a"),
        created_at="2026-01-01T00:00:00+00:00",
    )
    path = tmp_path / "plan.json"
    assert first == second
    assert first.cache_keys == ("a", "b")
    write_prepared_plan(path, first)
    assert load_prepared_plan(path) == first

    raw = path.read_text(encoding="utf-8").replace('"device": "cpu"', '"device": "cuda"')
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(PreparedPlanError, match="digest mismatch"):
        load_prepared_plan(path)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("cache_keys", [1]),
        ("feature_shape", [4, True]),
        ("effective_counts", {"class": {"calm": -1}}),
        ("sample_ledger", ["not-an-object"]),
        ("disposition_counts", {"included": 1.5}),
    ],
)
def test_redigested_malformed_plan_is_rejected_at_typed_boundary(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    settings = _settings(tmp_path)
    payload = tmp_path / "features.npz"
    payload.write_bytes(b"safe-features")
    plan = _build_test_plan(settings, payload)
    raw = plan.to_dict()
    raw[field] = invalid
    unsigned = dict(raw)
    unsigned.pop("overall_digest")
    raw["overall_digest"] = digest_payload(unsigned)
    path = tmp_path / "malformed-plan.json"
    atomic_write_json(path, raw)
    with pytest.raises(PreparedPlanError, match=field):
        load_prepared_plan(path)


def test_interrupted_atomic_publication_leaves_no_ready_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "prepared.json"

    def _fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr("ser._internal.models.training_readiness.os.replace", _fail_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        atomic_write_json(destination, {"ready": True})
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*"))


@pytest.mark.parametrize(
    "field",
    [
        "settings_digest",
        "registry_digest",
        "manifest_digest",
        "media_digest",
        "recipe_digest",
        "split_ledger_digest",
        "quarantine_ledger_digest",
        "backend_id",
        "model_id",
        "model_revision",
        "device",
        "dtype",
        "cache_namespace",
        "cache_version",
        "cache_keys",
        "package_version",
    ],
)
def test_any_relevant_input_change_invalidates_prepared_plan(
    tmp_path: Path,
    field: str,
) -> None:
    settings = _settings(tmp_path)
    payload = tmp_path / "features.npz"
    payload.write_bytes(b"safe-features")
    readiness = _readiness(settings)
    plan = _build_test_plan(settings, payload)
    changed_readiness = replace(
        readiness,
        settings_digest="changed" if field == "settings_digest" else readiness.settings_digest,
        registry_digest="changed" if field == "registry_digest" else readiness.registry_digest,
        manifest_digest="changed" if field == "manifest_digest" else readiness.manifest_digest,
        media_digest="changed" if field == "media_digest" else readiness.media_digest,
    )
    with pytest.raises(PreparedPlanError, match=field):
        validate_prepared_plan(
            plan,
            settings=settings,
            readiness=changed_readiness,
            backend_id="changed" if field == "backend_id" else "handcrafted",
            model_id="changed" if field == "model_id" else "builtin",
            model_revision="changed" if field == "model_revision" else "builtin",
            device="changed" if field == "device" else "cpu",
            dtype="changed" if field == "dtype" else "float64",
            recipe_digest="changed" if field == "recipe_digest" else "recipe",
            split_ledger_digest="changed" if field == "split_ledger_digest" else "split",
            quarantine_ledger_digest=(
                "changed" if field == "quarantine_ledger_digest" else "quarantine"
            ),
            cache_namespace="changed" if field == "cache_namespace" else "features",
            cache_version="changed" if field == "cache_version" else "v1",
            cache_keys=("changed",) if field == "cache_keys" else ("key",),
            package_version="changed" if field == "package_version" else "1.0.0",
        )


def test_settings_digest_serializes_read_only_mapping_config_fields() -> None:
    """Regression: settings_digest must not choke on MappingProxyType config fields.

    AppConfig.emotions is exposed as a read-only MappingProxyType. The digest path
    previously serialized settings via dataclasses.asdict(), whose deep-copy raised
    'TypeError: cannot pickle mappingproxy object'. That crash only surfaced once the
    readiness pass stopped stalling in split_feasibility and reached report assembly.
    """
    # reload_settings() stores emotions as a read-only MappingProxyType; construct it
    # explicitly so the regression does not depend on ambient configuration.
    settings = AppConfig(emotions=MappingProxyType({"01": "calm", "02": "happy"}))
    assert isinstance(settings.emotions, MappingProxyType)  # the crash trigger

    digest = settings_digest(settings)
    assert isinstance(digest, str) and len(digest) == 64
    assert digest == settings_digest(settings)  # deterministic

    # The read-only mapping's contents are actually captured in the canonical payload.
    payload = canonical_json_bytes(settings)
    assert b'"calm"' in payload and b'"happy"' in payload
