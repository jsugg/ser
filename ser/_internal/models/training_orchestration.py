"""Shared training-operation context, backend smoke, and prepared-feature storage."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ser.config import AppConfig
from ser.profiles import resolve_profile_name

from ..data import EmbeddingCache, Utterance  # noqa: TID251
from ..features.feature_extractor import _extract_feature_for_settings  # noqa: TID251
from ..pool import mean_std_pool, temporal_pooling_windows  # noqa: TID251
from ..repr import FeatureBackend  # noqa: TID251
from ..utils.audio_utils import read_audio_file  # noqa: TID251
from ..utils.logger import get_logger  # noqa: TID251
from .feature_runtime_encoding import encode_sequence_with_cache  # noqa: TID251
from .profile_runtime import (  # noqa: TID251
    ACCURATE_BACKEND_ID,
    ACCURATE_RESEARCH_BACKEND_ID,
    MEDIUM_BACKEND_ID,
    build_accurate_backend_for_settings,
    build_accurate_research_backend_for_settings,
    build_medium_backend_for_settings,
    resolve_accurate_model_id,
    resolve_accurate_research_model_id,
    resolve_medium_model_id,
    resolve_runtime_selectors_for_backend_id,
)
from .training_readiness import (  # noqa: TID251
    PREPARATION_CODE_VERSION,
    PREPARED_CACHE_VERSION,
    FailureDisposition,
    FailureScope,
    PreparedPlan,
    PreparedPlanError,
    QuarantineBudgetExceeded,
    QuarantinePolicy,
    QuarantineRecord,
    ReadinessReport,
    SmokeResult,
    TrainingMode,
    TrainingOperation,
    TrainingReadinessError,
    _allowed_media_roots,
    build_prepared_plan,
    build_quarantine_record,
    classify_failure,
    default_prepared_payload_path,
    default_prepared_plan_path,
    digest_payload,
    enforce_quarantine_budget,
    expected_cache_namespace,
    hash_file,
    load_prepared_plan,
    quarantine_ledger_digest,
    run_training_readiness,
    validate_prepared_plan,
    validated_cache_root,
    write_prepared_plan,
    write_quarantine_ledger,
)

logger = get_logger(__name__)
_PreparationT = TypeVar("_PreparationT")


@dataclass(slots=True)
class TrainingRunState:
    """Mutable process-local state shared across one training orchestration."""

    operation: TrainingOperation
    readiness: ReadinessReport | None = None
    utterances: tuple[Utterance, ...] = ()
    quarantine_records: list[QuarantineRecord] = field(default_factory=list)
    checked_backend: FeatureBackend | None = None
    checked_backend_id: str | None = None
    checked_model_id: str | None = None
    checked_device: str | None = None
    checked_dtype: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    recomputed_cache_entries: int = 0
    dropped_windows: int = 0
    bounded_retries: int = 0
    preparation_quarantine_changed: bool = False
    preparation_started_at: float | None = None
    containment_counts: Counter[str] = field(default_factory=Counter)
    medium_noise_stats_by_sample: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreparedFeaturePayload:
    """Safe non-pickle representation of reusable feature preparation."""

    x_train: NDArray[np.float64]
    x_dev: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: list[str]
    y_dev: list[str]
    y_test: list[str]
    metadata: Mapping[str, object]


_DEFAULT_STATE = TrainingRunState(operation=TrainingOperation())
_TRAINING_STATE: ContextVar[TrainingRunState] = ContextVar(
    "ser_training_run_state",
    default=_DEFAULT_STATE,
)


def current_training_state() -> TrainingRunState:
    """Returns the active operation state or the default real-training state."""
    return _TRAINING_STATE.get()


def training_operation_active() -> bool:
    """Returns whether a caller established an explicit invocation scope."""
    return current_training_state() is not _DEFAULT_STATE


def _close_backend(backend: object) -> None:
    """Closes one backend when it exposes a deterministic lifecycle hook."""
    for method_name in ("close", "shutdown"):
        method = getattr(backend, method_name, None)
        if callable(method):
            method()
            return


def close_checked_backend(state: TrainingRunState | None = None) -> None:
    """Releases the checked backend and clears all retained runtime references."""
    active_state = state or current_training_state()
    backend = active_state.checked_backend
    try:
        if backend is not None:
            _close_backend(backend)
    finally:
        active_state.checked_backend = None
        active_state.checked_backend_id = None
        active_state.checked_model_id = None
        active_state.checked_device = None
        active_state.checked_dtype = None


@contextmanager
def training_operation_scope(operation: TrainingOperation) -> Iterator[TrainingRunState]:
    """Activates one validated training operation for nested profile entrypoints."""
    operation.validate()
    state = TrainingRunState(operation=operation)
    token: Token[TrainingRunState] = _TRAINING_STATE.set(state)
    try:
        yield state
    finally:
        close_checked_backend(state)
        _TRAINING_STATE.reset(token)


def _package_version() -> str:
    """Returns installed package version with a source-tree fallback."""
    try:
        return version("ser")
    except PackageNotFoundError:
        return "1.0.0"


def _recipe_digest(settings: AppConfig) -> str:
    """Returns the current recipe content/identifier digest."""
    recipe = settings.dataset.recipe or "none"
    recipe_path = Path(recipe).expanduser()
    return (
        digest_payload(recipe_path.read_text(encoding="utf-8"))
        if recipe_path.is_file()
        else digest_payload(recipe)
    )


def _cache_keys(settings: AppConfig, namespace: str) -> tuple[str, ...]:
    """Returns current cache-relative keys bound to their exact content digests."""
    root = validated_cache_root(settings, namespace)
    return tuple(
        sorted(
            f"{path.relative_to(root)}:{hash_file(path)}"
            for path in root.rglob("*.npz")
            if path.is_file()
        )
    )


def _effective_state_utterances() -> tuple[Utterance, ...]:
    """Returns the invocation view after readiness and preparation quarantine."""
    state = current_training_state()
    excluded = {record.sample_id for record in state.quarantine_records}
    return tuple(item for item in state.utterances if item.sample_id not in excluded)


def _current_split_digest(settings: AppConfig) -> str:
    """Recomputes the deterministic split ledger for current effective samples."""
    from ser._internal.models.dataset_splitting import (  # noqa: TID251
        split_utterances_three_way,
    )

    train, dev, test, metadata = split_utterances_three_way(
        samples=list(_effective_state_utterances()),
        settings=settings,
        logger=logger,
    )
    return digest_payload(
        {
            "metadata": asdict(metadata),
            "train_sample_ids": sorted(item.sample_id for item in train),
            "dev_sample_ids": sorted(item.sample_id for item in dev),
            "test_sample_ids": sorted(item.sample_id for item in test),
        }
    )


def _current_quarantine_digest() -> str:
    """Returns the digest of the complete deterministic quarantine ledger."""
    return quarantine_ledger_digest(current_training_state().quarantine_records)


def _resolved_model_revision(*, backend_id: str, model_id: str) -> str:
    """Returns an exact backend-resolved revision or rejects unverifiable reuse."""
    if backend_id == "handcrafted" and model_id == "builtin":
        return PREPARATION_CODE_VERSION
    state = current_training_state()
    backend = state.checked_backend
    if (
        backend is None
        or state.checked_backend_id != backend_id
        or state.checked_model_id != model_id
    ):
        raise PreparedPlanError(
            "Prepared plan requires the exact readiness-checked backend instance."
        )
    revision = getattr(backend, "model_revision", None)
    if not isinstance(revision, str) or not revision.strip() or revision.strip() == model_id:
        raise PreparedPlanError(
            "Prepared plan backend revision is unpinned or could not be verified."
        )
    return revision.strip()


def _profile_smoke_runtime(
    settings: AppConfig,
) -> tuple[str, str, str, str, FeatureBackend]:
    """Resolves and builds the exact selected feature backend for smoke validation."""
    profile = resolve_profile_name(settings)
    if profile == "medium":
        backend_id = MEDIUM_BACKEND_ID
        model_id = resolve_medium_model_id(settings)
        device, dtype = resolve_runtime_selectors_for_backend_id(
            settings=settings,
            backend_id=backend_id,
            logger=logger,
        )
        return (
            backend_id,
            model_id,
            device,
            dtype,
            build_medium_backend_for_settings(model_id, device, dtype, settings),
        )
    if profile == "accurate":
        backend_id = ACCURATE_BACKEND_ID
        model_id = resolve_accurate_model_id(settings)
        device, dtype = resolve_runtime_selectors_for_backend_id(
            settings=settings,
            backend_id=backend_id,
            logger=logger,
        )
        return (
            backend_id,
            model_id,
            device,
            dtype,
            build_accurate_backend_for_settings(model_id, device, dtype, settings),
        )
    if profile == "accurate-research":
        backend_id = ACCURATE_RESEARCH_BACKEND_ID
        model_id = resolve_accurate_research_model_id(settings)
        device, dtype = resolve_runtime_selectors_for_backend_id(
            settings=settings,
            backend_id=backend_id,
            logger=logger,
        )
        return (
            backend_id,
            model_id,
            device,
            dtype,
            build_accurate_research_backend_for_settings(model_id, device, settings),
        )
    raise ValueError(f"Profile {profile!r} does not use a sequence backend.")


def _run_selected_backend_smoke_unbounded(
    *,
    settings: AppConfig,
    samples: tuple[Utterance, ...],
    probe_cache_dir: Path,
) -> SmokeResult:
    """Runs bounded real audio/backend/pooling/cache validation without production writes."""
    if not samples:
        raise RuntimeError("Backend smoke requires at least one selected sample.")
    profile = resolve_profile_name(settings)
    if profile == "fast":
        features = [
            np.asarray(
                _extract_feature_for_settings(
                    str(sample.audio_path),
                    feature_flags=settings.feature_flags,
                    audio_read_config=settings.audio_read,
                ),
                dtype=np.float64,
            )
            for sample in samples
        ]
        if any(
            vector.ndim != 1 or vector.size <= 0 or not np.all(np.isfinite(vector))
            for vector in features
        ):
            raise ValueError("Handcrafted smoke feature output is non-finite or shape-invalid.")
        probe_path = probe_cache_dir / "handcrafted-smoke.npz"
        with probe_path.open("wb") as handle:
            np.savez_compressed(handle, features=np.vstack(features))
        with np.load(probe_path, allow_pickle=False) as payload:
            round_trip = np.asarray(payload["features"], dtype=np.float64)
        expected = np.vstack(features)
        return SmokeResult(
            attempted=len(samples),
            succeeded=len(samples),
            feature_dim=int(expected.shape[1]),
            cache_round_trip=bool(np.array_equal(round_trip, expected)),
            backend_id="handcrafted",
            model_id="builtin",
            device="cpu",
            dtype="float64",
        )

    backend_id, model_id, device, dtype, backend_object = _profile_smoke_runtime(settings)
    state = current_training_state()
    if state.checked_backend is not None:
        close_checked_backend(state)
    state.checked_backend = backend_object
    state.checked_backend_id = backend_id
    state.checked_model_id = model_id
    state.checked_device = device
    state.checked_dtype = dtype
    runtime_config = (
        settings.medium_runtime
        if profile == "medium"
        else (
            settings.accurate_runtime
            if profile == "accurate"
            else settings.accurate_research_runtime
        )
    )
    cache = EmbeddingCache(probe_cache_dir / "embeddings")
    pooled_rows: list[NDArray[np.float64]] = []
    cache_round_trip = True
    for sample in samples:

        def _read_probe(
            file_path: str,
            *,
            start_seconds: float | None = None,
            duration_seconds: float | None = None,
            current_sample: Utterance = sample,
        ) -> tuple[NDArray[np.float32], int]:
            del file_path
            return read_audio_file(
                str(current_sample.audio_path),
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                audio_read_config=settings.audio_read,
            )

        encoded = encode_sequence_with_cache(
            audio_path=str(sample.audio_path),
            start_seconds=sample.start_seconds,
            duration_seconds=sample.duration_seconds,
            backend=backend_object,
            cache=cache,
            backend_id=backend_id,
            model_id=model_id,
            frame_size_seconds=runtime_config.pool_window_size_seconds,
            frame_stride_seconds=runtime_config.pool_window_stride_seconds,
            log_prefix="Readiness smoke",
            logger=logger,
            read_audio=_read_probe,
        )
        windows = temporal_pooling_windows(
            encoded,
            window_size_seconds=runtime_config.pool_window_size_seconds,
            window_stride_seconds=runtime_config.pool_window_stride_seconds,
        )
        pooled = np.asarray(mean_std_pool(encoded, windows), dtype=np.float64)
        if pooled.ndim != 2 or pooled.shape[0] <= 0 or not np.all(np.isfinite(pooled)):
            raise ValueError("Backend smoke pooled feature output is non-finite or shape-invalid.")
        pooled_rows.append(pooled)
        second = encode_sequence_with_cache(
            audio_path=str(sample.audio_path),
            start_seconds=sample.start_seconds,
            duration_seconds=sample.duration_seconds,
            backend=backend_object,
            cache=cache,
            backend_id=backend_id,
            model_id=model_id,
            frame_size_seconds=runtime_config.pool_window_size_seconds,
            frame_stride_seconds=runtime_config.pool_window_stride_seconds,
            log_prefix="Readiness smoke round-trip",
            logger=logger,
            read_audio=_read_probe,
        )
        cache_round_trip = cache_round_trip and np.array_equal(
            encoded.embeddings, second.embeddings
        )
    feature_dim = int(pooled_rows[0].shape[1])
    if any(int(block.shape[1]) != feature_dim for block in pooled_rows):
        raise ValueError("Backend smoke feature dimensionality is inconsistent.")
    expected_classifier_dim = int(backend_object.feature_dim) * 2
    if feature_dim != expected_classifier_dim:
        raise ValueError(
            "Backend smoke feature dimension violates classifier contract: "
            f"expected={expected_classifier_dim} actual={feature_dim}."
        )
    return SmokeResult(
        attempted=len(samples),
        succeeded=len(samples),
        feature_dim=feature_dim,
        cache_round_trip=cache_round_trip,
        backend_id=backend_id,
        model_id=model_id,
        device=device,
        dtype=dtype,
    )


@contextmanager
def _backend_smoke_deadline(seconds: float) -> Iterator[None]:
    """Enforces a hard wall-clock deadline on the main-thread backend smoke phase."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Backend smoke requires main-thread hard-deadline support.")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "ITIMER_REAL"):
        raise RuntimeError("Backend smoke hard-deadline support is unavailable on this platform.")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Backend smoke exceeded hard wall-clock timeout ({seconds:.3f}s).")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def run_selected_backend_smoke(
    *,
    settings: AppConfig,
    samples: tuple[Utterance, ...],
    probe_cache_dir: Path,
) -> SmokeResult:
    """Runs selected-backend smoke under a configured hard wall-clock deadline."""
    raw_timeout = os.getenv("SER_TRAINING_SMOKE_TIMEOUT_SECONDS", "120")
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise ValueError("SER_TRAINING_SMOKE_TIMEOUT_SECONDS must be numeric.") from error
    if not np.isfinite(timeout_seconds) or not 0.0 < timeout_seconds <= 600.0:
        raise ValueError("Training smoke timeout must be finite and within (0, 600] seconds.")
    with _backend_smoke_deadline(timeout_seconds):
        return _run_selected_backend_smoke_unbounded(
            settings=settings,
            samples=samples,
            probe_cache_dir=probe_cache_dir,
        )


def reuse_checked_backend(
    *,
    backend_id: str,
    model_id: str,
    device: str,
    dtype: str,
    build: Callable[[], FeatureBackend],
) -> FeatureBackend:
    """Transfers the already checked backend when selectors match, otherwise rebuilds."""
    state = current_training_state()
    if (
        state.checked_backend is not None
        and state.checked_backend_id == backend_id
        and state.checked_model_id == model_id
        and state.checked_device == device
        and state.checked_dtype == dtype
    ):
        return state.checked_backend
    if state.checked_backend is not None:
        close_checked_backend(state)
    backend = build()
    state.checked_backend = backend
    state.checked_backend_id = backend_id
    state.checked_model_id = model_id
    state.checked_device = device
    state.checked_dtype = dtype
    return backend


def ensure_entrypoint_readiness(
    *,
    settings: AppConfig,
    load_utterances: Callable[[], list[Utterance] | None],
) -> tuple[ReadinessReport, tuple[Utterance, ...]]:
    """Runs mandatory readiness once for a profile entrypoint and caches its result."""
    state = current_training_state()
    cache_state = state is not _DEFAULT_STATE
    if cache_state and state.readiness is not None:
        return state.readiness, state.utterances
    smoke_runner = run_selected_backend_smoke
    report, utterances = run_training_readiness(
        settings=settings,
        load_utterances=load_utterances,
        smoke_runner=smoke_runner,
        repair=state.operation.repair,
        persist_quarantine_ledger=state.operation.mode is not TrainingMode.DRY_RUN,
    )
    for phase in (
        "configuration",
        "dataset_registry",
        "dataset_media",
        "split_feasibility",
        "filesystem_resources",
        "backend_smoke",
    ):
        phase_findings = [finding for finding in report.findings if finding.check == phase]
        status = (
            "FAIL"
            if any(finding.blocking for finding in phase_findings)
            else "WARN" if phase_findings else "PASS"
        )
        logger.info("CHECK %-25s %s", phase.replace("_", " "), status)
    if not report.ready:
        raise TrainingReadinessError(report)
    logger.info(
        "READY profile=%s plan=%s",
        report.profile,
        report.settings_digest[:12],
    )
    if cache_state:
        state.readiness = report
    state.utterances = utterances
    state.quarantine_records = list(report.quarantine)
    state.cache_hits = 0
    state.cache_misses = 0
    state.recomputed_cache_entries = 0
    state.dropped_windows = 0
    state.medium_noise_stats_by_sample.clear()
    return report, utterances


def record_cache_activity(*, cache_hit: bool, recomputed: bool) -> None:
    """Records bounded cache activity for final artifact provenance."""
    state = current_training_state()
    if cache_hit:
        state.cache_hits += 1
    else:
        state.cache_misses += 1
    if recomputed:
        state.recomputed_cache_entries += 1
        state.containment_counts["cache:cache_corrupt:recompute"] += 1


def record_optional_artifact_failure(error: Exception) -> None:
    """Records an explicitly typed optional-artifact failure or aborts."""
    classification = classify_failure(error, scope=FailureScope.OPTIONAL_ARTIFACT)
    if classification.disposition is not FailureDisposition.CONTINUE:
        raise error
    state = current_training_state()
    state.containment_counts["optional_artifact:optional_artifact_failed:continue"] += 1
    logger.warning("Optional artifact failed: %s", classification.diagnostic)


def record_medium_noise_stats(*, sample_id: str, stats: object) -> None:
    """Records per-sample medium noise statistics for canonical partition provenance."""
    if not sample_id:
        raise ValueError("Medium noise statistics require a sample identity.")
    if training_operation_active():
        current_training_state().medium_noise_stats_by_sample[sample_id] = stats


def bounded_retry_local_io(
    operation: Callable[[], _PreparationT],
    *,
    identity: str,
    max_retries: int = 2,
    base_delay_seconds: float = 0.05,
) -> _PreparationT:
    """Retries only typed transient local I/O with bounded deterministic jitter."""
    if max_retries < 0 or base_delay_seconds < 0.0:
        raise ValueError("Retry bounds must be non-negative.")
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except OSError as error:
            classification = classify_failure(error, scope=FailureScope.SAMPLE)
            if (
                classification.disposition is not FailureDisposition.BOUNDED_RETRY
                or attempt >= max_retries
            ):
                raise
            state = current_training_state()
            state.bounded_retries += 1
            state.containment_counts["sample:media_decode_failed:bounded_retry"] += 1
            jitter = 0.75 + (int(digest_payload(identity)[:8], 16) % 501) / 1000.0
            delay = base_delay_seconds * (2**attempt) * jitter
            logger.warning(
                "Retrying transient local I/O attempt=%d/%d delay=%.3fs identity=%s",
                attempt + 1,
                max_retries,
                delay,
                identity[:80],
            )
            time.sleep(delay)
    raise AssertionError("bounded retry loop exhausted without returning or raising")


def record_preparation_progress(*, processed: int, total: int, sample_id: str) -> None:
    """Logs bounded deterministic preparation progress and a credible linear ETA."""
    if processed <= 0 or total <= 0 or processed > total:
        raise ValueError("Preparation progress must satisfy 0 < processed <= total.")
    state = current_training_state()
    now = time.monotonic()
    if state.preparation_started_at is None or processed == 1:
        state.preparation_started_at = now
    interval = max(1, total // 10)
    if processed != 1 and processed != total and processed % interval:
        return
    elapsed = max(0.0, now - state.preparation_started_at)
    eta: float | None = None
    if processed >= 2 and elapsed > 0.0:
        eta = elapsed * (total - processed) / processed
    logger.info(
        "PREPARE processed=%d total=%d cache_hits=%d cache_misses=%d recomputed=%d "
        "quarantined=%d elapsed=%.1fs eta=%s sample=%s",
        processed,
        total,
        state.cache_hits,
        state.cache_misses,
        state.recomputed_cache_entries,
        len(state.quarantine_records),
        elapsed,
        f"{eta:.1f}s" if eta is not None else "unknown",
        sample_id[:80],
    )


def record_dropped_windows(count: int) -> None:
    """Adds a non-negative dropped-window count to invocation provenance."""
    if count < 0:
        raise ValueError("Dropped-window count must be non-negative.")
    state = current_training_state()
    state.dropped_windows += count
    state.containment_counts["window:window_low_variance:continue"] += count


def build_training_robustness_provenance() -> dict[str, object]:
    """Builds the durable readiness/quarantine/cache provenance payload."""
    state = current_training_state()
    readiness = state.readiness
    return {
        "readiness": (
            {
                "schema_version": readiness.schema_version,
                "settings_digest": readiness.settings_digest,
                "registry_digest": readiness.registry_digest,
                "manifest_digest": readiness.manifest_digest,
                "media_digest": readiness.media_digest,
            }
            if readiness is not None
            else None
        ),
        "quarantine": [record.to_dict() for record in state.quarantine_records],
        "statistics": {
            "quarantined_samples": len(state.quarantine_records),
            "dropped_windows": state.dropped_windows,
            "cache_hits": state.cache_hits,
            "cache_misses": state.cache_misses,
            "recomputed_cache_entries": state.recomputed_cache_entries,
            "bounded_retries": state.bounded_retries,
            "containment": dict(sorted(state.containment_counts.items())),
        },
    }


def handle_sample_encoding_failure(
    *,
    settings: AppConfig,
    sample: Utterance,
    error: Exception,
) -> bool:
    """Quarantines one proven local encoding failure only when every budget permits."""
    classification = classify_failure(
        error,
        scope=FailureScope.SAMPLE,
        sample=sample,
        allowed_roots=_allowed_media_roots(settings),
    )
    if classification.disposition is not FailureDisposition.QUARANTINE:
        return False
    state = current_training_state()
    enforce_quarantine_budget(
        policy=QuarantinePolicy.from_settings(settings),
        all_samples=state.utterances,
        existing_records=state.quarantine_records,
        candidate=sample,
        classification=classification,
    )
    from ser._internal.models.dataset_splitting import (  # noqa: TID251
        split_utterances_three_way,
    )

    projected_samples = [
        item
        for item in state.utterances
        if item.sample_id != sample.sample_id
        and all(record.sample_id != item.sample_id for record in state.quarantine_records)
    ]
    try:
        train_samples, dev_samples, test_samples, split_metadata = split_utterances_three_way(
            samples=projected_samples,
            settings=settings,
            logger=logger,
        )
    except (RuntimeError, ValueError) as split_error:
        raise QuarantineBudgetExceeded(
            f"Projected canonical split is infeasible: {split_error}"
        ) from split_error
    required_classes = {item.require_label() for item in projected_samples}
    minimum = settings.data_loader.min_remaining_per_class_split
    for partition_name, partition in (
        ("train", train_samples),
        ("dev", dev_samples),
        ("test", test_samples),
    ):
        support = {
            label: sum(item.label == label for item in partition) for label in required_classes
        }
        if any(count < minimum for count in support.values()):
            raise QuarantineBudgetExceeded(
                f"Projected {partition_name} split violates class support: {support}."
            )
    if split_metadata.speaker_overlap_count > 0:
        raise QuarantineBudgetExceeded("Projected quarantine introduces speaker leakage.")
    timestamp = datetime.now(UTC).isoformat()
    record = build_quarantine_record(
        sample=sample,
        classification=classification,
        occurred_at=timestamp,
        retry_count=0,
    )
    state.quarantine_records.append(record)
    state.containment_counts[f"sample:{classification.reason_code.value}:quarantine"] += 1
    state.utterances = tuple(
        item for item in state.utterances if item.sample_id != sample.sample_id
    )
    state.preparation_quarantine_changed = True
    ledger_path = settings.tmp_folder / f"quarantine-{resolve_profile_name(settings)}.jsonl"
    write_quarantine_ledger(ledger_path, state.quarantine_records)
    logger.warning(
        "Quarantined sample %s (corpus=%s reason=%s).",
        sample.sample_id,
        sample.corpus,
        classification.reason_code.value,
    )
    return True


def prepare_until_quarantine_stable(
    *,
    settings: AppConfig,
    prepare: Callable[[], _PreparationT],
) -> _PreparationT:
    """Repeats split/preparation whenever quarantine changes its effective inputs."""
    state = current_training_state()
    data_loader_config = getattr(settings, "data_loader", None)
    configured_max = getattr(data_loader_config, "max_failed_files", 25)
    max_failed_files = (
        configured_max
        if isinstance(configured_max, int) and not isinstance(configured_max, bool)
        else 25
    )
    max_passes = max_failed_files + 1
    for _ in range(max_passes):
        state.preparation_quarantine_changed = False
        prepared = prepare()
        if not state.preparation_quarantine_changed:
            return prepared
    raise RuntimeError("Preparation quarantine did not stabilize within its absolute budget.")


def prepared_plan_for_operation(settings: AppConfig) -> PreparedPlan | None:
    """Loads the explicit prepared plan requested by the active real-training operation."""
    path = current_training_state().operation.prepared_plan
    return load_prepared_plan(path) if path is not None else None


def serialize_metadata(value: object) -> object:
    """Converts prepared DTO metadata into bounded canonical JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return serialize_metadata(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): serialize_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [serialize_metadata(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Unsupported prepared metadata type: {type(value).__name__}")


def write_prepared_feature_payload(
    *,
    settings: AppConfig,
    x_train: NDArray[np.float64],
    x_dev: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_train: Sequence[str],
    y_dev: Sequence[str],
    y_test: Sequence[str],
    metadata: Mapping[str, object],
    path: Path | None = None,
) -> Path:
    """Atomically persists reusable features without pickle/object arrays."""
    destination = path or default_prepared_payload_path(settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                x_train=np.asarray(x_train, dtype=np.float64),
                x_dev=np.asarray(x_dev, dtype=np.float64),
                x_test=np.asarray(x_test, dtype=np.float64),
                y_train=np.asarray(list(y_train), dtype=np.str_),
                y_dev=np.asarray(list(y_dev), dtype=np.str_),
                y_test=np.asarray(list(y_test), dtype=np.str_),
                metadata_json=np.asarray(
                    json.dumps(serialize_metadata(metadata), sort_keys=True, separators=(",", ":")),
                    dtype=np.str_,
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
    finally:
        tmp_path.unlink(missing_ok=True)
    return destination


def read_prepared_feature_payload(plan: PreparedPlan) -> PreparedFeaturePayload:
    """Reads and semantically validates one digest-protected prepared payload."""
    payload_path = Path(plan.payload_path)
    try:
        if hash_file(payload_path) != plan.payload_digest:
            raise PreparedPlanError("Prepared feature payload digest changed after plan loading.")
        with np.load(payload_path, allow_pickle=False) as payload:
            required = {
                "x_train",
                "x_dev",
                "x_test",
                "y_train",
                "y_dev",
                "y_test",
                "metadata_json",
            }
            if set(payload.files) != required:
                raise PreparedPlanError("Prepared feature payload keys are incomplete or unknown.")
            raw_x_train = np.asarray(payload["x_train"])
            raw_x_dev = np.asarray(payload["x_dev"])
            raw_x_test = np.asarray(payload["x_test"])
            raw_y_train = np.asarray(payload["y_train"])
            raw_y_dev = np.asarray(payload["y_dev"])
            raw_y_test = np.asarray(payload["y_test"])
            raw_metadata = np.asarray(payload["metadata_json"])
    except PreparedPlanError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise PreparedPlanError(f"Prepared feature payload is invalid: {error}") from error
    if any(
        str(matrix.dtype) != plan.feature_dtype for matrix in (raw_x_train, raw_x_dev, raw_x_test)
    ):
        raise PreparedPlanError("Prepared feature payload dtype differs from the plan.")
    if any(matrix.ndim != 2 for matrix in (raw_x_train, raw_x_dev, raw_x_test)):
        raise PreparedPlanError("Prepared feature matrices must all be two-dimensional.")
    combined_shape = (
        int(raw_x_train.shape[0] + raw_x_dev.shape[0] + raw_x_test.shape[0]),
        int(raw_x_train.shape[1]),
    )
    if len({raw_x_train.shape[1], raw_x_dev.shape[1], raw_x_test.shape[1]}) != 1 or (
        combined_shape != plan.feature_shape
    ):
        raise PreparedPlanError("Prepared feature matrices differ from the planned shape.")
    x_train = np.asarray(raw_x_train, dtype=np.float64)
    x_dev = np.asarray(raw_x_dev, dtype=np.float64)
    x_test = np.asarray(raw_x_test, dtype=np.float64)
    if not all(np.all(np.isfinite(matrix)) for matrix in (x_train, x_dev, x_test)):
        raise PreparedPlanError("Prepared feature matrices contain non-finite values.")
    if any(labels.ndim != 1 for labels in (raw_y_train, raw_y_dev, raw_y_test)):
        raise PreparedPlanError("Prepared labels must be one-dimensional arrays.")
    y_train = [str(item) for item in raw_y_train]
    y_dev = [str(item) for item in raw_y_dev]
    y_test = [str(item) for item in raw_y_test]
    if any(not label.strip() for label in (*y_train, *y_dev, *y_test)):
        raise PreparedPlanError("Prepared labels must be non-empty strings.")
    if (
        x_train.shape[0] != len(y_train)
        or x_dev.shape[0] != len(y_dev)
        or x_test.shape[0] != len(y_test)
    ):
        raise PreparedPlanError("Prepared feature/label row counts do not match.")
    if raw_metadata.ndim != 0:
        raise PreparedPlanError("Prepared feature metadata payload must be scalar JSON.")
    try:
        parsed = json.loads(str(raw_metadata.item()))
    except json.JSONDecodeError as error:
        raise PreparedPlanError("Prepared feature metadata is invalid JSON.") from error
    if not isinstance(parsed, dict):
        raise PreparedPlanError("Prepared feature metadata must be a JSON object.")
    for key, expected in (
        ("sample_ledger", list(plan.sample_ledger)),
        ("window_ledger", list(plan.window_ledger)),
        ("disposition_counts", dict(plan.disposition_counts)),
    ):
        if parsed.get(key) != serialize_metadata(expected):
            raise PreparedPlanError(f"Prepared feature metadata {key} differs from the plan.")
    included_ids = {
        str(row["sample_id"])
        for row in plan.sample_ledger
        if row.get("disposition") == "included" and isinstance(row.get("sample_id"), str)
    }
    current_ids = {item.sample_id for item in _effective_state_utterances()}
    if included_ids != current_ids:
        raise PreparedPlanError("Prepared sample ledger differs from current effective utterances.")
    planned_labels = [
        str(row["class"]) for row in plan.window_ledger if row.get("disposition") == "included"
    ]
    if planned_labels != [*y_train, *y_dev, *y_test]:
        raise PreparedPlanError("Prepared window ledger labels differ from payload labels.")
    return PreparedFeaturePayload(x_train, x_dev, x_test, y_train, y_dev, y_test, parsed)


def active_plan_path(settings: AppConfig) -> Path:
    """Returns the explicit plan path or the deterministic default destination."""
    return current_training_state().operation.prepared_plan or default_prepared_plan_path(settings)


def canonical_train_rows(
    *,
    settings: AppConfig,
    x_train: NDArray[np.float64],
    y_train: Sequence[str],
    train_sample_ids: Sequence[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Filters fresh two-way preparation to the canonical three-way train subset."""
    filtered_x, filtered_y, _, _ = canonical_train_partition(
        settings=settings,
        x_train=x_train,
        y_train=y_train,
        train_metadata=train_sample_ids,
        sample_id=lambda item: item,
    )
    return filtered_x, filtered_y


def canonical_train_partition(
    *,
    settings: AppConfig,
    x_train: NDArray[np.float64],
    y_train: Sequence[str],
    train_metadata: Sequence[_PreparationT],
    sample_id: Callable[[_PreparationT], str],
) -> tuple[NDArray[np.float64], list[str], list[_PreparationT], list[Utterance]]:
    """Returns canonical train rows, metadata, and utterances from fresh preparation."""
    from ser._internal.models.dataset_splitting import (  # noqa: TID251
        split_utterances_three_way,
    )

    train, _, _, _ = split_utterances_three_way(
        samples=list(_effective_state_utterances()), settings=settings, logger=logger
    )
    allowed = {item.sample_id for item in train}
    identifiers = [sample_id(item) for item in train_metadata]
    indices = [index for index, identifier in enumerate(identifiers) if identifier in allowed]
    if len(train_metadata) != len(y_train) or len(y_train) != int(x_train.shape[0]):
        raise PreparedPlanError("Fresh training rows lack exact per-window sample identities.")
    return (
        np.asarray(x_train[indices], dtype=np.float64),
        [y_train[index] for index in indices],
        [train_metadata[index] for index in indices],
        train,
    )


def training_meta_sample_ids(metadata: Sequence[object]) -> list[str]:
    """Extracts validated sample identities from typed or mapping window metadata."""
    identifiers: list[str] = []
    for item in metadata:
        raw = (
            item.get("sample_id") if isinstance(item, Mapping) else getattr(item, "sample_id", None)
        )
        if not isinstance(raw, str) or not raw:
            raise PreparedPlanError("Training window metadata lacks a valid sample_id.")
        identifiers.append(raw)
    return identifiers


def _prepared_ledgers(
    *,
    settings: AppConfig,
    y_train: Sequence[str],
    y_dev: Sequence[str],
    y_test: Sequence[str],
    metadata: Mapping[str, object],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    dict[str, int],
    dict[str, Mapping[str, int]],
]:
    """Builds final actual split/window/disposition ledgers for one prepared payload."""
    from ser._internal.models.dataset_splitting import (  # noqa: TID251
        split_utterances_three_way,
    )

    effective = list(_effective_state_utterances())
    train, dev, test, _ = split_utterances_three_way(
        samples=effective, settings=settings, logger=logger
    )
    train_ids = [item.sample_id for item in train]
    dev_ids = [item.sample_id for item in dev]
    test_ids = [item.sample_id for item in test]
    declared_train_ids = metadata.get("train_sample_ids")
    declared_test_ids = metadata.get("test_sample_ids")
    declared_dev_ids = metadata.get("dev_sample_ids")
    if (
        declared_train_ids is not None
        and list(cast(Sequence[str], declared_train_ids)) != train_ids
    ):
        raise PreparedPlanError("Prepared train sample ledger differs from the current split.")
    if declared_test_ids is not None and list(cast(Sequence[str], declared_test_ids)) != test_ids:
        raise PreparedPlanError("Prepared test sample ledger differs from the current split.")
    if declared_dev_ids is not None and list(cast(Sequence[str], declared_dev_ids)) != dev_ids:
        raise PreparedPlanError("Prepared dev sample ledger differs from the current split.")
    sample_by_id = {item.sample_id: item for item in effective}
    sample_rows: list[Mapping[str, object]] = [
        {
            "sample_id": sample_id,
            "partition": partition,
            "class": sample_by_id[sample_id].require_label(),
            "disposition": "included",
        }
        for partition, identifiers in (("train", train_ids), ("dev", dev_ids), ("test", test_ids))
        for sample_id in identifiers
    ]
    sample_rows.extend(
        {
            "sample_id": record.sample_id,
            "partition": record.split,
            "class": record.primary_class,
            "disposition": "quarantined",
            "reason_code": record.reason_code.value,
        }
        for record in current_training_state().quarantine_records
    )

    def _window_rows(
        partition: str,
        labels: Sequence[str],
        sample_ids: Sequence[str],
        raw_meta: object,
    ) -> list[Mapping[str, object]]:
        if raw_meta is None:
            if len(labels) != len(sample_ids):
                raise PreparedPlanError(
                    f"Prepared {partition} rows require an explicit per-window sample ledger."
                )
            meta_rows: list[Mapping[str, object]] = [
                {"sample_id": sample_id} for sample_id in sample_ids
            ]
        else:
            normalized = serialize_metadata(raw_meta)
            if not isinstance(normalized, list) or not all(
                isinstance(item, dict) for item in normalized
            ):
                raise PreparedPlanError(f"Prepared {partition} window metadata is invalid.")
            meta_rows = cast(list[Mapping[str, object]], normalized)
        if len(meta_rows) != len(labels):
            raise PreparedPlanError(f"Prepared {partition} window ledger count is stale.")
        rows: list[Mapping[str, object]] = []
        allowed_ids = set(sample_ids)
        for index, (label, meta) in enumerate(zip(labels, meta_rows, strict=True)):
            sample_id = meta.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in allowed_ids:
                raise PreparedPlanError(
                    f"Prepared {partition} window references an unknown sample."
                )
            rows.append(
                {
                    "window_id": f"{partition}:{index}",
                    "sample_id": sample_id,
                    "partition": partition,
                    "class": label,
                    "disposition": "included",
                }
            )
        return rows

    window_rows = _window_rows("train", y_train, train_ids, metadata.get("train_meta"))
    window_rows.extend(_window_rows("dev", y_dev, dev_ids, metadata.get("dev_meta")))
    window_rows.extend(_window_rows("test", y_test, test_ids, metadata.get("test_meta")))
    dropped = current_training_state().dropped_windows
    if dropped:
        window_rows.append({"window_id": "dropped", "disposition": "dropped", "count": dropped})
    dispositions = {
        "included_samples": len(effective),
        "quarantined_samples": len(current_training_state().quarantine_records),
        "included_windows": len(y_train) + len(y_dev) + len(y_test),
        "dropped_windows": dropped,
    }
    counts: dict[str, Mapping[str, int]] = {
        "partition_samples": {"train": len(train_ids), "test": len(test_ids), "dev": len(dev_ids)},
        "partition_windows": {"train": len(y_train), "test": len(y_test), "dev": len(y_dev)},
        "class": dict(Counter(item.require_label() for item in effective)),
        "corpus": dict(Counter(item.corpus for item in effective)),
        "language": dict(Counter(item.language or "unknown" for item in effective)),
        "native_split": dict(Counter(str(item.split or "unspecified") for item in effective)),
        "partition_support": {"train": 1, "test": 1, "dev": 1},
        "disposition": dispositions,
    }
    return tuple(sample_rows), tuple(window_rows), dispositions, counts


def publish_prepared_features(
    *,
    settings: AppConfig,
    backend_id: str,
    model_id: str,
    device: str,
    dtype: str,
    utterances: Sequence[Utterance],
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    y_train: Sequence[str],
    y_test: Sequence[str],
    metadata: Mapping[str, object],
    cache_namespace: str,
    windowing_policy: Mapping[str, object],
    noise_statistics: Mapping[str, object],
) -> PreparedPlan:
    """Atomically publishes safe reusable features followed by their ready plan."""
    started_at = time.perf_counter()
    logger.info(
        "PREPARED_PUBLISH_START backend_id=%s model_id=%s cache_namespace=%s rows=%d",
        backend_id,
        model_id,
        cache_namespace,
        int(x_train.shape[0] + x_test.shape[0]),
    )
    state = current_training_state()
    readiness = state.readiness
    if readiness is None:
        raise RuntimeError("Prepared publication requires a completed readiness pass.")
    trusted_namespace = expected_cache_namespace(settings)
    if cache_namespace != trusted_namespace:
        raise PreparedPlanError(
            f"Prepared cache namespace must be {trusted_namespace!r} for the active profile."
        )
    validated_cache_root(settings, trusted_namespace)
    from ser._internal.models.dataset_splitting import (  # noqa: TID251
        split_utterances,
        split_utterances_three_way,
    )

    effective = list(_effective_state_utterances())
    initial_train, _, _ = split_utterances(samples=effective, settings=settings, logger=logger)
    train_partition, dev_partition, test_partition, _ = split_utterances_three_way(
        samples=effective, settings=settings, logger=logger
    )
    train_ids = [item.sample_id for item in train_partition]
    dev_ids = [item.sample_id for item in dev_partition]
    test_ids = [item.sample_id for item in test_partition]
    required_classes = {item.require_label() for item in effective}
    minimum = settings.data_loader.min_remaining_per_class_split
    for partition_name, partition in (
        ("train", train_partition),
        ("dev", dev_partition),
        ("test", test_partition),
    ):
        support = {
            label: sum(item.label == label for item in partition) for label in required_classes
        }
        if any(count < minimum for count in support.values()):
            raise PreparedPlanError(
                f"Prepared {partition_name} split violates minimum class support: {support}."
            )
    raw_train_meta = metadata.get("train_meta")
    if raw_train_meta is None:
        if len(y_train) != len(initial_train):
            raise PreparedPlanError(
                "Prepared train rows require per-window metadata for dev split."
            )
        train_meta_rows: list[Mapping[str, object]] = [
            {"sample_id": item.sample_id} for item in initial_train
        ]
    else:
        normalized_train_meta = serialize_metadata(raw_train_meta)
        if not isinstance(normalized_train_meta, list) or not all(
            isinstance(item, dict) for item in normalized_train_meta
        ):
            raise PreparedPlanError("Prepared train window metadata is invalid.")
        train_meta_rows = cast(list[Mapping[str, object]], normalized_train_meta)
    if len(train_meta_rows) != len(y_train) or len(y_train) != int(x_train.shape[0]):
        raise PreparedPlanError("Prepared train feature/window metadata counts differ.")
    train_id_set = set(train_ids)
    dev_id_set = set(dev_ids)
    retained_indices = [
        index for index, row in enumerate(train_meta_rows) if row.get("sample_id") in train_id_set
    ]
    dev_indices = [
        index for index, row in enumerate(train_meta_rows) if row.get("sample_id") in dev_id_set
    ]
    if len(retained_indices) + len(dev_indices) != len(train_meta_rows):
        raise PreparedPlanError("Prepared train windows do not map to train/dev samples.")
    x_dev = np.asarray(x_train[dev_indices], dtype=np.float64)
    y_dev = [y_train[index] for index in dev_indices]
    x_train = np.asarray(x_train[retained_indices], dtype=np.float64)
    y_train = [y_train[index] for index in retained_indices]
    metadata = {
        **metadata,
        "train_sample_ids": train_ids,
        "dev_sample_ids": dev_ids,
        "test_sample_ids": test_ids,
        "train_meta": [train_meta_rows[index] for index in retained_indices],
        "dev_meta": [train_meta_rows[index] for index in dev_indices],
    }
    effective_utterances = _effective_state_utterances()
    sample_ledger, window_ledger, dispositions, final_counts = _prepared_ledgers(
        settings=settings,
        y_train=y_train,
        y_dev=y_dev,
        y_test=y_test,
        metadata=metadata,
    )
    payload_metadata = {
        **metadata,
        "quarantine": [record.to_dict() for record in state.quarantine_records],
        "effective_sample_ids": [item.sample_id for item in effective_utterances],
        "sample_ledger": list(sample_ledger),
        "window_ledger": list(window_ledger),
        "disposition_counts": dispositions,
    }
    payload_path = write_prepared_feature_payload(
        settings=settings,
        x_train=x_train,
        x_dev=x_dev,
        x_test=x_test,
        y_train=y_train,
        y_dev=y_dev,
        y_test=y_test,
        metadata=payload_metadata,
    )
    cache_keys = _cache_keys(settings, cache_namespace)
    recipe_digest = _recipe_digest(settings)
    split_digest = _current_split_digest(settings)
    quarantine_digest = _current_quarantine_digest()
    model_revision = _resolved_model_revision(backend_id=backend_id, model_id=model_id)
    plan = build_prepared_plan(
        settings=settings,
        readiness=readiness,
        backend_id=backend_id,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
        dtype=dtype,
        recipe_digest=recipe_digest,
        split_ledger_digest=split_digest,
        quarantine_ledger_digest=quarantine_digest,
        cache_namespace=trusted_namespace,
        cache_version=PREPARED_CACHE_VERSION,
        cache_keys=cache_keys,
        effective_counts=final_counts,
        sample_ledger=sample_ledger,
        window_ledger=window_ledger,
        disposition_counts=dispositions,
        feature_shape=(
            int(x_train.shape[0] + x_dev.shape[0] + x_test.shape[0]),
            int(x_train.shape[1]),
        ),
        feature_dtype=str(np.asarray(x_train).dtype),
        windowing_policy=windowing_policy,
        noise_statistics=noise_statistics,
        payload_path=payload_path,
        package_version=_package_version(),
    )
    write_prepared_plan(default_prepared_plan_path(settings), plan)
    logger.info(
        "PREPARED_PUBLISH_DONE payload_path=%s plan_path=%s feature_shape=%s elapsed=%.1fs",
        payload_path,
        default_prepared_plan_path(settings),
        plan.feature_shape,
        time.perf_counter() - started_at,
    )
    return plan


def validate_operation_plan(
    *,
    settings: AppConfig,
    backend_id: str,
    model_id: str,
    device: str,
    dtype: str,
) -> PreparedPlan | None:
    """Loads and validates an explicit plan against the current readiness snapshot."""
    plan = prepared_plan_for_operation(settings)
    if plan is None:
        return None
    readiness = current_training_state().readiness
    if readiness is None:
        raise RuntimeError("Prepared plan validation requires a completed readiness pass.")
    trusted_namespace = expected_cache_namespace(settings)
    validated_cache_root(settings, plan.cache_namespace)
    model_revision = _resolved_model_revision(backend_id=backend_id, model_id=model_id)
    validate_prepared_plan(
        plan,
        settings=settings,
        readiness=readiness,
        backend_id=backend_id,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
        dtype=dtype,
        recipe_digest=_recipe_digest(settings),
        split_ledger_digest=_current_split_digest(settings),
        quarantine_ledger_digest=_current_quarantine_digest(),
        cache_namespace=trusted_namespace,
        cache_version=PREPARED_CACHE_VERSION,
        cache_keys=_cache_keys(settings, trusted_namespace),
        package_version=_package_version(),
    )
    return plan


__all__ = [
    "PreparedFeaturePayload",
    "TrainingRunState",
    "active_plan_path",
    "build_training_robustness_provenance",
    "close_checked_backend",
    "current_training_state",
    "ensure_entrypoint_readiness",
    "handle_sample_encoding_failure",
    "prepared_plan_for_operation",
    "prepare_until_quarantine_stable",
    "publish_prepared_features",
    "read_prepared_feature_payload",
    "record_cache_activity",
    "record_dropped_windows",
    "reuse_checked_backend",
    "run_selected_backend_smoke",
    "serialize_metadata",
    "training_operation_scope",
    "training_operation_active",
    "validate_operation_plan",
    "write_prepared_feature_payload",
]
