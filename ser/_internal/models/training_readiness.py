"""Typed readiness, preparation-plan, and fault-containment contracts for training."""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import json
import logging
import math
import os
import resource
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import soundfile as sf

from ser.config import AppConfig
from ser.profiles import ProfileName, resolve_profile_name

from ..data.application.registry_snapshot import (  # noqa: TID251
    collect_dataset_registry_snapshot,
)
from ..data.manifest import Utterance  # noqa: TID251
from ..utils.audio_utils import AudioDecodeError, AudioIntegrityError  # noqa: TID251

logger: logging.Logger = logging.getLogger(__name__)
READINESS_SCHEMA_VERSION = 1
PREPARED_PLAN_SCHEMA_VERSION = 2
PREPARATION_CODE_VERSION = "ser-training-preparation-v2"
PREPARED_CACHE_VERSION = "embedding-cache-v1"
PROFILE_CACHE_NAMESPACES: Mapping[ProfileName, str] = {
    "fast": "fast_features",
    "medium": "medium_embeddings",
    "accurate": "accurate_embeddings",
    "accurate-research": "accurate_research_embeddings",
}
DEFAULT_SMOKE_SAMPLE_CAP = 16
_GIT_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_TRANSIENT_LOCAL_IO_ERRNOS = frozenset({errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.ETIMEDOUT})


def _readiness_status(findings: Sequence[ReadinessFinding]) -> str:
    """Returns a concise status label for one readiness phase."""
    if any(finding.blocking for finding in findings):
        return "FAIL"
    return "WARN" if findings else "PASS"


def _readiness_log_level(status: str) -> int:
    """Returns a standard log level for one readiness status label."""
    if status == "FAIL":
        return logging.ERROR
    if status == "WARN":
        return logging.WARNING
    return logging.INFO


def _log_repair_record(record: RepairRecord) -> None:
    """Emits one bounded repair action result."""
    level = logging.INFO if record.succeeded else logging.WARNING
    logger.log(
        level,
        "REPAIR_ACTION action=%s target=%s succeeded=%s detail=%s",
        record.action,
        record.target[:200],
        record.succeeded,
        record.message[:500],
    )


def _progress_interval(total: int) -> int:
    """Returns a low-volume progress interval for large readiness inventories."""
    return max(1, total // 10)


def _should_log_progress(
    *,
    processed: int,
    total: int,
    last_logged_at: float,
    now: float,
) -> bool:
    """Returns whether one bounded progress event should be emitted."""
    return (
        processed == 1
        or processed == total
        or processed % _progress_interval(total) == 0
        or (now - last_logged_at) >= 30.0
    )


class FailureScope(StrEnum):
    """Scope at which a training failure is known to apply."""

    RUN = "run"
    CORPUS = "corpus"
    SAMPLE = "sample"
    WINDOW = "window"
    CACHE = "cache"
    OPTIONAL_ARTIFACT = "optional_artifact"


class FailureDisposition(StrEnum):
    """Permitted action after one classified failure."""

    ABORT = "abort"
    REPAIR_THEN_RETRY = "repair_then_retry"
    BOUNDED_RETRY = "bounded_retry"
    RECOMPUTE = "recompute"
    QUARANTINE = "quarantine"
    CONTINUE = "continue"


class FailureSeverity(StrEnum):
    """Structured severity for training findings."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FailureReasonCode(StrEnum):
    """Stable reason codes emitted by readiness and containment services."""

    INVALID_CONFIGURATION = "invalid_configuration"
    DATASET_NOT_FOUND = "dataset_not_found"
    REGISTRY_UNHEALTHY = "registry_unhealthy"
    MANIFEST_INVALID = "manifest_invalid"
    MEDIA_MISSING = "media_missing"
    MEDIA_NOT_REGULAR = "media_not_regular"
    MEDIA_EMPTY = "media_empty"
    MEDIA_DECODE_FAILED = "media_decode_failed"
    GIT_LFS_POINTER = "git_lfs_pointer"
    DUPLICATE_SAMPLE_ID = "duplicate_sample_id"
    DUPLICATE_CONTENT = "duplicate_content"
    PATH_ALIAS = "path_alias"
    INSUFFICIENT_CLASS_SUPPORT = "insufficient_class_support"
    SPLIT_LEAKAGE = "split_leakage"
    OUTPUT_UNWRITABLE = "output_unwritable"
    DISK_SPACE_LOW = "disk_space_low"
    RESOURCE_LIMIT = "resource_limit"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_SMOKE_TIMEOUT = "backend_smoke_timeout"
    BACKEND_OUTPUT_INVALID = "backend_output_invalid"
    SAMPLE_AUDIO_CORRUPT = "sample_audio_corrupt"
    SAMPLE_AUDIO_MISSING = "sample_audio_missing"
    WINDOW_LOW_VARIANCE = "window_low_variance"
    CACHE_CORRUPT = "cache_corrupt"
    OPTIONAL_ARTIFACT_FAILED = "optional_artifact_failed"
    QUARANTINE_BUDGET_EXCEEDED = "quarantine_budget_exceeded"
    PREPARED_PLAN_INVALID = "prepared_plan_invalid"
    REPAIR_FAILED = "repair_failed"


class TrainingMode(StrEnum):
    """Training orchestration mode selected by the CLI or library boundary."""

    TRAIN = "train"
    DRY_RUN = "dry_run"
    PREPARE_ONLY = "prepare_only"


@dataclass(frozen=True, slots=True)
class TrainingOperation:
    """Mode and checkpoint inputs for one training invocation."""

    mode: TrainingMode = TrainingMode.TRAIN
    repair: bool = False
    prepared_plan: Path | None = None

    def validate(self) -> None:
        """Validates mutually compatible operation flags."""
        if self.repair and self.mode not in {TrainingMode.DRY_RUN, TrainingMode.PREPARE_ONLY}:
            raise ValueError("--repair is valid only with --dry-run or --prepare-only.")
        if self.prepared_plan is not None and self.mode is not TrainingMode.TRAIN:
            raise ValueError("--prepared-plan is valid only for real training.")


@dataclass(frozen=True, slots=True)
class FailureClassification:
    """Typed failure classification used before any continuation decision."""

    scope: FailureScope
    reason_code: FailureReasonCode
    disposition: FailureDisposition
    severity: FailureSeverity
    diagnostic: str


@dataclass(frozen=True, slots=True)
class QuarantinePolicy:
    """Bias-aware limits for excluding proven sample-local failures."""

    max_absolute: int
    max_global_ratio: float
    max_corpus_ratio: float
    max_class_ratio: float
    max_per_reason: int
    min_remaining_per_class_split: int
    strict: bool = False

    def __post_init__(self) -> None:
        """Validates all budget boundaries eagerly."""
        for name, value in (
            ("max_absolute", self.max_absolute),
            ("max_per_reason", self.max_per_reason),
            ("min_remaining_per_class_split", self.min_remaining_per_class_split),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name, ratio in (
            ("max_global_ratio", self.max_global_ratio),
            ("max_corpus_ratio", self.max_corpus_ratio),
            ("max_class_ratio", self.max_class_ratio),
        ):
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1].")

    @classmethod
    def from_settings(cls, settings: AppConfig) -> QuarantinePolicy:
        """Builds the typed policy while preserving the legacy ratio surface."""
        config = settings.data_loader
        return cls(
            max_absolute=config.max_failed_files,
            max_global_ratio=config.max_failed_file_ratio,
            max_corpus_ratio=config.max_failed_file_ratio_per_corpus,
            max_class_ratio=config.max_failed_file_ratio_per_class,
            max_per_reason=config.max_failures_per_reason,
            min_remaining_per_class_split=config.min_remaining_per_class_split,
            strict=config.strict_quarantine,
        )


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    """One deterministic, bounded quarantine-ledger record."""

    sample_id: str
    corpus: str
    path_digest: str
    primary_class: str
    split: str
    scope: FailureScope
    reason_code: FailureReasonCode
    diagnostic: str
    first_occurrence: str
    last_occurrence: str
    retry_count: int
    disposition: FailureDisposition = FailureDisposition.QUARANTINE

    def to_dict(self) -> dict[str, object]:
        """Returns one canonical JSON-compatible ledger row."""
        return {
            "sample_id": self.sample_id,
            "corpus": self.corpus,
            "path_digest": self.path_digest,
            "primary_class": self.primary_class,
            "split": self.split,
            "scope": self.scope.value,
            "reason_code": self.reason_code.value,
            "diagnostic": self.diagnostic[:500],
            "first_occurrence": self.first_occurrence,
            "last_occurrence": self.last_occurrence,
            "retry_count": self.retry_count,
            "disposition": self.disposition.value,
        }

    def identity_dict(self) -> dict[str, object]:
        """Returns stable ledger identity without audit-only wall-clock fields."""
        payload = self.to_dict()
        payload.pop("first_occurrence")
        payload.pop("last_occurrence")
        return payload


class QuarantineBudgetExceeded(RuntimeError):
    """Raised when one projected exclusion violates any quarantine invariant."""


class TrainingReadinessError(RuntimeError):
    """Raised when mandatory readiness validation contains blocking findings."""

    def __init__(self, report: ReadinessReport) -> None:
        super().__init__("Training readiness validation failed; inspect the readiness report.")
        self.report = report


class WindowContainmentError(ValueError):
    """Known isolated window rejection that may be dropped with accounting."""


class CacheEntryCorruptError(ValueError):
    """Known derived-cache corruption that may be recomputed."""


class OptionalArtifactError(OSError):
    """Known failure of a declared optional artifact."""


class PreparedPlanError(RuntimeError):
    """Raised when a prepared plan is malformed, stale, or digest-invalid."""


@dataclass(frozen=True, slots=True)
class ReadinessFinding:
    """One stable readiness validation result."""

    check: str
    reason_code: FailureReasonCode
    severity: FailureSeverity
    message: str
    blocking: bool
    scope: FailureScope = FailureScope.RUN
    sample_id: str | None = None
    corpus: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Returns a JSON-compatible finding."""
        return {
            "check": self.check,
            "reason_code": self.reason_code.value,
            "severity": self.severity.value,
            "message": self.message,
            "blocking": self.blocking,
            "scope": self.scope.value,
            "sample_id": self.sample_id,
            "corpus": self.corpus,
        }


@dataclass(frozen=True, slots=True)
class RepairRecord:
    """Audit record for one explicitly requested allowlisted repair."""

    action: str
    target: str
    succeeded: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        """Returns a JSON-compatible repair record."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SmokeResult:
    """Bounded backend-smoke output verified by the readiness service."""

    attempted: int
    succeeded: int
    feature_dim: int
    cache_round_trip: bool
    backend_id: str
    model_id: str
    device: str
    dtype: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Durable result of one complete readiness pass."""

    schema_version: int
    created_at: str
    profile: ProfileName
    settings_digest: str
    registry_digest: str
    manifest_digest: str
    media_digest: str
    selected_sample_ids: tuple[str, ...]
    findings: tuple[ReadinessFinding, ...]
    repairs: tuple[RepairRecord, ...] = ()
    smoke: SmokeResult | None = None
    retries: int = 0
    recomputed_cache_entries: int = 0
    quarantined_samples: int = 0
    dropped_windows: int = 0
    quarantine: tuple[QuarantineRecord, ...] = ()
    effective_sample_ids: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        """Returns whether no blocking validation result remains."""
        return not any(finding.blocking for finding in self.findings)

    def to_dict(self) -> dict[str, object]:
        """Returns the stable readiness-report payload."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "profile": self.profile,
            "ready": self.ready,
            "digests": {
                "settings": self.settings_digest,
                "registry": self.registry_digest,
                "manifest": self.manifest_digest,
                "media": self.media_digest,
            },
            "selected_sample_ids": list(self.selected_sample_ids),
            "findings": [finding.to_dict() for finding in self.findings],
            "repairs": [repair.to_dict() for repair in self.repairs],
            "smoke": asdict(self.smoke) if self.smoke is not None else None,
            "statistics": {
                "retries": self.retries,
                "recomputed_cache_entries": self.recomputed_cache_entries,
                "quarantined_samples": self.quarantined_samples,
                "dropped_windows": self.dropped_windows,
            },
            "quarantine": [record.to_dict() for record in self.quarantine],
            "effective_sample_ids": list(self.effective_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class PreparedPlan:
    """Digest-protected declaration of reusable training preparation state."""

    schema_version: int
    created_at: str
    code_version: str
    package_version: str
    profile: ProfileName
    backend_id: str
    model_id: str
    model_revision: str
    device: str
    dtype: str
    settings_digest: str
    registry_digest: str
    manifest_digest: str
    media_digest: str
    recipe_digest: str
    split_ledger_digest: str
    quarantine_ledger_digest: str
    cache_namespace: str
    cache_version: str
    cache_keys: tuple[str, ...]
    effective_counts: Mapping[str, Mapping[str, int]]
    sample_ledger: tuple[Mapping[str, object], ...]
    window_ledger: tuple[Mapping[str, object], ...]
    disposition_counts: Mapping[str, int]
    feature_shape: tuple[int, int]
    feature_dtype: str
    windowing_policy: Mapping[str, object]
    noise_statistics: Mapping[str, object]
    validation_findings: tuple[Mapping[str, object], ...]
    repairs: tuple[Mapping[str, object], ...]
    payload_path: str
    payload_digest: str
    overall_digest: str = field(default="")

    def unsigned_dict(self) -> dict[str, object]:
        """Returns canonical plan content excluding its self-authenticating digest."""
        payload = asdict(self)
        payload.pop("overall_digest", None)
        return cast(dict[str, object], _json_value(payload))

    def to_dict(self) -> dict[str, object]:
        """Returns canonical plan content including the overall digest."""
        payload = self.unsigned_dict()
        payload["overall_digest"] = self.overall_digest
        return payload


class SmokeRunner(Protocol):
    """Runs bounded real-backend validation for selected samples."""

    def __call__(
        self,
        *,
        settings: AppConfig,
        samples: tuple[Utterance, ...],
        probe_cache_dir: Path,
    ) -> SmokeResult: ...


def _json_value(value: object) -> object:
    """Normalizes dataclass/config values into deterministic JSON-compatible forms."""
    if is_dataclass(value) and not isinstance(value, type):
        # Walk fields directly instead of asdict(): asdict deep-copies leaf values,
        # which raises TypeError on read-only fields (e.g. a MappingProxyType). The
        # per-field recursion is digest-equivalent because canonical_json_bytes sorts.
        return {member.name: _json_value(getattr(value, member.name)) for member in fields(value)}
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_json_value(item) for item in value), key=repr)
    if isinstance(value, float):
        return value if math.isfinite(value) else f"nonfinite:{value!r}"
    if value is None or isinstance(value, str | int | bool):
        return value
    return repr(value)


def canonical_json_bytes(payload: object) -> bytes:
    """Serializes one normalized payload with stable ordering and separators."""
    return json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_payload(payload: object) -> str:
    """Returns the SHA-256 digest for one canonical payload."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def hash_file(path: Path) -> str:
    """Returns a streaming SHA-256 file digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pcm_digest(path: Path, *, sample_id: str | None = None) -> str:
    """Streams a container-independent mono float32 PCM identity."""
    started_at = time.perf_counter()
    digest = hashlib.sha256()
    with sf.SoundFile(path) as handle:
        digest.update(str(handle.samplerate).encode("ascii"))
        frames_read = 0
        last_progress_at = started_at
        for block in handle.blocks(blocksize=65_536, dtype="float32", always_2d=True):
            mono = np.asarray(block, dtype="<f4").mean(axis=1, dtype=np.float32)
            digest.update(np.asarray(mono, dtype="<f4").tobytes(order="C"))
            frames_read += int(block.shape[0])
            progress_now = time.perf_counter()
            if sample_id is not None and (progress_now - last_progress_at) >= 30.0:
                logger.info(
                    "DATASET_MEDIA_HASH_PROGRESS sample=%s frames=%d total_frames=%d elapsed=%.1fs",
                    sample_id[:80],
                    frames_read,
                    int(handle.frames),
                    progress_now - started_at,
                )
                last_progress_at = progress_now
    return digest.hexdigest()


def expected_cache_namespace(settings: AppConfig) -> str:
    """Returns the code-owned cache namespace for the selected profile."""
    return PROFILE_CACHE_NAMESPACES[resolve_profile_name(settings)]


def validated_cache_root(settings: AppConfig, namespace: str) -> Path:
    """Resolves one relative app-owned cache namespace without path escape."""
    candidate = Path(namespace)
    if candidate.is_absolute() or not namespace.strip() or ".." in candidate.parts:
        raise PreparedPlanError("Prepared cache namespace must be a safe relative path.")
    root = (settings.tmp_folder / candidate).resolve(strict=False)
    tmp_root = settings.tmp_folder.resolve(strict=False)
    if not root.is_relative_to(tmp_root):
        raise PreparedPlanError("Prepared cache namespace escapes the application temp root.")
    return root


def quarantine_ledger_digest(records: Iterable[QuarantineRecord]) -> str:
    """Returns stable quarantine identity while retaining timestamps in audit rows."""
    rows = [
        record.identity_dict()
        for record in sorted(records, key=lambda row: (row.sample_id, row.reason_code.value))
    ]
    return digest_payload(rows)


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically publishes one JSON document and removes staging files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_value(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, raw_tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_quarantine_ledger(path: Path, records: Iterable[QuarantineRecord]) -> str:
    """Atomically writes audit JSONL and returns its stable identity digest."""
    sorted_records = sorted(records, key=lambda row: (row.sample_id, row.reason_code.value))
    rows = [record.to_dict() for record in sorted_records]
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(raw_tmp_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return quarantine_ledger_digest(sorted_records)


def select_smoke_samples(
    utterances: Sequence[Utterance],
    *,
    cap: int = DEFAULT_SMOKE_SAMPLE_CAP,
) -> tuple[Utterance, ...]:
    """Selects a deterministic bounded matrix covering corpus/format/read/language strata."""
    if cap <= 0:
        raise ValueError("Smoke sample cap must be positive.")

    strata: dict[tuple[str, str, str, str], Utterance] = {}
    coverage_keys: list[tuple[str, str, str, str, str, str, str]] = []
    coverage_candidates: dict[tuple[str, str, str, str, str, str, str], Utterance] = {}
    for utterance in utterances:
        suffix = utterance.audio_path.suffix.lower() or "<none>"
        read_kind = "segment" if utterance.start_seconds is not None else "full"
        stratum = (utterance.corpus, suffix, read_kind, utterance.language or "<none>")
        representative = strata.get(stratum)
        if representative is None:
            if len(strata) == cap:
                raise ValueError(
                    "Smoke sample cap cannot cover every required "
                    "corpus/format/read/language stratum: "
                    f"required>{cap} cap={cap}."
                )
            strata[stratum] = utterance
        elif (utterance.sample_id, str(utterance.audio_path)) < (
            representative.sample_id,
            str(representative.audio_path),
        ):
            strata[stratum] = utterance

        coverage_key = (
            utterance.label or "",
            utterance.language or "",
            utterance.corpus,
            utterance.sample_id,
            str(utterance.audio_path),
            str(utterance.start_seconds),
            str(utterance.duration_seconds),
        )
        insertion_index = bisect_left(coverage_keys, coverage_key)
        if insertion_index == len(coverage_keys) or coverage_keys[insertion_index] != coverage_key:
            coverage_keys.insert(insertion_index, coverage_key)
            coverage_candidates[coverage_key] = utterance
            if len(coverage_keys) > cap:
                removed_key = coverage_keys.pop()
                coverage_candidates.pop(removed_key)

    selected: list[Utterance] = []
    seen_ids: set[str] = set()
    for key in sorted(strata):
        candidate = strata[key]
        selected.append(candidate)
        seen_ids.add(candidate.sample_id)
        if len(selected) == cap:
            return tuple(selected)
    for coverage_key in coverage_keys:
        candidate = coverage_candidates[coverage_key]
        if candidate.sample_id in seen_ids:
            continue
        selected.append(candidate)
        seen_ids.add(candidate.sample_id)
        if len(selected) == cap:
            break
    return tuple(selected)


def classify_failure(
    error: Exception,
    *,
    scope: FailureScope,
    sample: Utterance | None = None,
    allowed_roots: Sequence[Path] = (),
) -> FailureClassification:
    """Classifies only known exception types; unknown failures remain aborting."""
    diagnostic = (str(error).strip() or type(error).__name__)[:500]
    if isinstance(error, AudioIntegrityError) and "Git LFS" in diagnostic:
        return FailureClassification(
            scope=FailureScope.CORPUS,
            reason_code=FailureReasonCode.GIT_LFS_POINTER,
            disposition=FailureDisposition.ABORT,
            severity=FailureSeverity.ERROR,
            diagnostic=diagnostic,
        )
    if scope is FailureScope.WINDOW and isinstance(error, WindowContainmentError):
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.WINDOW_LOW_VARIANCE,
            disposition=FailureDisposition.CONTINUE,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    if scope is FailureScope.CACHE and isinstance(error, CacheEntryCorruptError):
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.CACHE_CORRUPT,
            disposition=FailureDisposition.RECOMPUTE,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    if scope is FailureScope.OPTIONAL_ARTIFACT and isinstance(error, OptionalArtifactError):
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.OPTIONAL_ARTIFACT_FAILED,
            disposition=FailureDisposition.CONTINUE,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    if scope is FailureScope.SAMPLE and (
        isinstance(error, TimeoutError | InterruptedError)
        or (isinstance(error, OSError) and error.errno in _TRANSIENT_LOCAL_IO_ERRNOS)
    ):
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.MEDIA_DECODE_FAILED,
            disposition=FailureDisposition.BOUNDED_RETRY,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    missing_sample_is_proven = False
    if scope is FailureScope.SAMPLE and isinstance(error, FileNotFoundError) and sample is not None:
        filename = error.filename
        if isinstance(filename, str):
            failed_path = Path(filename).expanduser().resolve(strict=False)
            sample_path = sample.audio_path.expanduser().resolve(strict=False)
            missing_sample_is_proven = failed_path == sample_path and any(
                sample_path.is_relative_to(root.expanduser().resolve(strict=False))
                for root in allowed_roots
            )
    if missing_sample_is_proven:
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.SAMPLE_AUDIO_MISSING,
            disposition=FailureDisposition.QUARANTINE,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    if scope is FailureScope.SAMPLE and isinstance(error, AudioDecodeError):
        return FailureClassification(
            scope=scope,
            reason_code=FailureReasonCode.SAMPLE_AUDIO_CORRUPT,
            disposition=FailureDisposition.QUARANTINE,
            severity=FailureSeverity.WARNING,
            diagnostic=diagnostic,
        )
    return FailureClassification(
        scope=scope,
        reason_code=FailureReasonCode.BACKEND_OUTPUT_INVALID,
        disposition=FailureDisposition.ABORT,
        severity=FailureSeverity.ERROR,
        diagnostic=diagnostic,
    )


def enforce_quarantine_budget(
    *,
    policy: QuarantinePolicy,
    all_samples: Sequence[Utterance],
    existing_records: Sequence[QuarantineRecord],
    candidate: Utterance,
    classification: FailureClassification,
) -> None:
    """Rejects one projected quarantine when any global or bias-aware budget fails."""
    if classification.scope is not FailureScope.SAMPLE:
        raise QuarantineBudgetExceeded("Only sample-local failures may be quarantined.")
    if classification.disposition is not FailureDisposition.QUARANTINE:
        raise QuarantineBudgetExceeded("Failure classification does not permit quarantine.")
    if policy.strict:
        raise QuarantineBudgetExceeded("Strict quarantine policy disables sample exclusion.")
    if not all_samples:
        raise QuarantineBudgetExceeded("Cannot quarantine from an empty sample inventory.")
    projected = len(existing_records) + 1
    if projected > policy.max_absolute:
        raise QuarantineBudgetExceeded("Absolute quarantine budget exceeded.")
    if projected / len(all_samples) > policy.max_global_ratio:
        raise QuarantineBudgetExceeded("Global quarantine ratio exceeded.")

    corpus_total = sum(item.corpus == candidate.corpus for item in all_samples)
    corpus_failed = sum(row.corpus == candidate.corpus for row in existing_records) + 1
    if corpus_total <= 0 or corpus_failed / corpus_total > policy.max_corpus_ratio:
        raise QuarantineBudgetExceeded("Per-corpus quarantine ratio exceeded.")

    label = candidate.require_label()
    class_total = sum(item.label == label for item in all_samples)
    class_failed = sum(row.primary_class == label for row in existing_records) + 1
    if class_total <= 0 or class_failed / class_total > policy.max_class_ratio:
        raise QuarantineBudgetExceeded("Per-class quarantine ratio exceeded.")

    same_reason = sum(row.reason_code is classification.reason_code for row in existing_records) + 1
    if same_reason > policy.max_per_reason:
        raise QuarantineBudgetExceeded("Per-reason systematic-failure threshold exceeded.")

    split = str(candidate.split or "unspecified")
    remaining = sum(
        item.label == label
        and str(item.split or "unspecified") == split
        and item.sample_id != candidate.sample_id
        and all(record.sample_id != item.sample_id for record in existing_records)
        for item in all_samples
    )
    if remaining < policy.min_remaining_per_class_split:
        raise QuarantineBudgetExceeded("Minimum remaining class/split support would be violated.")


def build_quarantine_record(
    *,
    sample: Utterance,
    classification: FailureClassification,
    occurred_at: str,
    retry_count: int,
) -> QuarantineRecord:
    """Builds a bounded ledger row without exposing a raw source path."""
    return QuarantineRecord(
        sample_id=sample.sample_id,
        corpus=sample.corpus,
        path_digest=hashlib.sha256(str(sample.audio_path).encode("utf-8")).hexdigest(),
        primary_class=sample.require_label(),
        split=str(sample.split or "unspecified"),
        scope=classification.scope,
        reason_code=classification.reason_code,
        diagnostic=classification.diagnostic,
        first_occurrence=occurred_at,
        last_occurrence=occurred_at,
        retry_count=retry_count,
        disposition=classification.disposition,
    )


def settings_digest(settings: AppConfig) -> str:
    """Returns the complete effective-settings digest relevant to training."""
    return digest_payload(settings)


def default_readiness_report_path(settings: AppConfig) -> Path:
    """Returns the atomic readiness-report location for the active profile."""
    return settings.tmp_folder / f"training-readiness-{resolve_profile_name(settings)}.json"


def default_prepared_plan_path(settings: AppConfig) -> Path:
    """Returns the prepared-plan location for the active profile."""
    return settings.tmp_folder / f"prepared-training-{resolve_profile_name(settings)}.json"


def default_prepared_payload_path(settings: AppConfig) -> Path:
    """Returns the reusable feature-payload location for the active profile."""
    return settings.tmp_folder / f"prepared-training-{resolve_profile_name(settings)}.npz"


def _inventory_digests(
    settings: AppConfig,
    utterances: Sequence[Utterance],
    *,
    media_utterances: Sequence[Utterance] | None = None,
) -> tuple[str, str, str]:
    started_at = time.perf_counter()
    logger.info(
        "INVENTORY_DIGEST_START utterances=%d media_utterances=%d",
        len(utterances),
        len(media_utterances if media_utterances is not None else utterances),
    )
    registry = collect_dataset_registry_snapshot(settings=settings)
    registry_payload = {
        "entries": [asdict(entry) for entry in registry.entries],
        "issues": [asdict(issue) for issue in registry.issues],
    }
    manifest_rows = [
        {
            "sample_id": item.sample_id,
            "corpus": item.corpus,
            "label": item.label,
            "speaker_id": item.speaker_id,
            "session_id": item.session_id,
            "language": item.language,
            "split": item.split,
            "start_seconds": item.start_seconds,
            "duration_seconds": item.duration_seconds,
            "normalized_audio_sha256": item.normalized_audio_sha256,
            "dataset_revision": item.dataset_revision,
        }
        for item in sorted(utterances, key=lambda row: row.sample_id)
    ]
    media_rows: list[dict[str, object]] = []
    media_inventory = utterances if media_utterances is None else media_utterances
    sorted_media_inventory = sorted(media_inventory, key=lambda row: row.sample_id)
    total = len(sorted_media_inventory)
    last_progress_at = time.perf_counter()
    hashed_count = 0
    for processed, item in enumerate(sorted_media_inventory, start=1):
        path = item.audio_path.expanduser()
        if not path.is_file():
            media_rows.append(
                {
                    "sample_id": item.sample_id,
                    "status": "missing_or_non_regular",
                }
            )
            continue
        stat = path.stat()
        file_hash = hash_file(path)
        hashed_count += 1
        media_rows.append(
            {
                "sample_id": item.sample_id,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": file_hash,
            }
        )
        progress_now = time.perf_counter()
        if _should_log_progress(
            processed=processed,
            total=total,
            last_logged_at=last_progress_at,
            now=progress_now,
        ):
            logger.info(
                "INVENTORY_DIGEST_PROGRESS checked=%d total=%d hashed=%d elapsed=%.1fs sample=%s",
                processed,
                total,
                hashed_count,
                progress_now - started_at,
                item.sample_id[:80],
            )
            last_progress_at = progress_now
    result = (
        digest_payload(registry_payload),
        digest_payload(manifest_rows),
        digest_payload(media_rows),
    )
    logger.info(
        "INVENTORY_DIGEST_DONE media_rows=%d hashed=%d elapsed=%.1fs",
        len(media_rows),
        hashed_count,
        time.perf_counter() - started_at,
    )
    return result


def _configuration_findings(settings: AppConfig) -> list[ReadinessFinding]:
    findings: list[ReadinessFinding] = []
    numeric_checks: tuple[tuple[str, float, Callable[[float], bool]], ...] = (
        ("training.test_size", settings.training.test_size, lambda value: 0.0 < value < 1.0),
        ("training.dev_size", settings.training.dev_size, lambda value: 0.0 < value < 1.0),
        (
            "data_loader.max_failed_file_ratio",
            settings.data_loader.max_failed_file_ratio,
            lambda value: 0.0 <= value <= 1.0,
        ),
        (
            "audio_read.retry_delay_seconds",
            settings.audio_read.retry_delay_seconds,
            lambda value: value >= 0.0,
        ),
        (
            "data_loader.max_failed_file_ratio_per_corpus",
            settings.data_loader.max_failed_file_ratio_per_corpus,
            lambda value: 0.0 <= value <= 1.0,
        ),
        (
            "data_loader.max_failed_file_ratio_per_class",
            settings.data_loader.max_failed_file_ratio_per_class,
            lambda value: 0.0 <= value <= 1.0,
        ),
        ("nn.alpha", settings.nn.alpha, lambda value: value >= 0.0),
        ("nn.epsilon", settings.nn.epsilon, lambda value: value > 0.0),
        (
            "medium_training.min_window_std",
            settings.medium_training.min_window_std,
            lambda value: value >= 0.0,
        ),
    )
    for name, value, predicate in numeric_checks:
        if not math.isfinite(value) or not predicate(value):
            findings.append(
                ReadinessFinding(
                    check="configuration",
                    reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                    severity=FailureSeverity.ERROR,
                    message=f"Invalid training setting {name}={value!r}.",
                    blocking=True,
                )
            )
    if settings.training.test_size + settings.training.dev_size >= 1.0:
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="training.test_size + training.dev_size must be below 1.0.",
                blocking=True,
            )
        )
    for profile_name, runtime in (
        ("fast", settings.fast_runtime),
        ("medium", settings.medium_runtime),
        ("accurate", settings.accurate_runtime),
        ("accurate-research", settings.accurate_research_runtime),
    ):
        runtime_values = (
            ("timeout_seconds", runtime.timeout_seconds, lambda value: value >= 0.0),
            ("retry_backoff_seconds", runtime.retry_backoff_seconds, lambda value: value >= 0.0),
            (
                "pool_window_size_seconds",
                runtime.pool_window_size_seconds,
                lambda value: value > 0.0,
            ),
            (
                "pool_window_stride_seconds",
                runtime.pool_window_stride_seconds,
                lambda value: value > 0.0,
            ),
            (
                "post_hysteresis_enter_confidence",
                runtime.post_hysteresis_enter_confidence,
                lambda value: 0.0 <= value <= 1.0,
            ),
            (
                "post_hysteresis_exit_confidence",
                runtime.post_hysteresis_exit_confidence,
                lambda value: 0.0 <= value <= 1.0,
            ),
            (
                "post_min_segment_duration_seconds",
                runtime.post_min_segment_duration_seconds,
                lambda value: value >= 0.0,
            ),
        )
        for name, value, predicate in runtime_values:
            if not math.isfinite(value) or not predicate(value):
                findings.append(
                    ReadinessFinding(
                        check="configuration",
                        reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                        severity=FailureSeverity.ERROR,
                        message=f"Invalid {profile_name} runtime setting {name}={value!r}.",
                        blocking=True,
                    )
                )
        if runtime.pool_window_stride_seconds > runtime.pool_window_size_seconds:
            findings.append(
                ReadinessFinding(
                    check="configuration",
                    reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                    severity=FailureSeverity.ERROR,
                    message=f"{profile_name} pooling stride cannot exceed its window size.",
                    blocking=True,
                )
            )
        if (
            runtime.max_timeout_retries < 0
            or runtime.max_transient_retries < 0
            or runtime.post_smoothing_window_frames <= 0
        ):
            findings.append(
                ReadinessFinding(
                    check="configuration",
                    reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                    severity=FailureSeverity.ERROR,
                    message=f"{profile_name} retry/smoothing counts are invalid.",
                    blocking=True,
                )
            )
    if (
        settings.training.random_state < 0
        or settings.nn.random_state < 0
        or settings.audio_read.max_retries < 0
        or settings.data_loader.max_workers <= 0
        or settings.data_loader.max_failed_files < 0
        or settings.data_loader.max_failures_per_reason < 0
        or settings.data_loader.min_remaining_per_class_split < 0
        or settings.nn.max_iter <= 0
        or settings.medium_training.max_windows_per_clip < 0
    ):
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="Training seeds, retries, workers, iterations, and budgets are invalid.",
                blocking=True,
            )
        )
    requested_device = settings.torch_runtime.device.split(":", 1)[0]
    if requested_device not in {"auto", "cpu", "cuda", "mps", "xpu"} or (
        settings.torch_runtime.dtype not in {"auto", "float16", "float32", "bfloat16"}
    ):
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="Torch device/dtype selector is unsupported.",
                blocking=True,
            )
        )
    if requested_device in {"cuda", "mps", "xpu"} and importlib.util.find_spec("torch") is None:
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.BACKEND_UNAVAILABLE,
                severity=FailureSeverity.ERROR,
                message=f"Requested device {requested_device!r} requires torch before backend load.",
                blocking=True,
            )
        )
    if (
        settings.nn.max_iter <= 0
        or settings.nn.alpha < 0.0
        or settings.nn.epsilon <= 0.0
        or any(size <= 0 for size in settings.nn.hidden_layer_sizes)
    ):
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="Classifier iteration, regularization, epsilon, and layer sizes are invalid.",
                blocking=True,
            )
        )
    if (
        settings.medium_training.min_window_std < 0.0
        or settings.medium_training.max_windows_per_clip < 0
    ):
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="Medium noise-control thresholds must be non-negative.",
                blocking=True,
            )
        )
    if settings.audio_read.max_retries <= 0 or settings.data_loader.max_workers <= 0:
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message="Audio retry and dataset worker counts must be positive.",
                blocking=True,
            )
        )
    try:
        QuarantinePolicy.from_settings(settings)
    except ValueError as error:
        findings.append(
            ReadinessFinding(
                check="configuration",
                reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                severity=FailureSeverity.ERROR,
                message=str(error),
                blocking=True,
            )
        )
    return findings


def _restricted_backend_access_findings(settings: AppConfig) -> list[ReadinessFinding]:
    """Validates restricted backend policy before any backend construction."""
    if resolve_profile_name(settings) != "accurate-research":
        return []
    from ser._internal.license_check import (  # noqa: TID251
        BackendLicensePolicyError,
        ensure_backend_access,
        load_persisted_backend_consents,
        parse_allowed_restricted_backends_env,
    )

    try:
        ensure_backend_access(
            backend_id="emotion2vec",
            restricted_backends_enabled=settings.runtime_flags.restricted_backends,
            allowed_restricted_backends=parse_allowed_restricted_backends_env(),
            persisted_consents=load_persisted_backend_consents(settings=settings),
        )
    except (BackendLicensePolicyError, OSError, ValueError) as error:
        return [
            ReadinessFinding(
                check="restricted_backend_access",
                reason_code=FailureReasonCode.BACKEND_UNAVAILABLE,
                severity=FailureSeverity.ERROR,
                message=str(error),
                blocking=True,
            )
        ]
    return []


def _registry_findings(settings: AppConfig) -> list[ReadinessFinding]:
    snapshot = collect_dataset_registry_snapshot(settings=settings)
    return [
        ReadinessFinding(
            check="dataset_registry",
            reason_code=(
                FailureReasonCode.GIT_LFS_POINTER
                if "lfs" in issue.code.lower() or "lfs" in issue.message.lower()
                else FailureReasonCode.REGISTRY_UNHEALTHY
            ),
            severity=FailureSeverity.ERROR,
            message=f"{issue.dataset_id}: {issue.message}",
            blocking=True,
            scope=FailureScope.CORPUS,
        )
        for issue in snapshot.issues
    ]


def _allowed_media_roots(settings: AppConfig) -> tuple[Path, ...]:
    """Returns canonical roots that manifest media paths are permitted to inhabit."""
    snapshot = collect_dataset_registry_snapshot(settings=settings)
    if snapshot.entries:
        return tuple(
            sorted({entry.dataset_root.expanduser() for entry in snapshot.entries}, key=str)
        )
    roots = {settings.dataset.folder.expanduser()}
    roots.update(path.expanduser().parent for path in settings.dataset.manifest_paths)
    return tuple(sorted(roots, key=str))


def _contain_readiness_sample_failures(
    *,
    settings: AppConfig,
    utterances: Sequence[Utterance],
    findings: Sequence[ReadinessFinding],
    occurred_at: str,
    allowed_roots: Sequence[Path] | None = None,
) -> tuple[list[ReadinessFinding], tuple[QuarantineRecord, ...], tuple[Utterance, ...]]:
    """Converts allowlisted isolated media findings into budgeted durable exclusions."""
    started_at = time.perf_counter()
    total_findings = len(findings)
    last_progress_at = started_at
    budget_exceeded = 0
    logger.info(
        "DATASET_MEDIA_CONTAINMENT_START findings=%d samples=%d",
        total_findings,
        len(utterances),
    )
    by_id = {item.sample_id: item for item in utterances}
    policy = QuarantinePolicy.from_settings(settings)
    resolved_allowed_roots = (
        tuple(allowed_roots) if allowed_roots is not None else _allowed_media_roots(settings)
    )
    records: list[QuarantineRecord] = []
    resolved_findings: list[ReadinessFinding] = []
    local_reasons = {
        FailureReasonCode.MEDIA_MISSING,
        FailureReasonCode.MEDIA_EMPTY,
        FailureReasonCode.MEDIA_DECODE_FAILED,
    }

    def _maybe_log_containment_progress(processed: int) -> None:
        nonlocal last_progress_at
        progress_now = time.perf_counter()
        if _should_log_progress(
            processed=processed,
            total=total_findings,
            last_logged_at=last_progress_at,
            now=progress_now,
        ):
            logger.info(
                "DATASET_MEDIA_CONTAINMENT_PROGRESS processed=%d total=%d quarantined=%d "
                "budget_exceeded=%d elapsed=%.1fs",
                processed,
                total_findings,
                len(records),
                budget_exceeded,
                progress_now - started_at,
            )
            last_progress_at = progress_now

    for processed, finding in enumerate(findings, start=1):
        if (
            finding.scope is not FailureScope.SAMPLE
            or finding.reason_code not in local_reasons
            or finding.sample_id is None
        ):
            resolved_findings.append(finding)
            _maybe_log_containment_progress(processed)
            continue
        sample = by_id.get(finding.sample_id)
        if sample is None:
            resolved_findings.append(finding)
            _maybe_log_containment_progress(processed)
            continue
        error: Exception = (
            FileNotFoundError(errno.ENOENT, finding.message, str(sample.audio_path))
            if finding.reason_code is FailureReasonCode.MEDIA_MISSING
            else AudioDecodeError(finding.message)
        )
        classification = classify_failure(
            error,
            scope=FailureScope.SAMPLE,
            sample=sample,
            allowed_roots=resolved_allowed_roots,
        )
        try:
            enforce_quarantine_budget(
                policy=policy,
                all_samples=utterances,
                existing_records=records,
                candidate=sample,
                classification=classification,
            )
        except QuarantineBudgetExceeded as error:
            budget_exceeded += 1
            resolved_findings.append(finding)
            resolved_findings.append(
                ReadinessFinding(
                    check="quarantine_budget",
                    reason_code=FailureReasonCode.QUARANTINE_BUDGET_EXCEEDED,
                    severity=FailureSeverity.ERROR,
                    message=f"{sample.sample_id}: {error}",
                    blocking=True,
                    scope=FailureScope.RUN,
                    sample_id=sample.sample_id,
                    corpus=sample.corpus,
                )
            )
            _maybe_log_containment_progress(processed)
            continue
        record = build_quarantine_record(
            sample=sample,
            classification=classification,
            occurred_at=occurred_at,
            retry_count=0,
        )
        records.append(record)
        resolved_findings.append(
            replace(
                finding,
                severity=FailureSeverity.WARNING,
                message=f"{finding.message} Quarantined within all configured budgets.",
                blocking=False,
            )
        )
        _maybe_log_containment_progress(processed)
    quarantined_ids = {record.sample_id for record in records}
    effective = tuple(item for item in utterances if item.sample_id not in quarantined_ids)
    logger.info(
        "DATASET_MEDIA_CONTAINMENT_DONE findings=%d quarantined=%d effective=%d "
        "budget_exceeded=%d elapsed=%.1fs",
        total_findings,
        len(records),
        len(effective),
        budget_exceeded,
        time.perf_counter() - started_at,
    )
    return resolved_findings, tuple(records), effective


def _media_findings(
    utterances: Sequence[Utterance],
    *,
    allowed_roots: Sequence[Path],
) -> list[ReadinessFinding]:
    findings: list[ReadinessFinding] = []
    seen_ids: set[str] = set()
    seen_paths: dict[tuple[Path, float | None, float | None], str] = {}
    seen_hashes: dict[tuple[str, float | None, float | None], str] = {}
    started_at = time.perf_counter()
    sorted_utterances = sorted(utterances, key=lambda row: row.sample_id)
    total = len(sorted_utterances)
    hashed_count = 0
    last_progress_at = started_at
    logger.info("DATASET_MEDIA_CHECK_START samples=%d", total)

    def _maybe_log_media_progress(processed: int, item: Utterance, *, stage: str) -> None:
        nonlocal last_progress_at
        progress_now = time.perf_counter()
        if _should_log_progress(
            processed=processed,
            total=total,
            last_logged_at=last_progress_at,
            now=progress_now,
        ):
            logger.info(
                "DATASET_MEDIA_PROGRESS checked=%d total=%d hashed=%d findings=%d "
                "elapsed=%.1fs sample=%s corpus=%s stage=%s",
                processed,
                total,
                hashed_count,
                len(findings),
                progress_now - started_at,
                item.sample_id[:80],
                item.corpus[:80],
                stage,
            )
            last_progress_at = progress_now

    for processed, item in enumerate(sorted_utterances, start=1):
        if item.sample_id in seen_ids:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.DUPLICATE_SAMPLE_ID,
                    severity=FailureSeverity.ERROR,
                    message=f"Duplicate sample ID {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.CORPUS,
                )
            )
        seen_ids.add(item.sample_id)
        path = item.audio_path.expanduser()
        resolved_candidate = path.resolve(strict=False)
        if not any(
            resolved_candidate.is_relative_to(root.resolve(strict=False)) for root in allowed_roots
        ):
            findings.append(
                ReadinessFinding(
                    check="manifest_integrity",
                    reason_code=FailureReasonCode.MANIFEST_INVALID,
                    severity=FailureSeverity.ERROR,
                    message=f"Media path escapes every allowed dataset root for {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.CORPUS,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="allowed_roots")
            continue
        if not path.exists():
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_MISSING,
                    severity=FailureSeverity.ERROR,
                    message=f"Media is missing for sample {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="exists")
            continue
        if not path.is_file():
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_NOT_REGULAR,
                    severity=FailureSeverity.ERROR,
                    message=f"Media is not a regular file for sample {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="is_file")
            continue
        file_size = path.stat().st_size
        if file_size == 0:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_EMPTY,
                    severity=FailureSeverity.ERROR,
                    message=f"Media is empty for sample {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="size")
            continue
        with path.open("rb") as handle:
            if handle.read(len(_GIT_LFS_PREFIX)) == _GIT_LFS_PREFIX:
                findings.append(
                    ReadinessFinding(
                        check="dataset_media",
                        reason_code=FailureReasonCode.GIT_LFS_POINTER,
                        severity=FailureSeverity.ERROR,
                        message=f"Git LFS pointer found in corpus {item.corpus!r}.",
                        blocking=True,
                        scope=FailureScope.CORPUS,
                        sample_id=item.sample_id,
                        corpus=item.corpus,
                    )
                )
                _maybe_log_media_progress(processed, item, stage="lfs_pointer")
                continue
        resolved_path = path.resolve()
        media_identity = (resolved_path, item.start_seconds, item.duration_seconds)
        if media_identity in seen_paths:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.PATH_ALIAS,
                    severity=FailureSeverity.ERROR,
                    message=(
                        f"Samples {seen_paths[media_identity]!r} and {item.sample_id!r} "
                        "resolve to the same media path."
                    ),
                    blocking=True,
                    scope=FailureScope.CORPUS,
                )
            )
        else:
            seen_paths[media_identity] = item.sample_id
        try:
            info = sf.info(str(path))
        except PermissionError as error:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_DECODE_FAILED,
                    severity=FailureSeverity.ERROR,
                    message=f"Media permission failure for {item.sample_id!r}: {error}",
                    blocking=True,
                    scope=FailureScope.RUN,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="metadata")
            continue
        except (OSError, RuntimeError) as error:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_DECODE_FAILED,
                    severity=FailureSeverity.ERROR,
                    message=f"Media metadata decode failed for {item.sample_id!r}: {error}",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
            _maybe_log_media_progress(processed, item, stage="metadata")
            continue
        if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_DECODE_FAILED,
                    severity=FailureSeverity.ERROR,
                    message=f"Invalid audio metadata for sample {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
        media_duration = float(info.frames) / float(info.samplerate)
        segment_end = float(item.start_seconds or 0.0) + float(
            item.duration_seconds or media_duration
        )
        if segment_end > media_duration + 1e-3:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.MEDIA_DECODE_FAILED,
                    severity=FailureSeverity.ERROR,
                    message=f"Segment bounds exceed media duration for {item.sample_id!r}.",
                    blocking=True,
                    scope=FailureScope.SAMPLE,
                    sample_id=item.sample_id,
                    corpus=item.corpus,
                )
            )
        if item.normalized_audio_sha256:
            content_hash = item.normalized_audio_sha256
        else:
            if file_size >= 25 * 1024 * 1024:
                logger.info(
                    "DATASET_MEDIA_HASH_START sample=%s size=%d path=%s",
                    item.sample_id[:80],
                    file_size,
                    path,
                )
            hash_started_at = time.perf_counter()
            content_hash = normalized_pcm_digest(path, sample_id=item.sample_id)
            hashed_count += 1
            if file_size >= 25 * 1024 * 1024:
                logger.info(
                    "DATASET_MEDIA_HASH_DONE sample=%s size=%d elapsed=%.1fs",
                    item.sample_id[:80],
                    file_size,
                    time.perf_counter() - hash_started_at,
                )
        content_identity = (content_hash, item.start_seconds, item.duration_seconds)
        if content_identity in seen_hashes:
            findings.append(
                ReadinessFinding(
                    check="dataset_media",
                    reason_code=FailureReasonCode.DUPLICATE_CONTENT,
                    severity=FailureSeverity.ERROR,
                    message=(
                        f"Samples {seen_hashes[content_identity]!r} and {item.sample_id!r} "
                        "have duplicate normalized content."
                    ),
                    blocking=True,
                    scope=FailureScope.CORPUS,
                )
            )
        else:
            seen_hashes[content_identity] = item.sample_id
        _maybe_log_media_progress(processed, item, stage="hash")
    logger.info(
        "DATASET_MEDIA_CHECK_DONE samples=%d hashed=%d findings=%d elapsed=%.1fs",
        total,
        hashed_count,
        len(findings),
        time.perf_counter() - started_at,
    )
    return findings


def _split_findings(
    utterances: Sequence[Utterance],
    *,
    settings: AppConfig,
) -> list[ReadinessFinding]:
    findings: list[ReadinessFinding] = []
    labels = Counter(item.label for item in utterances if item.label is not None)
    if len(labels) < 2 or any(count < 2 for count in labels.values()):
        findings.append(
            ReadinessFinding(
                check="split_feasibility",
                reason_code=FailureReasonCode.INSUFFICIENT_CLASS_SUPPORT,
                severity=FailureSeverity.ERROR,
                message="At least two classes with two or more utterances each are required.",
                blocking=True,
            )
        )
    else:
        from ser._internal.models.dataset_splitting import (  # noqa: TID251
            split_utterances_three_way,
        )

        try:
            train_samples, dev_samples, test_samples, split_metadata = split_utterances_three_way(
                samples=list(utterances),
                settings=settings,
                logger=logging.getLogger(__name__),
            )
            minimum = settings.data_loader.min_remaining_per_class_split
            for partition_name, partition in (
                ("train", train_samples),
                ("dev", dev_samples),
                ("test", test_samples),
            ):
                support = {
                    label: sum(item.label == label for item in partition) for label in labels
                }
                if any(count < minimum for count in support.values()):
                    findings.append(
                        ReadinessFinding(
                            check="split_feasibility",
                            reason_code=FailureReasonCode.INSUFFICIENT_CLASS_SUPPORT,
                            severity=FailureSeverity.ERROR,
                            message=(
                                f"{partition_name} split violates minimum class support: {support}."
                            ),
                            blocking=True,
                        )
                    )
            if split_metadata.speaker_overlap_count > 0:
                findings.append(
                    ReadinessFinding(
                        check="split_feasibility",
                        reason_code=FailureReasonCode.SPLIT_LEAKAGE,
                        severity=FailureSeverity.ERROR,
                        message="Deterministic split contains overlapping speakers.",
                        blocking=True,
                    )
                )
        except (RuntimeError, ValueError) as error:
            findings.append(
                ReadinessFinding(
                    check="split_feasibility",
                    reason_code=FailureReasonCode.INSUFFICIENT_CLASS_SUPPORT,
                    severity=FailureSeverity.ERROR,
                    message=str(error),
                    blocking=True,
                )
            )
    for identity_name in ("speaker_id", "session_id", "normalized_audio_sha256"):
        partition_by_identity: dict[str, set[str]] = defaultdict(set)
        for item in utterances:
            identity = getattr(item, identity_name)
            if identity is not None and item.split is not None:
                partition_by_identity[str(identity)].add(str(item.split))
        leaked = sorted(
            key for key, partitions in partition_by_identity.items() if len(partitions) > 1
        )
        if leaked:
            findings.append(
                ReadinessFinding(
                    check="split_feasibility",
                    reason_code=FailureReasonCode.SPLIT_LEAKAGE,
                    severity=FailureSeverity.ERROR,
                    message=f"{identity_name} leakage across protected partitions ({len(leaked)}).",
                    blocking=True,
                )
            )
    return findings


def _probe_directory(path: Path) -> None:
    """Checks write and atomic-rename behavior without retaining probe artifacts."""
    existing_parent = path.expanduser()
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir() or not os.access(existing_parent, os.W_OK):
        raise OSError(f"Required path parent is not writable: {path}")
    probe_dir = path if path.is_dir() else existing_parent
    descriptor, raw_source = tempfile.mkstemp(prefix=".ser-write-probe-", dir=probe_dir)
    source = Path(raw_source)
    destination = source.with_suffix(".renamed")
    try:
        os.write(descriptor, b"ser")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(source, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        source.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)


def _resource_findings(settings: AppConfig, sample_count: int) -> list[ReadinessFinding]:
    findings: list[ReadinessFinding] = []
    required_paths = {
        settings.models.folder,
        settings.tmp_folder,
        settings.models.training_report_file.parent,
        settings.models.model_cache_dir,
    }
    for path in sorted(required_paths, key=str):
        try:
            _probe_directory(path)
        except OSError as error:
            findings.append(
                ReadinessFinding(
                    check="filesystem_resources",
                    reason_code=FailureReasonCode.OUTPUT_UNWRITABLE,
                    severity=FailureSeverity.ERROR,
                    message=str(error),
                    blocking=True,
                )
            )
    existing_tmp_parent = settings.tmp_folder
    while not existing_tmp_parent.exists() and existing_tmp_parent != existing_tmp_parent.parent:
        existing_tmp_parent = existing_tmp_parent.parent
    free_bytes = shutil.disk_usage(existing_tmp_parent).free
    estimated_bytes = max(128 * 1024 * 1024, sample_count * 4 * 1024 * 1024)
    if free_bytes < estimated_bytes * 2:
        findings.append(
            ReadinessFinding(
                check="filesystem_resources",
                reason_code=FailureReasonCode.DISK_SPACE_LOW,
                severity=FailureSeverity.ERROR,
                message=(
                    f"Available disk ({free_bytes} bytes) is below conservative requirement "
                    f"({estimated_bytes * 2} bytes)."
                ),
                blocking=True,
            )
        )
    try:
        soft_fd_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (OSError, ValueError):
        soft_fd_limit = resource.RLIM_INFINITY
    required_fds = max(64, settings.data_loader.max_workers * 4)
    if soft_fd_limit != resource.RLIM_INFINITY and soft_fd_limit < required_fds:
        findings.append(
            ReadinessFinding(
                check="filesystem_resources",
                reason_code=FailureReasonCode.RESOURCE_LIMIT,
                severity=FailureSeverity.ERROR,
                message=f"File descriptor limit {soft_fd_limit} is below required {required_fds}.",
                blocking=True,
            )
        )
    try:
        available_memory = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, TypeError):
        available_memory = 0
    required_memory = max(512 * 1024 * 1024, sample_count * 2 * 1024 * 1024)
    if available_memory and available_memory < required_memory:
        findings.append(
            ReadinessFinding(
                check="filesystem_resources",
                reason_code=FailureReasonCode.RESOURCE_LIMIT,
                severity=FailureSeverity.ERROR,
                message=(
                    f"Available memory {available_memory} is below conservative requirement "
                    f"{required_memory}."
                ),
                blocking=True,
            )
        )
    return findings


def _backend_failure_reason(error: Exception) -> FailureReasonCode:
    """Maps known load/dependency failures without misclassifying output-contract defects."""
    if isinstance(error, TimeoutError):
        return FailureReasonCode.BACKEND_SMOKE_TIMEOUT
    if isinstance(error, ImportError):
        return FailureReasonCode.BACKEND_UNAVAILABLE
    message = str(error).lower()
    if isinstance(error, RuntimeError) and any(
        marker in message
        for marker in ("dependency", "unavailable", "not installed", "failed to load", "model load")
    ):
        return FailureReasonCode.BACKEND_UNAVAILABLE
    return FailureReasonCode.BACKEND_OUTPUT_INVALID


def _network_repairs_allowed() -> bool:
    """Returns whether explicit policy permits repair-time network access."""
    return os.getenv("SER_TRAINING_REPAIR_ALLOW_NETWORK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _run_repair_command(
    command: Sequence[str], *, timeout_seconds: float = 120.0
) -> tuple[bool, str]:
    """Runs one checked repair command with timeout and bounded diagnostics."""
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)[:2000]
    detail = (completed.stderr or completed.stdout or "completed").strip()[:2000]
    return completed.returncode == 0, detail


def _git_root(path: Path) -> Path | None:
    """Returns the nearest compatible Git checkout root."""
    candidate = path.expanduser().resolve(strict=False)
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _apply_pre_validation_repairs(settings: AppConfig) -> list[RepairRecord]:
    """Applies every safe local repair before running the same validation pass."""
    started_at = time.perf_counter()
    logger.info("REPAIR_START")
    records: list[RepairRecord] = []

    def _record(record: RepairRecord) -> None:
        records.append(record)
        _log_repair_record(record)

    owned_directories = {
        settings.tmp_folder,
        settings.models.folder,
        settings.models.model_cache_dir,
        settings.models.training_report_file.parent,
    }
    for path in sorted(owned_directories, key=str):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            _record(RepairRecord("create_application_directory", str(path), False, str(error)))
        else:
            _record(
                RepairRecord(
                    "create_application_directory",
                    str(path),
                    True,
                    "Directory is present.",
                )
            )

    staging_patterns = (".ser-write-probe-*", ".prepared-training-*.*", ".training-readiness-*.*")
    removed_staging = 0
    staging_failures = 0
    if settings.tmp_folder.is_dir():
        for pattern in staging_patterns:
            for path in settings.tmp_folder.rglob(pattern):
                if path.is_file():
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        staging_failures += 1
                    else:
                        removed_staging += 1
    _record(
        RepairRecord(
            "clean_application_staging",
            str(settings.tmp_folder),
            staging_failures == 0,
            (
                f"Removed {removed_staging} abandoned staging/probe file(s); "
                f"failures={staging_failures}."
            ),
        )
    )

    invalidated = 0
    invalidation_failures = 0
    cache_roots: list[Path] = []
    for namespace in PROFILE_CACHE_NAMESPACES.values():
        try:
            cache_roots.append(validated_cache_root(settings, namespace))
        except PreparedPlanError:
            invalidation_failures += 1
    for cache_root in cache_roots:
        if not cache_root.is_dir():
            continue
        for cache_path in cache_root.rglob("*.npz"):
            try:
                with np.load(cache_path, allow_pickle=False) as payload:
                    _ = tuple(payload.files)
            except (OSError, ValueError, EOFError):
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    invalidation_failures += 1
                else:
                    invalidated += 1
    _record(
        RepairRecord(
            "invalidate_derived_cache",
            ",".join(str(path) for path in cache_roots),
            invalidation_failures == 0,
            (
                f"Invalidated {invalidated} corrupt derived cache entrie(s); "
                f"failures={invalidation_failures}."
            ),
        )
    )

    from ser._internal.data.dataset_prepare import (  # noqa: TID251
        prepare_from_registry_entry,
    )
    from ser._internal.data.dataset_registry import load_dataset_registry  # noqa: TID251
    from ser._internal.data.label_ontology import resolve_label_ontology  # noqa: TID251

    registry = load_dataset_registry(settings=settings)
    ontology = resolve_label_ontology(settings)
    for entry in sorted(registry.values(), key=lambda item: item.dataset_id):
        if entry.manifest_path.expanduser().is_file():
            continue
        try:
            built = prepare_from_registry_entry(settings=settings, entry=entry, ontology=ontology)
            succeeded = bool(built) and all(path.is_file() for path in built)
            detail = (
                f"Built {len(built)} manifest(s)." if succeeded else "No manifest was produced."
            )
        except (OSError, RuntimeError, ValueError) as error:
            succeeded = False
            detail = str(error)[:2000]
        _record(RepairRecord("rebuild_manifest", entry.dataset_id, succeeded, detail))

    registry_snapshot = collect_dataset_registry_snapshot(settings=settings)
    lfs_dataset_ids = {
        issue.dataset_id
        for issue in registry_snapshot.issues
        if "lfs" in issue.code.lower() or "lfs" in issue.message.lower()
    }
    git_roots = {
        root
        for entry in registry.values()
        if entry.dataset_id in lfs_dataset_ids
        and (root := _git_root(entry.dataset_root)) is not None
    }
    for root in sorted(git_roots, key=str):
        succeeded, detail = _run_repair_command(("git", "-C", str(root), "lfs", "checkout"))
        if not succeeded and _network_repairs_allowed():
            pulled, pull_detail = _run_repair_command(("git", "-C", str(root), "lfs", "pull"))
            if pulled:
                succeeded, detail = _run_repair_command(("git", "-C", str(root), "lfs", "checkout"))
            else:
                detail = pull_detail
        _record(RepairRecord("hydrate_git_lfs", str(root), succeeded, detail))
    logger.info(
        "REPAIR_DONE actions=%d failures=%d elapsed=%.1fs",
        len(records),
        sum(not record.succeeded for record in records),
        time.perf_counter() - started_at,
    )
    return records


@contextmanager
def _network_repair_deadline(seconds: float) -> Iterator[None]:
    """Hard-bounds a main-thread network repair or rejects unsupported execution."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Network repair requires main-thread hard-deadline support.")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "ITIMER_REAL"):
        raise RuntimeError("Network repair hard-deadline support is unavailable.")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError(f"Pinned model repair exceeded {seconds:.0f} seconds.")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _repair_pinned_model(settings: AppConfig) -> RepairRecord:
    """Redownloads only an explicitly revision-pinned Hugging Face model."""
    if not _network_repairs_allowed():
        return RepairRecord(
            "redownload_pinned_model",
            resolve_profile_name(settings),
            False,
            "Network policy denied repair; set SER_TRAINING_REPAIR_ALLOW_NETWORK=1 explicitly.",
        )
    profile = resolve_profile_name(settings)
    model_id = (
        settings.models.medium_model_id
        if profile == "medium"
        else settings.models.accurate_model_id if profile == "accurate" else ""
    )
    if "@" not in model_id:
        return RepairRecord(
            "redownload_pinned_model",
            model_id or profile,
            False,
            "Model repair requires an exact '<repo>@<revision>' pin.",
        )
    repo_id, revision = model_id.rsplit("@", 1)
    if not repo_id or not revision:
        return RepairRecord("redownload_pinned_model", model_id, False, "Model pin is invalid.")
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        return RepairRecord("redownload_pinned_model", model_id, False, str(error)[:2000])
    try:
        live_cache = settings.models.huggingface_cache_root.resolve(strict=False)
        live_cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".pinned-model-repair-", dir=live_cache.parent
        ) as raw_staging:
            staging_root = Path(raw_staging)
            staging = staging_root / "cache"
            if live_cache.is_dir():
                shutil.copytree(live_cache, staging)
            else:
                staging.mkdir()
            with _network_repair_deadline(300.0):
                resolved_snapshot = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        revision=revision,
                        cache_dir=staging,
                        force_download=True,
                        etag_timeout=30,
                        max_workers=4,
                    )
                )
                if (
                    not resolved_snapshot.resolve(strict=False).is_relative_to(
                        staging.resolve(strict=False)
                    )
                    or not resolved_snapshot.is_dir()
                ):
                    raise RuntimeError("Downloaded model snapshot did not validate inside staging.")
                backup = Path(
                    tempfile.mkdtemp(prefix=f".{live_cache.name}.rollback-", dir=live_cache.parent)
                )
                backup.rmdir()
                moved_live = False
                try:
                    if live_cache.exists():
                        os.replace(live_cache, backup)
                        moved_live = True
                    os.replace(staging, live_cache)
                except BaseException:
                    if moved_live and backup.exists() and not live_cache.exists():
                        os.replace(backup, live_cache)
                    raise
                else:
                    if backup.exists():
                        shutil.rmtree(backup)
    except (HfHubHTTPError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        return RepairRecord("redownload_pinned_model", model_id, False, str(error)[:2000])
    return RepairRecord(
        "redownload_pinned_model",
        model_id,
        True,
        "Pinned model snapshot redownloaded.",
    )


def run_training_readiness(
    *,
    settings: AppConfig,
    load_utterances: Callable[[], list[Utterance] | None],
    smoke_runner: SmokeRunner | None = None,
    smoke_sample_cap: int = DEFAULT_SMOKE_SAMPLE_CAP,
    repair: bool = False,
    persist_quarantine_ledger: bool = True,
    report_path: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[ReadinessReport, tuple[Utterance, ...]]:
    """Runs the shared deterministic readiness contract and atomically reports it."""
    started_at = time.perf_counter()
    profile = resolve_profile_name(settings)
    logger.info(
        "READINESS_START profile=%s repair=%s smoke=%s report_path=%s",
        profile,
        repair,
        smoke_runner is not None,
        report_path or default_readiness_report_path(settings),
    )
    created_at = now().astimezone(UTC).isoformat()
    findings: list[ReadinessFinding] = []
    repairs: list[RepairRecord] = []
    resolved_utterances: tuple[Utterance, ...] = ()
    effective_utterances: tuple[Utterance, ...] = ()
    selected: tuple[Utterance, ...] = ()
    smoke: SmokeResult | None = None
    quarantine_records: tuple[QuarantineRecord, ...] = ()
    destination = report_path or default_readiness_report_path(settings)

    try:
        if repair:
            repair_started_at = time.perf_counter()
            repairs = _apply_pre_validation_repairs(settings)
            repair_findings = [
                ReadinessFinding(
                    check="repair",
                    reason_code=FailureReasonCode.REPAIR_FAILED,
                    severity=FailureSeverity.ERROR,
                    message=f"Repair {repair_record.action!r} failed: {repair_record.message}",
                    blocking=True,
                )
                for repair_record in repairs
                if not repair_record.succeeded
            ]
            findings.extend(repair_findings)
            repair_status = _readiness_status(repair_findings)
            logger.log(
                _readiness_log_level(repair_status),
                "READINESS_PHASE_DONE phase=repair status=%s findings=%d elapsed=%.1fs",
                repair_status,
                len(repair_findings),
                time.perf_counter() - repair_started_at,
            )

        phase_started_at = time.perf_counter()
        logger.info("READINESS_PHASE_START phase=configuration")
        configuration_findings = _configuration_findings(settings)
        configuration_findings.extend(_restricted_backend_access_findings(settings))
        findings.extend(configuration_findings)
        configuration_status = _readiness_status(configuration_findings)
        logger.log(
            _readiness_log_level(configuration_status),
            "READINESS_PHASE_DONE phase=configuration status=%s findings=%d elapsed=%.1fs",
            configuration_status,
            len(configuration_findings),
            time.perf_counter() - phase_started_at,
        )

        phase_started_at = time.perf_counter()
        logger.info("READINESS_PHASE_START phase=dataset_registry")
        registry_findings = _registry_findings(settings)
        findings.extend(registry_findings)
        registry_status = _readiness_status(registry_findings)
        logger.log(
            _readiness_log_level(registry_status),
            "READINESS_PHASE_DONE phase=dataset_registry status=%s findings=%d elapsed=%.1fs",
            registry_status,
            len(registry_findings),
            time.perf_counter() - phase_started_at,
        )

        phase_started_at = time.perf_counter()
        logger.info("READINESS_PHASE_START phase=manifest_load")
        try:
            utterances = load_utterances()
        except (OSError, RuntimeError, ValueError) as error:
            utterances = None
            findings.append(
                ReadinessFinding(
                    check="manifest_integrity",
                    reason_code=(
                        FailureReasonCode.GIT_LFS_POINTER
                        if "git lfs" in str(error).lower()
                        else FailureReasonCode.MANIFEST_INVALID
                    ),
                    severity=FailureSeverity.ERROR,
                    message=str(error),
                    blocking=True,
                    scope=(
                        FailureScope.CORPUS if "git lfs" in str(error).lower() else FailureScope.RUN
                    ),
                )
            )
        resolved_utterances = tuple(utterances or ())
        manifest_findings = [
            finding for finding in findings if finding.check == "manifest_integrity"
        ]
        if not resolved_utterances:
            empty_dataset_finding = ReadinessFinding(
                check="dataset_registry",
                reason_code=FailureReasonCode.DATASET_NOT_FOUND,
                severity=FailureSeverity.ERROR,
                message="No effective training utterances were resolved.",
                blocking=True,
            )
            findings.append(empty_dataset_finding)
            manifest_findings.append(empty_dataset_finding)
        manifest_status = _readiness_status(manifest_findings)
        logger.log(
            _readiness_log_level(manifest_status),
            "READINESS_PHASE_DONE phase=manifest_load status=%s utterances=%d findings=%d elapsed=%.1fs",
            manifest_status,
            len(resolved_utterances),
            len(manifest_findings),
            time.perf_counter() - phase_started_at,
        )

        effective_utterances = resolved_utterances
        if resolved_utterances:
            phase_started_at = time.perf_counter()
            logger.info(
                "READINESS_PHASE_START phase=dataset_media samples=%d",
                len(resolved_utterances),
            )
            roots_started_at = time.perf_counter()
            logger.info(
                "DATASET_MEDIA_ROOTS_START samples=%d",
                len(resolved_utterances),
            )
            allowed_roots = _allowed_media_roots(settings)
            logger.info(
                "DATASET_MEDIA_ROOTS_DONE roots=%d elapsed=%.1fs",
                len(allowed_roots),
                time.perf_counter() - roots_started_at,
            )
            media_findings = _media_findings(resolved_utterances, allowed_roots=allowed_roots)
            contained, quarantine_records, effective_utterances = (
                _contain_readiness_sample_failures(
                    settings=settings,
                    utterances=resolved_utterances,
                    findings=media_findings,
                    occurred_at=created_at,
                    allowed_roots=allowed_roots,
                )
            )
            findings.extend(contained)
            if quarantine_records and persist_quarantine_ledger:
                write_quarantine_ledger(
                    settings.tmp_folder / f"quarantine-{profile}.jsonl",
                    quarantine_records,
                )
            media_status = _readiness_status(contained)
            logger.log(
                _readiness_log_level(media_status),
                "READINESS_PHASE_DONE phase=dataset_media status=%s effective=%d quarantined=%d findings=%d elapsed=%.1fs",
                media_status,
                len(effective_utterances),
                len(quarantine_records),
                len(contained),
                time.perf_counter() - phase_started_at,
            )

            phase_started_at = time.perf_counter()
            logger.info(
                "READINESS_PHASE_START phase=split_feasibility samples=%d",
                len(effective_utterances),
            )
            split_findings = _split_findings(effective_utterances, settings=settings)
            findings.extend(split_findings)
            split_status = _readiness_status(split_findings)
            logger.log(
                _readiness_log_level(split_status),
                "READINESS_PHASE_DONE phase=split_feasibility status=%s findings=%d elapsed=%.1fs",
                split_status,
                len(split_findings),
                time.perf_counter() - phase_started_at,
            )

        phase_started_at = time.perf_counter()
        logger.info(
            "READINESS_PHASE_START phase=filesystem_resources effective=%d",
            len(effective_utterances),
        )
        resource_findings = _resource_findings(settings, len(effective_utterances))
        findings.extend(resource_findings)
        resource_status = _readiness_status(resource_findings)
        logger.log(
            _readiness_log_level(resource_status),
            "READINESS_PHASE_DONE phase=filesystem_resources status=%s findings=%d elapsed=%.1fs",
            resource_status,
            len(resource_findings),
            time.perf_counter() - phase_started_at,
        )

        phase_started_at = time.perf_counter()
        logger.info("READINESS_PHASE_START phase=backend_smoke")
        try:
            selected = (
                select_smoke_samples(effective_utterances, cap=smoke_sample_cap)
                if effective_utterances
                else ()
            )
        except ValueError as error:
            findings.append(
                ReadinessFinding(
                    check="backend_smoke_selection",
                    reason_code=FailureReasonCode.INVALID_CONFIGURATION,
                    severity=FailureSeverity.ERROR,
                    message=str(error),
                    blocking=True,
                )
            )
        smoke_findings_start = len(findings)
        if smoke_runner is not None and not any(finding.blocking for finding in findings):
            probe_parent = settings.tmp_folder
            probe_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="ser-smoke-", dir=probe_parent) as probe_dir:
                try:
                    smoke = smoke_runner(
                        settings=settings,
                        samples=selected,
                        probe_cache_dir=Path(probe_dir),
                    )
                    if (
                        smoke.attempted != len(selected)
                        or smoke.succeeded != smoke.attempted
                        or smoke.feature_dim <= 0
                        or not smoke.cache_round_trip
                    ):
                        raise ValueError(
                            "Backend smoke result violates the feature/cache contract."
                        )
                except (ImportError, OSError, RuntimeError, ValueError) as error:
                    findings.append(
                        ReadinessFinding(
                            check="backend_smoke",
                            reason_code=_backend_failure_reason(error),
                            severity=FailureSeverity.ERROR,
                            message=str(error),
                            blocking=True,
                        )
                    )
        smoke_findings = findings[smoke_findings_start:]
        smoke_status = _readiness_status(smoke_findings)
        logger.log(
            _readiness_log_level(smoke_status),
            "READINESS_PHASE_DONE phase=backend_smoke status=%s selected=%d findings=%d elapsed=%.1fs",
            smoke_status,
            len(selected),
            len(smoke_findings),
            time.perf_counter() - phase_started_at,
        )

        phase_started_at = time.perf_counter()
        logger.info("READINESS_PHASE_START phase=inventory_digest")
        if effective_utterances and not any(
            finding.reason_code
            in {
                FailureReasonCode.MEDIA_NOT_REGULAR,
                FailureReasonCode.MANIFEST_INVALID,
                FailureReasonCode.GIT_LFS_POINTER,
            }
            and finding.blocking
            for finding in findings
        ):
            registry_digest, manifest_digest, media_digest = _inventory_digests(
                settings,
                resolved_utterances,
                media_utterances=resolved_utterances,
            )
        else:
            snapshot = collect_dataset_registry_snapshot(settings=settings)
            registry_digest = digest_payload(
                {
                    "entries": [asdict(item) for item in snapshot.entries],
                    "issues": [asdict(item) for item in snapshot.issues],
                }
            )
            manifest_digest = digest_payload([])
            media_digest = digest_payload([])
        logger.info(
            "READINESS_PHASE_DONE phase=inventory_digest status=PASS elapsed=%.1fs",
            time.perf_counter() - phase_started_at,
        )

        report = ReadinessReport(
            schema_version=READINESS_SCHEMA_VERSION,
            created_at=created_at,
            profile=profile,
            settings_digest=settings_digest(settings),
            registry_digest=registry_digest,
            manifest_digest=manifest_digest,
            media_digest=media_digest,
            selected_sample_ids=tuple(item.sample_id for item in selected),
            findings=tuple(findings),
            repairs=tuple(repairs),
            smoke=smoke,
            quarantined_samples=len(quarantine_records),
            quarantine=quarantine_records,
            effective_sample_ids=tuple(item.sample_id for item in effective_utterances),
        )
        if repair and any(
            finding.reason_code is FailureReasonCode.BACKEND_UNAVAILABLE and finding.blocking
            for finding in findings
        ):
            logger.info("REPAIR_START action=redownload_pinned_model")
            model_repair = _repair_pinned_model(settings)
            _log_repair_record(model_repair)
            revalidated, revalidated_utterances = run_training_readiness(
                settings=settings,
                load_utterances=load_utterances,
                smoke_runner=smoke_runner,
                smoke_sample_cap=smoke_sample_cap,
                repair=False,
                report_path=destination,
                now=now,
            )
            report = replace(
                revalidated,
                repairs=tuple([*repairs, model_repair]),
            )
            atomic_write_json(destination, report.to_dict())
            logger.info(
                "READINESS_DONE profile=%s ready=%s findings=%d repairs=%d report_path=%s elapsed=%.1fs",
                report.profile,
                report.ready,
                len(report.findings),
                len(report.repairs),
                destination,
                time.perf_counter() - started_at,
            )
            return report, revalidated_utterances
        atomic_write_json(destination, report.to_dict())
        logger.info(
            "READINESS_DONE profile=%s ready=%s findings=%d repairs=%d report_path=%s elapsed=%.1fs",
            report.profile,
            report.ready,
            len(report.findings),
            len(report.repairs),
            destination,
            time.perf_counter() - started_at,
        )
        return report, effective_utterances
    except Exception:
        logger.exception(
            "READINESS_DONE profile=%s ready=false status=ERROR elapsed=%.1fs",
            profile,
            time.perf_counter() - started_at,
        )
        raise


def ensure_training_ready(
    *,
    settings: AppConfig,
    load_utterances: Callable[[], list[Utterance] | None],
    smoke_runner: SmokeRunner | None = None,
    repair: bool = False,
    report_path: Path | None = None,
) -> tuple[ReadinessReport, tuple[Utterance, ...]]:
    """Runs readiness and raises a typed validation error on any blocking finding."""
    report, utterances = run_training_readiness(
        settings=settings,
        load_utterances=load_utterances,
        smoke_runner=smoke_runner,
        repair=repair,
        report_path=report_path,
    )
    if not report.ready:
        raise TrainingReadinessError(report)
    return report, utterances


def build_prepared_plan(
    *,
    settings: AppConfig,
    readiness: ReadinessReport,
    backend_id: str,
    model_id: str,
    model_revision: str,
    device: str,
    dtype: str,
    recipe_digest: str,
    split_ledger_digest: str,
    quarantine_ledger_digest: str,
    cache_namespace: str,
    cache_version: str,
    cache_keys: Sequence[str],
    effective_counts: Mapping[str, Mapping[str, int]],
    feature_shape: tuple[int, int],
    feature_dtype: str,
    windowing_policy: Mapping[str, object],
    noise_statistics: Mapping[str, object],
    payload_path: Path,
    package_version: str,
    sample_ledger: Sequence[Mapping[str, object]] = (),
    window_ledger: Sequence[Mapping[str, object]] = (),
    disposition_counts: Mapping[str, int] | None = None,
    created_at: str | None = None,
) -> PreparedPlan:
    """Builds and signs a complete prepared-plan document."""
    if not readiness.ready:
        raise PreparedPlanError("Cannot build a prepared plan from failed readiness validation.")
    if not payload_path.is_file():
        raise PreparedPlanError(f"Prepared feature payload does not exist: {payload_path}")
    plan = PreparedPlan(
        schema_version=PREPARED_PLAN_SCHEMA_VERSION,
        created_at=created_at or datetime.now(UTC).isoformat(),
        code_version=PREPARATION_CODE_VERSION,
        package_version=package_version,
        profile=resolve_profile_name(settings),
        backend_id=backend_id,
        model_id=model_id,
        model_revision=model_revision,
        device=device,
        dtype=dtype,
        settings_digest=readiness.settings_digest,
        registry_digest=readiness.registry_digest,
        manifest_digest=readiness.manifest_digest,
        media_digest=readiness.media_digest,
        recipe_digest=recipe_digest,
        split_ledger_digest=split_ledger_digest,
        quarantine_ledger_digest=quarantine_ledger_digest,
        cache_namespace=cache_namespace,
        cache_version=cache_version,
        cache_keys=tuple(sorted(cache_keys)),
        effective_counts=effective_counts,
        sample_ledger=tuple(sample_ledger),
        window_ledger=tuple(window_ledger),
        disposition_counts=dict(disposition_counts or {}),
        feature_shape=feature_shape,
        feature_dtype=feature_dtype,
        windowing_policy=windowing_policy,
        noise_statistics=noise_statistics,
        validation_findings=tuple(finding.to_dict() for finding in readiness.findings),
        repairs=tuple(repair.to_dict() for repair in readiness.repairs),
        payload_path=str(payload_path.resolve()),
        payload_digest=hash_file(payload_path),
    )
    return replace(plan, overall_digest=digest_payload(plan.unsigned_dict()))


def write_prepared_plan(path: Path, plan: PreparedPlan) -> None:
    """Validates and atomically publishes a complete prepared plan."""
    if plan.overall_digest != digest_payload(plan.unsigned_dict()):
        raise PreparedPlanError("Prepared plan overall digest is invalid.")
    atomic_write_json(path, plan.to_dict())


def _validate_json_value(value: object, *, location: str) -> None:
    """Exhaustively validates one untrusted JSON subtree."""
    if value is None or isinstance(value, str | bool):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise PreparedPlanError(f"Prepared plan {location} contains a non-finite number.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise PreparedPlanError(f"Prepared plan {location} has an invalid key.")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise PreparedPlanError(
        f"Prepared plan {location} has unsupported type {type(value).__name__}."
    )


def _validate_prepared_plan_mapping(raw: dict[object, object]) -> dict[str, object]:
    """Validates every prepared-plan field before typed construction."""
    expected = {
        "schema_version",
        "created_at",
        "code_version",
        "package_version",
        "profile",
        "backend_id",
        "model_id",
        "model_revision",
        "device",
        "dtype",
        "settings_digest",
        "registry_digest",
        "manifest_digest",
        "media_digest",
        "recipe_digest",
        "split_ledger_digest",
        "quarantine_ledger_digest",
        "cache_namespace",
        "cache_version",
        "cache_keys",
        "effective_counts",
        "sample_ledger",
        "window_ledger",
        "disposition_counts",
        "feature_shape",
        "feature_dtype",
        "windowing_policy",
        "noise_statistics",
        "validation_findings",
        "repairs",
        "payload_path",
        "payload_digest",
        "overall_digest",
    }
    if any(not isinstance(key, str) for key in raw):
        raise PreparedPlanError("Prepared plan keys must all be strings.")
    normalized = cast(dict[str, object], raw)
    missing = expected - normalized.keys()
    unknown = normalized.keys() - expected
    if missing or unknown:
        raise PreparedPlanError(
            f"Prepared plan fields differ from schema: missing={sorted(missing)} unknown={sorted(unknown)}."
        )
    _validate_json_value(normalized, location="root")
    if not isinstance(normalized["schema_version"], int) or isinstance(
        normalized["schema_version"], bool
    ):
        raise PreparedPlanError("Prepared plan schema_version must be an integer.")
    string_fields = expected - {
        "schema_version",
        "cache_keys",
        "effective_counts",
        "sample_ledger",
        "window_ledger",
        "disposition_counts",
        "feature_shape",
        "windowing_policy",
        "noise_statistics",
        "validation_findings",
        "repairs",
    }
    for field_name in string_fields:
        value = normalized[field_name]
        if not isinstance(value, str) or not value.strip():
            raise PreparedPlanError(f"Prepared plan {field_name} must be a non-empty string.")
    if normalized["profile"] not in PROFILE_CACHE_NAMESPACES:
        raise PreparedPlanError("Prepared plan profile is unsupported.")
    for field_name in ("cache_keys",):
        value = normalized[field_name]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and bool(item) for item in value
        ):
            raise PreparedPlanError(f"Prepared plan {field_name} must be a string list.")
    for field_name in ("sample_ledger", "window_ledger", "validation_findings", "repairs"):
        value = normalized[field_name]
        if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
            raise PreparedPlanError(f"Prepared plan {field_name} must be an object list.")
    for field_name in (
        "effective_counts",
        "disposition_counts",
        "windowing_policy",
        "noise_statistics",
    ):
        if not isinstance(normalized[field_name], dict):
            raise PreparedPlanError(f"Prepared plan {field_name} must be an object.")
    counts = cast(dict[str, object], normalized["effective_counts"])
    if not all(
        isinstance(group, dict)
        and all(
            isinstance(key, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 0
            for key, count in group.items()
        )
        for group in counts.values()
    ):
        raise PreparedPlanError(
            "Prepared plan effective_counts must contain non-negative integers."
        )
    dispositions = cast(dict[str, object], normalized["disposition_counts"])
    if not all(
        isinstance(key, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for key, count in dispositions.items()
    ):
        raise PreparedPlanError("Prepared plan disposition_counts is invalid.")
    shape = normalized["feature_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape
        )
    ):
        raise PreparedPlanError("Prepared plan feature_shape must contain two positive integers.")
    if not Path(cast(str, normalized["payload_path"])).is_absolute():
        raise PreparedPlanError("Prepared plan payload_path must be absolute.")
    return normalized


def load_prepared_plan(path: Path) -> PreparedPlan:
    """Loads one prepared plan using explicit boundary validation."""
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise PreparedPlanError(f"Unable to read prepared plan {path}: {error}") from error
    if not isinstance(raw, dict):
        raise PreparedPlanError("Prepared plan root must be a JSON object.")
    validated_raw = _validate_prepared_plan_mapping(raw)
    try:
        raw_feature_shape = cast(list[object], validated_raw["feature_shape"])
        if not isinstance(raw_feature_shape, list) or len(raw_feature_shape) != 2:
            raise ValueError("feature_shape must contain exactly two integers")
        plan = PreparedPlan(
            schema_version=cast(int, validated_raw["schema_version"]),
            created_at=cast(str, validated_raw["created_at"]),
            code_version=str(raw["code_version"]),
            package_version=str(raw["package_version"]),
            profile=cast(ProfileName, raw["profile"]),
            backend_id=str(raw["backend_id"]),
            model_id=str(raw["model_id"]),
            model_revision=str(raw["model_revision"]),
            device=str(raw["device"]),
            dtype=str(raw["dtype"]),
            settings_digest=str(raw["settings_digest"]),
            registry_digest=str(raw["registry_digest"]),
            manifest_digest=str(raw["manifest_digest"]),
            media_digest=str(raw["media_digest"]),
            recipe_digest=str(raw["recipe_digest"]),
            split_ledger_digest=str(raw["split_ledger_digest"]),
            quarantine_ledger_digest=str(raw["quarantine_ledger_digest"]),
            cache_namespace=str(raw["cache_namespace"]),
            cache_version=str(raw["cache_version"]),
            cache_keys=tuple(str(item) for item in cast(list[object], raw["cache_keys"])),
            effective_counts=cast(Mapping[str, Mapping[str, int]], raw["effective_counts"]),
            sample_ledger=tuple(cast(list[Mapping[str, object]], raw["sample_ledger"])),
            window_ledger=tuple(cast(list[Mapping[str, object]], raw["window_ledger"])),
            disposition_counts=cast(Mapping[str, int], raw["disposition_counts"]),
            feature_shape=(cast(int, raw_feature_shape[0]), cast(int, raw_feature_shape[1])),
            feature_dtype=str(raw["feature_dtype"]),
            windowing_policy=cast(Mapping[str, object], raw["windowing_policy"]),
            noise_statistics=cast(Mapping[str, object], raw["noise_statistics"]),
            validation_findings=tuple(cast(list[Mapping[str, object]], raw["validation_findings"])),
            repairs=tuple(cast(list[Mapping[str, object]], raw["repairs"])),
            payload_path=str(raw["payload_path"]),
            payload_digest=str(raw["payload_digest"]),
            overall_digest=str(raw["overall_digest"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PreparedPlanError(f"Prepared plan schema is invalid: {error}") from error
    if plan.schema_version != PREPARED_PLAN_SCHEMA_VERSION:
        raise PreparedPlanError(f"Unsupported prepared plan schema {plan.schema_version}.")
    if plan.code_version != PREPARATION_CODE_VERSION:
        raise PreparedPlanError("Prepared plan code/schema version is stale.")
    if plan.overall_digest != digest_payload(plan.unsigned_dict()):
        raise PreparedPlanError("Prepared plan overall digest mismatch.")
    payload_path = Path(plan.payload_path)
    if not payload_path.is_file() or hash_file(payload_path) != plan.payload_digest:
        raise PreparedPlanError("Prepared feature payload is missing or digest-invalid.")
    return plan


def validate_prepared_plan(
    plan: PreparedPlan,
    *,
    settings: AppConfig,
    readiness: ReadinessReport,
    backend_id: str,
    model_id: str,
    model_revision: str,
    device: str,
    dtype: str,
    recipe_digest: str,
    split_ledger_digest: str,
    quarantine_ledger_digest: str,
    cache_namespace: str,
    cache_version: str,
    cache_keys: Sequence[str],
    package_version: str,
) -> None:
    """Rejects any prepared plan whose checked training inputs have changed."""
    expected: dict[str, object] = {
        "profile": resolve_profile_name(settings),
        "backend_id": backend_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "device": device,
        "dtype": dtype,
        "settings_digest": readiness.settings_digest,
        "registry_digest": readiness.registry_digest,
        "manifest_digest": readiness.manifest_digest,
        "media_digest": readiness.media_digest,
        "recipe_digest": recipe_digest,
        "split_ledger_digest": split_ledger_digest,
        "quarantine_ledger_digest": quarantine_ledger_digest,
        "cache_namespace": cache_namespace,
        "cache_version": cache_version,
        "cache_keys": tuple(sorted(cache_keys)),
        "package_version": package_version,
    }
    mismatches = [name for name, value in expected.items() if getattr(plan, name) != value]
    if mismatches:
        raise PreparedPlanError(
            "Prepared plan is stale for: " + ", ".join(sorted(mismatches)) + "."
        )


def effective_counts(utterances: Sequence[Utterance]) -> dict[str, dict[str, int]]:
    """Returns deterministic effective counts by required reporting dimensions."""
    dimensions: dict[str, Counter[str]] = {
        "corpus": Counter(item.corpus for item in utterances),
        "class": Counter(item.label or "unknown" for item in utterances),
        "language": Counter(item.language or "unknown" for item in utterances),
        "split": Counter(str(item.split or "unspecified") for item in utterances),
        "disposition": Counter({"included": len(utterances)}),
    }
    return {name: dict(sorted(counter.items())) for name, counter in dimensions.items()}


__all__ = [
    "DEFAULT_SMOKE_SAMPLE_CAP",
    "FailureClassification",
    "FailureDisposition",
    "FailureReasonCode",
    "FailureScope",
    "FailureSeverity",
    "PreparedPlan",
    "PreparedPlanError",
    "QuarantineBudgetExceeded",
    "QuarantinePolicy",
    "QuarantineRecord",
    "ReadinessFinding",
    "ReadinessReport",
    "SmokeResult",
    "TrainingMode",
    "TrainingOperation",
    "TrainingReadinessError",
    "atomic_write_json",
    "build_prepared_plan",
    "build_quarantine_record",
    "canonical_json_bytes",
    "classify_failure",
    "default_prepared_payload_path",
    "default_prepared_plan_path",
    "default_readiness_report_path",
    "digest_payload",
    "effective_counts",
    "enforce_quarantine_budget",
    "ensure_training_ready",
    "hash_file",
    "load_prepared_plan",
    "run_training_readiness",
    "select_smoke_samples",
    "settings_digest",
    "validate_prepared_plan",
    "write_prepared_plan",
    "write_quarantine_ledger",
]
