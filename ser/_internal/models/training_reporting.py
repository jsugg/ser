"""Training-report and report-control payload helpers."""

from __future__ import annotations

import glob
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class PersistedArtifactsLike(Protocol):
    """Structural contract for persisted model artifact paths."""

    @property
    def pickle_path(self) -> Path: ...

    @property
    def secure_path(self) -> Path | None: ...


class MediumNoiseControlStatsLike(Protocol):
    """Structural contract for medium noise-control counters."""

    @property
    def total_windows(self) -> int: ...

    @property
    def kept_windows(self) -> int: ...

    @property
    def dropped_low_std_windows(self) -> int: ...

    @property
    def dropped_cap_windows(self) -> int: ...

    @property
    def forced_keep_windows(self) -> int: ...


class SplitMetadataLike(Protocol):
    """Structural contract for grouped split diagnostics."""

    @property
    def split_strategy(self) -> str: ...

    @property
    def speaker_grouped(self) -> bool: ...

    @property
    def speaker_id_coverage(self) -> float: ...

    @property
    def train_unique_speakers(self) -> int: ...

    @property
    def test_unique_speakers(self) -> int: ...

    @property
    def speaker_overlap_count(self) -> int: ...


class DatasetConfigLike(Protocol):
    """Structural contract for settings exposing dataset glob configuration."""

    @property
    def glob_pattern(self) -> str: ...


class SettingsLike(Protocol):
    """Structural contract for settings exposing dataset configuration."""

    @property
    def dataset(self) -> DatasetConfigLike: ...


def build_grouped_evaluation_controls(
    split_metadata: SplitMetadataLike,
) -> dict[str, object]:
    """Builds grouped-evaluation control payload from split metadata."""
    return {
        "split_strategy": split_metadata.split_strategy,
        "speaker_grouped": split_metadata.speaker_grouped,
        "speaker_id_coverage": split_metadata.speaker_id_coverage,
        "train_unique_speakers": split_metadata.train_unique_speakers,
        "test_unique_speakers": split_metadata.test_unique_speakers,
        "speaker_overlap_count": split_metadata.speaker_overlap_count,
    }


def _medium_noise_split_payload(
    stats: MediumNoiseControlStatsLike,
) -> dict[str, int]:
    """Builds train/test branch payload for medium noise controls."""
    return {
        "total_windows": stats.total_windows,
        "kept_windows": stats.kept_windows,
        "dropped_low_std_windows": stats.dropped_low_std_windows,
        "dropped_cap_windows": stats.dropped_cap_windows,
        "forced_keep_windows": stats.forced_keep_windows,
    }


def build_medium_noise_controls(
    *,
    min_window_std: float,
    max_windows_per_clip: int,
    train_stats: MediumNoiseControlStatsLike,
    test_stats: MediumNoiseControlStatsLike,
) -> dict[str, object]:
    """Builds medium noise-control report payload from train/test counters."""
    return {
        "min_window_std": min_window_std,
        "max_windows_per_clip": max_windows_per_clip,
        "train": _medium_noise_split_payload(train_stats),
        "test": _medium_noise_split_payload(test_stats),
    }


def build_training_report(
    *,
    dataset_glob_pattern: str,
    artifact_version: int,
    artifact_schema_version: str,
    accuracy: float,
    macro_f1: float,
    ser_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    feature_vector_size: int,
    labels: list[str],
    artifacts: PersistedArtifactsLike,
    artifact_metadata: dict[str, object],
    data_controls: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    globber: Callable[[str], list[str]] | None = None,
) -> dict[str, object]:
    """Builds training report with metrics and artifact traceability payloads."""
    resolved_globber = glob.glob if globber is None else globber
    corpus_samples = len(resolved_globber(dataset_glob_pattern))
    effective_samples = train_samples + test_samples
    label_distribution = dict(Counter(labels))
    model_artifacts: dict[str, str] = {"pickle": str(artifacts.pickle_path)}
    if artifacts.secure_path is not None:
        model_artifacts["secure"] = str(artifacts.secure_path)

    report: dict[str, object] = {
        "artifact_version": artifact_version,
        "artifact_schema_version": artifact_schema_version,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "dataset_glob_pattern": dataset_glob_pattern,
        "dataset_corpus_samples": corpus_samples,
        "dataset_effective_samples": effective_samples,
        "dataset_skipped_samples": max(0, corpus_samples - effective_samples),
        "train_samples": train_samples,
        "test_samples": test_samples,
        "feature_vector_size": feature_vector_size,
        "labels": sorted(set(labels)),
        "label_distribution": label_distribution,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "metrics": ser_metrics,
        "artifact_metadata": artifact_metadata,
        "model_artifacts": model_artifacts,
    }
    if data_controls is not None:
        report["data_controls"] = data_controls
    if provenance is not None:
        report["provenance"] = provenance
    return report


def build_training_report_for_settings(
    *,
    read_settings: Callable[[], SettingsLike],
    artifact_version: int,
    artifact_schema_version: str,
    accuracy: float,
    macro_f1: float,
    ser_metrics: dict[str, object],
    train_samples: int,
    test_samples: int,
    feature_vector_size: int,
    labels: list[str],
    artifacts: PersistedArtifactsLike,
    artifact_metadata: dict[str, object],
    data_controls: dict[str, object] | None = None,
    provenance: dict[str, object] | None = None,
    globber: Callable[[str], list[str]] | None = None,
) -> dict[str, object]:
    """Builds a training report using the current settings dataset glob."""
    settings = read_settings()
    return build_training_report(
        dataset_glob_pattern=settings.dataset.glob_pattern,
        artifact_version=artifact_version,
        artifact_schema_version=artifact_schema_version,
        accuracy=accuracy,
        macro_f1=macro_f1,
        ser_metrics=ser_metrics,
        train_samples=train_samples,
        test_samples=test_samples,
        feature_vector_size=feature_vector_size,
        labels=labels,
        artifacts=artifacts,
        artifact_metadata=artifact_metadata,
        data_controls=data_controls,
        provenance=provenance,
        globber=globber,
    )
