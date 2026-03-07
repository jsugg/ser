"""Tests for settings-aware training-report helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ser.models import training_reporting


class _Artifacts:
    """Minimal persisted-artifact stub for report assembly tests."""

    pickle_path = Path("models/ser_model.pkl")
    secure_path = None


@dataclass(frozen=True)
class _DatasetConfig:
    glob_pattern: str


@dataclass(frozen=True)
class _Settings:
    dataset: _DatasetConfig


def test_build_training_report_for_settings_uses_dataset_glob_pattern() -> None:
    """Settings-aware report helper should read the dataset glob from current settings."""
    report = training_reporting.build_training_report_for_settings(
        read_settings=lambda: _Settings(
            dataset=_DatasetConfig(glob_pattern="dataset/*.wav")
        ),
        artifact_version=2,
        artifact_schema_version="v2",
        accuracy=1.0,
        macro_f1=1.0,
        ser_metrics={"uar": 1.0},
        train_samples=3,
        test_samples=1,
        feature_vector_size=4,
        labels=["happy", "sad", "happy"],
        artifacts=_Artifacts(),
        artifact_metadata={"profile": "fast"},
        globber=lambda pattern: [pattern, pattern],
    )

    assert report["dataset_glob_pattern"] == "dataset/*.wav"
    assert report["dataset_corpus_samples"] == 2
    assert report["dataset_effective_samples"] == 4
