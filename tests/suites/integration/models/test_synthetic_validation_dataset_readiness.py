"""Hosted validation lanes must build a synthetic dataset that passes training readiness."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from ser._internal.config.bootstrap import reload_settings  # noqa: TID251
from ser._internal.data.adapters.ravdess import build_ravdess_utterances  # noqa: TID251
from ser._internal.data.label_ontology import resolve_label_ontology  # noqa: TID251
from ser._internal.data.manifest import Utterance  # noqa: TID251
from ser._internal.models.training_readiness import run_training_readiness  # noqa: TID251
from ser.config import AppConfig

_BUILDER = "scripts/build_synthetic_ravdess_dataset.py"
_INVOCATION = re.compile(rf"python \./{re.escape(_BUILDER)}(?P<args>[^\n]*)")
# Lanes train with the default split policy; neutralize developer overrides of it.
_SPLIT_POLICY_ENV = (
    "SER_TEST_SIZE",
    "SER_DEV_SIZE",
    "SER_MIN_REMAINING_PER_CLASS_SPLIT",
    "SER_DATASET_MANIFESTS",
    "SER_DATASET_RECIPE",
)


def _builder_invocations(repo_root: Path) -> list[tuple[str, tuple[str, ...]]]:
    invocations: list[tuple[str, tuple[str, ...]]] = []
    for workflow in sorted((repo_root / ".github" / "workflows").glob("*.yml")):
        for match in _INVOCATION.finditer(workflow.read_text(encoding="utf-8")):
            invocations.append((workflow.name, tuple(match.group("args").split())))
    return invocations


def _lane_settings(work_root: Path) -> AppConfig:
    settings = reload_settings()
    return replace(
        settings,
        tmp_folder=work_root / "tmp",
        models=replace(
            settings.models,
            folder=work_root / "models",
            model_cache_dir=work_root / "cache",
        ),
    )


def _load_lane_utterances(settings: AppConfig) -> list[Utterance] | None:
    return build_ravdess_utterances(
        dataset_root=settings.dataset.folder,
        dataset_glob_pattern=settings.dataset.glob_pattern,
        emotion_code_map=dict(settings.emotions),
        default_language=settings.default_language,
        ontology=resolve_label_ontology(settings),
        max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
    )


def test_every_lane_passes_an_explicit_actor_list(repo_root: Path) -> None:
    """The builder's default actor list is not what the lanes are validated against."""
    invocations = _builder_invocations(repo_root)

    assert invocations, "No workflow builds the synthetic dataset."
    assert [name for name, args in invocations if "--actors" not in args] == []


def test_lane_synthetic_datasets_pass_training_readiness(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each distinct lane dataset must satisfy the speaker-disjoint three-way split."""
    for variable in _SPLIT_POLICY_ENV:
        monkeypatch.delenv(variable, raising=False)

    for index, args in enumerate(sorted({args for _, args in _builder_invocations(repo_root)})):
        dataset_root = tmp_path / f"dataset-{index}"
        subprocess.run(
            [sys.executable, str(repo_root / _BUILDER), "--output-root", str(dataset_root), *args],
            check=True,
            capture_output=True,
        )
        monkeypatch.setenv("DATASET_FOLDER", str(dataset_root))
        settings = _lane_settings(tmp_path / f"work-{index}")

        report, _ = run_training_readiness(
            settings=settings,
            load_utterances=partial(_load_lane_utterances, settings),
            persist_quarantine_ledger=False,
            report_path=tmp_path / f"readiness-{index}.json",
        )

        assert report.ready, (args, [finding.message for finding in report.findings])
