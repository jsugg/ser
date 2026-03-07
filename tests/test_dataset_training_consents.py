"""Contract tests for training-time dataset consent orchestration helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import ser.models.dataset_training_consents as training_consents
from ser.config import AppConfig
from ser.data import Utterance
from ser.data.dataset_consents import DatasetConsentError
from ser.data.manifest import MANIFEST_SCHEMA_VERSION


def _settings_stub(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                consents_file=tmp_path / "dataset_consents.json",
            )
        ),
    )


def _utterance_stub(tmp_path: Path) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id="sample-1",
        corpus="stub-corpus",
        audio_path=tmp_path / "clip.wav",
        label="neutral",
    )


def test_ensure_dataset_training_consents_noninteractive_reraises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Non-interactive training should fail closed on missing consents."""

    def _raise_missing(*, settings: AppConfig, utterances: list[Utterance]) -> None:
        del settings, utterances
        raise DatasetConsentError("missing dataset consent")

    monkeypatch.setattr(training_consents, "ensure_dataset_consents", _raise_missing)

    with pytest.raises(DatasetConsentError, match="missing dataset consent"):
        training_consents.ensure_dataset_training_consents(
            utterances=[_utterance_stub(tmp_path)],
            settings=_settings_stub(tmp_path),
            logger_warning=lambda *_args: None,
            stdin_isatty=lambda _fd: False,
        )


def test_ensure_dataset_training_consents_interactive_accept_persists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Interactive training should persist missing consent IDs after acceptance."""
    persisted: dict[str, object] = {}

    def _raise_missing(*, settings: AppConfig, utterances: list[Utterance]) -> None:
        del settings, utterances
        raise DatasetConsentError("consent required")

    def _compute_missing(
        *,
        settings: AppConfig,
        utterances: list[Utterance],
    ) -> tuple[set[str], set[str]]:
        del settings, utterances
        return {"academic_only"}, {"msp-license"}

    def _persist(
        *,
        settings: AppConfig,
        accept_policy_ids: list[str],
        accept_license_ids: list[str],
        source: str,
    ) -> None:
        persisted["settings"] = settings
        persisted["accept_policy_ids"] = accept_policy_ids
        persisted["accept_license_ids"] = accept_license_ids
        persisted["source"] = source

    monkeypatch.setattr(training_consents, "ensure_dataset_consents", _raise_missing)
    monkeypatch.setattr(
        training_consents,
        "compute_missing_dataset_consents",
        _compute_missing,
    )
    monkeypatch.setattr(training_consents, "persist_dataset_consents", _persist)

    training_consents.ensure_dataset_training_consents(
        utterances=[_utterance_stub(tmp_path)],
        settings=_settings_stub(tmp_path),
        logger_warning=lambda *_args: None,
        stdin_isatty=lambda _fd: True,
        prompt_input=lambda: "accept",
        prompt_print=lambda *_args, **_kwargs: None,
    )

    assert persisted["accept_policy_ids"] == ["academic_only"]
    assert persisted["accept_license_ids"] == ["msp-license"]
    assert persisted["source"] == "training"
