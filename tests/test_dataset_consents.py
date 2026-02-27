"""Tests for dataset policy/license consent persistence and gating."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from ser.config import AppConfig
from ser.data.dataset_consents import (
    DatasetConsentError,
    compute_missing_dataset_consents,
    ensure_dataset_consents,
    is_policy_restricted,
    load_persisted_dataset_consents,
    persist_dataset_consents,
)
from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance


def _settings(tmp_path: Path) -> AppConfig:
    return cast(
        AppConfig,
        SimpleNamespace(models=SimpleNamespace(folder=tmp_path / "data" / "models")),
    )


def _utterance(
    *,
    sample_id: str,
    label: str,
    policy_id: str | None,
    license_id: str | None,
) -> Utterance:
    return Utterance(
        schema_version=MANIFEST_SCHEMA_VERSION,
        sample_id=sample_id,
        corpus="ravdess",
        audio_path=Path(f"{sample_id}.wav"),
        label=label,
        speaker_id="ravdess:1",
        dataset_policy_id=policy_id,
        dataset_license_id=license_id,
    )


def test_is_policy_restricted_behaviour() -> None:
    """Restricted policy resolver should include current policy ids only."""
    assert is_policy_restricted("academic_only") is True
    assert is_policy_restricted("share_alike") is True
    assert is_policy_restricted("noncommercial") is True
    assert is_policy_restricted("unknown") is False
    assert is_policy_restricted(None) is False


def test_persist_and_load_dataset_consents(tmp_path: Path) -> None:
    """Persisted consents should round-trip deterministically."""
    settings = _settings(tmp_path)

    persist_dataset_consents(
        settings=settings,
        accept_policy_ids=["academic_only"],
        accept_license_ids=["msp-academic-license"],
        source="unit_test",
    )
    loaded = load_persisted_dataset_consents(settings=settings)

    assert loaded.policy_consents == {"academic_only": "unit_test"}
    assert loaded.license_consents == {"msp-academic-license": "unit_test"}


def test_compute_missing_dataset_consents_ignores_unspecified_policy(
    tmp_path: Path,
) -> None:
    """Legacy records without policy id should remain non-blocking by default."""
    settings = _settings(tmp_path)
    utterances = [
        _utterance(
            sample_id="legacy-0",
            label="happy",
            policy_id=None,
            license_id=None,
        ),
        _utterance(
            sample_id="modern-0",
            label="sad",
            policy_id="academic_only",
            license_id="msp-academic-license",
        ),
    ]

    missing_policies, missing_licenses = compute_missing_dataset_consents(
        settings=settings,
        utterances=utterances,
    )

    assert missing_policies == {"academic_only"}
    assert missing_licenses == {"msp-academic-license"}


def test_ensure_dataset_consents_raises_for_missing_persisted_consent(
    tmp_path: Path,
) -> None:
    """Training gate should raise actionable error when required consents are missing."""
    settings = _settings(tmp_path)
    utterances = [
        _utterance(
            sample_id="restricted-0",
            label="happy",
            policy_id="noncommercial",
            license_id="cc-by-nc-sa-4.0",
        )
    ]

    with pytest.raises(
        DatasetConsentError, match="ser configure --accept-dataset-policy"
    ):
        ensure_dataset_consents(settings=settings, utterances=utterances)
