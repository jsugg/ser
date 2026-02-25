"""Tests for provenance metadata emitted in artifacts and training reports."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from sklearn.neural_network import MLPClassifier

from ser import config
from ser.license_check import build_provenance_metadata, persist_backend_consent
from ser.models import emotion_model as em


def _classifier() -> MLPClassifier:
    """Builds deterministic classifier for artifact metadata tests."""
    return MLPClassifier(
        hidden_layer_sizes=(2,),
        max_iter=5,
        solver="lbfgs",
        random_state=0,
    )


def test_build_provenance_metadata_contains_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance helper should emit stable required fields for audits."""
    monkeypatch.setenv("SER_CODE_REVISION", "deadbeef")
    settings = config.reload_settings()

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id="hf_whisper",
        profile="accurate",
    )

    assert provenance["code_revision"] == "deadbeef"
    assert isinstance(provenance["dependency_manifest_fingerprint"], str)
    assert provenance["backend_id"] == "hf_whisper"
    assert provenance["backend_license_id"] == "MIT"
    assert provenance["profile"] == "accurate"
    assert provenance["dataset_glob_pattern"] == settings.dataset.glob_pattern
    assert provenance["runtime_restricted_backends_enabled"] is False
    assert provenance["backend_is_restricted"] is False
    assert provenance["backend_access_allowed"] is True
    assert provenance["backend_access_source"] == "unrestricted"
    assert isinstance(provenance["restricted_backend_policy_fingerprint"], str)
    assert isinstance(provenance["license_source_url"], str)


def test_build_provenance_metadata_records_persisted_consent_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Restricted backend provenance should include persisted consent details."""
    monkeypatch.setenv(
        "SER_RESTRICTED_BACKENDS_CONSENT_FILE",
        str(tmp_path / "restricted-backend-consent.json"),
    )
    settings = config.reload_settings()
    persist_backend_consent(
        settings=settings,
        backend_id="emotion2vec",
        consent_source="unit_test",
    )

    provenance = build_provenance_metadata(
        settings=settings,
        backend_id="emotion2vec",
        profile="accurate-research",
    )

    assert provenance["backend_is_restricted"] is True
    assert provenance["backend_access_allowed"] is True
    assert provenance["backend_access_source"] == "persisted_consent"
    assert provenance["restricted_backend_consent_source"] == "unit_test"
    assert isinstance(provenance["restricted_backend_consent_accepted_at_utc"], str)


def test_build_model_artifact_persists_provenance_payload() -> None:
    """Artifact metadata envelope should preserve validated provenance payload."""
    provenance = {
        "code_revision": "cafebabe",
        "dependency_manifest_fingerprint": "f00d",
        "backend_id": "hf_xlsr",
        "backend_license_id": "Apache-2.0",
        "profile": "medium",
        "dataset_glob_pattern": "ser/dataset/ravdess/Actor_*/*.wav",
        "runtime_restricted_backends_enabled": False,
        "license_source_url": "https://huggingface.co/facebook/wav2vec2-xls-r-300m",
    }
    artifact = em._build_model_artifact(
        model=_classifier(),
        feature_vector_size=4,
        training_samples=12,
        labels=["happy", "sad"],
        backend_id="hf_xlsr",
        profile="medium",
        feature_dim=4,
        frame_size_seconds=1.0,
        frame_stride_seconds=1.0,
        pooling_strategy="mean_std",
        provenance=provenance,
    )

    loaded = em._deserialize_model_artifact(artifact)
    assert loaded.artifact_metadata is not None
    assert loaded.artifact_metadata["provenance"] == provenance


def test_build_training_report_includes_provenance_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training report should include provenance block when provided by caller."""
    monkeypatch.setattr(
        em,
        "get_settings",
        lambda: SimpleNamespace(dataset=SimpleNamespace(glob_pattern="unused")),
    )
    monkeypatch.setattr(em.glob, "glob", lambda _pattern: ["file_0.wav"])

    provenance = {
        "code_revision": "cafebabe",
        "dependency_manifest_fingerprint": "f00d",
        "backend_id": "handcrafted",
        "backend_license_id": "ISC",
        "profile": "fast",
        "dataset_glob_pattern": "unused",
        "runtime_restricted_backends_enabled": False,
        "license_source_url": "https://github.com/librosa/librosa/blob/main/LICENSE.md",
    }
    report = em._build_training_report(
        accuracy=1.0,
        macro_f1=1.0,
        ser_metrics={
            "labels": ["happy", "sad"],
            "uar": 1.0,
            "macro_f1": 1.0,
            "per_class_recall": {"happy": 1.0, "sad": 1.0},
            "confusion_matrix": [[1, 0], [0, 1]],
        },
        train_samples=2,
        test_samples=2,
        feature_vector_size=4,
        labels=["happy", "sad", "happy", "sad"],
        artifacts=em.PersistedArtifacts(
            pickle_path=Path("ser_model.pkl"),
            secure_path=None,
        ),
        artifact_metadata={
            "artifact_version": em.MODEL_ARTIFACT_VERSION,
            "artifact_schema_version": "v2",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_vector_size": 4,
            "training_samples": 2,
            "labels": ["happy", "sad"],
            "backend_id": "handcrafted",
            "profile": "fast",
            "feature_dim": 4,
            "frame_size_seconds": 3.0,
            "frame_stride_seconds": 1.0,
            "pooling_strategy": "mean",
            "provenance": provenance,
        },
        provenance=provenance,
    )

    assert report["provenance"] == provenance
