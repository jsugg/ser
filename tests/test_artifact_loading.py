"""Contracts for model artifact loading helper orchestration."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from ser.models.artifact_loading import (
    load_model_with_resolution,
    load_pickle_model_artifact,
    load_secure_model_artifact,
    resolve_model_for_loading,
    resolve_model_for_loading_from_settings,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    path: Path
    artifact_format: Literal["pickle", "skops"]


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    artifact_metadata: dict[str, object] | None


def test_load_model_with_resolution_preserves_file_not_found() -> None:
    """File-not-found from resolver should pass through without remapping."""

    def _raise_file_not_found(
        _settings: object,
        *,
        expected_backend_id: str | None = None,
        expected_profile: str | None = None,
        expected_backend_model_id: str | None = None,
    ) -> tuple[_Candidate, _LoadedModel]:
        del expected_backend_id, expected_profile, expected_backend_model_id
        raise FileNotFoundError("Train it first.")

    with pytest.raises(FileNotFoundError, match="Train it first"):
        _ = load_model_with_resolution(
            settings=None,
            settings_resolver=lambda: object(),
            resolve_model=_raise_file_not_found,
            logger=logging.getLogger("tests.artifact_loading.file_not_found"),
        )


def test_load_model_with_resolution_wraps_unexpected_errors() -> None:
    """Unexpected resolver failures should map to stable ValueError contract."""

    def _raise_runtime_error(
        _settings: object,
        *,
        expected_backend_id: str | None = None,
        expected_profile: str | None = None,
        expected_backend_model_id: str | None = None,
    ) -> tuple[_Candidate, _LoadedModel]:
        del expected_backend_id, expected_profile, expected_backend_model_id
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="configured locations"):
        _ = load_model_with_resolution(
            settings=None,
            settings_resolver=lambda: object(),
            resolve_model=_raise_runtime_error,
            logger=logging.getLogger("tests.artifact_loading.runtime_error"),
        )


def test_load_pickle_model_artifact_deserializes_payload(tmp_path: Path) -> None:
    """Pickle loader helper should deserialize payload with provided callback."""
    candidate_path = tmp_path / "ser_model.pkl"
    with candidate_path.open("wb") as handle:
        pickle.dump({"value": 7}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_payload(payload: object) -> int:
        if not isinstance(payload, dict):
            raise AssertionError("Expected dictionary payload.")
        value = payload.get("value")
        if not isinstance(value, int):
            raise AssertionError("Expected integer value.")
        return value

    loaded = load_pickle_model_artifact(
        candidate_path=candidate_path,
        deserialize_payload=_deserialize_payload,
    )

    assert loaded == 7


def test_load_secure_model_artifact_rejects_untrusted_types(tmp_path: Path) -> None:
    """Secure loader helper should fail closed when untrusted types are present."""

    class _FakeSkopsModule:
        @staticmethod
        def get_untrusted_types(*, file: str) -> object:
            del file
            return ["dangerous"]

        @staticmethod
        def load(_file: str, *, trusted: list[object]) -> object:
            del trusted
            return object()

    with pytest.raises(ValueError, match="contains untrusted types"):
        _ = load_secure_model_artifact(
            candidate_path=tmp_path / "ser_model.skops",
            model_instance_check=lambda _payload: True,
            training_report_file=tmp_path / "training_report.json",
            read_training_report_feature_size=lambda _path: 3,
            loaded_model_factory=lambda payload, expected_feature_size: (
                payload,
                expected_feature_size,
            ),
            import_module_fn=lambda _name: _FakeSkopsModule(),
        )


def test_load_secure_model_artifact_returns_loaded_model(tmp_path: Path) -> None:
    """Secure loader helper should return factory output for trusted payloads."""
    sentinel_payload = object()
    captured_report_path: Path | None = None

    class _FakeSkopsModule:
        @staticmethod
        def get_untrusted_types(*, file: str) -> object:
            del file
            return []

        @staticmethod
        def load(_file: str, *, trusted: list[object]) -> object:
            del trusted
            return sentinel_payload

    def _read_feature_size(path: Path) -> int | None:
        nonlocal captured_report_path
        captured_report_path = path
        return 11

    loaded = load_secure_model_artifact(
        candidate_path=tmp_path / "ser_model.skops",
        model_instance_check=lambda payload: payload is sentinel_payload,
        training_report_file=tmp_path / "training_report.json",
        read_training_report_feature_size=_read_feature_size,
        loaded_model_factory=lambda payload, expected_feature_size: (
            payload,
            expected_feature_size,
        ),
        import_module_fn=lambda _name: _FakeSkopsModule(),
    )

    assert loaded == (sentinel_payload, 11)
    assert captured_report_path == tmp_path / "training_report.json"


def test_resolve_model_for_loading_logs_candidate_failures_at_debug(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fallback candidate load failures should be debug noise when a later candidate succeeds."""
    primary = _Candidate(path=tmp_path / "ser_model.pkl", artifact_format="pickle")
    fallback = _Candidate(
        path=tmp_path / "ser_model_fast_full.pkl",
        artifact_format="pickle",
    )
    primary.path.write_bytes(b"primary")
    fallback.path.write_bytes(b"fallback")
    expected = _LoadedModel(artifact_metadata={"profile": "fast"})

    def _load_pickle(candidate: _Candidate) -> _LoadedModel:
        if candidate is primary:
            raise ValueError(
                "Unexpected model object type in artifact envelope: NoneType."
            )
        return expected

    logger = logging.getLogger("tests.artifact_loading.debug_fallback")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    resolved_candidate, resolved_model = resolve_model_for_loading(
        candidates=(primary, fallback),
        load_secure_model=lambda candidate: (_ for _ in ()).throw(
            AssertionError(f"unexpected secure load: {candidate}")
        ),
        load_pickle_model=_load_pickle,
        logger=logger,
    )

    assert resolved_candidate is fallback
    assert resolved_model is expected
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG
    ]
    assert any(
        "Failed to load pickle model artifact" in message for message in debug_messages
    )
    assert all(record.levelno < logging.WARNING for record in caplog.records)


def test_resolve_model_for_loading_from_settings_uses_settings_paths(
    tmp_path: Path,
) -> None:
    """Settings-aware resolution should derive candidates and secure loads from settings."""
    secure_model = tmp_path / "ser_model.skops"
    pickle_model = tmp_path / "ser_model.pkl"
    secure_model.write_bytes(b"secure")
    pickle_model.write_bytes(b"pickle")
    settings = object()
    expected = _LoadedModel(artifact_metadata={"profile": "fast"})
    captured: dict[str, object] = {}

    def _load_secure(
        candidate: _Candidate,
        active_settings: object,
    ) -> _LoadedModel:
        captured["candidate"] = candidate
        captured["settings"] = active_settings
        return expected

    resolved_candidate, resolved_model = resolve_model_for_loading_from_settings(
        settings,
        folder=tmp_path,
        secure_model_file=secure_model,
        model_file=pickle_model,
        candidate_factory=_Candidate,
        load_secure_model_for_settings=_load_secure,
        load_pickle_model=lambda candidate: (_ for _ in ()).throw(
            AssertionError(f"unexpected pickle load: {candidate}")
        ),
        logger=logging.getLogger("tests.artifact_loading.settings_resolution"),
    )

    assert resolved_candidate.path == secure_model
    assert resolved_model is expected
    assert captured["candidate"] == resolved_candidate
    assert captured["settings"] is settings
