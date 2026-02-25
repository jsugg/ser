"""Tests for backend license policy gates."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

import ser.config as config
import ser.runtime.backend_hooks as backend_hooks
from ser.license_check import (
    BackendLicensePolicyError,
    ensure_backend_access,
    evaluate_backend_access,
    load_persisted_backend_consents,
    parse_allowed_restricted_backends_env,
    persist_backend_consent,
)
from ser.runtime.contracts import InferenceRequest
from ser.runtime.schema import OUTPUT_SCHEMA_VERSION, InferenceResult


@pytest.fixture(autouse=True)
def _reset_settings() -> Generator[None, None, None]:
    """Keeps runtime settings deterministic across tests."""
    config.reload_settings()
    yield
    config.reload_settings()


def test_unrestricted_backend_allowed_without_restricted_opt_in() -> None:
    """Permissive backends should remain executable by default."""
    decision = evaluate_backend_access(
        backend_id="hf_whisper",
        restricted_backends_enabled=False,
    )
    assert decision.allowed is True
    assert decision.policy.license_id == "MIT"


def test_restricted_backend_requires_explicit_opt_in() -> None:
    """Restricted backends must fail closed unless opt-in flag is enabled."""
    with pytest.raises(
        BackendLicensePolicyError,
        match="SER_ENABLE_RESTRICTED_BACKENDS=true",
    ):
        ensure_backend_access(
            backend_id="emotion2vec",
            restricted_backends_enabled=False,
        )


def test_restricted_backend_allows_execution_when_opted_in() -> None:
    """Restricted backends should be available only when explicit opt-in is set."""
    decision = evaluate_backend_access(
        backend_id="emotion2vec",
        restricted_backends_enabled=True,
    )
    assert decision.allowed is True
    assert decision.policy.restricted is True


def test_restricted_backend_allows_execution_when_allowlisted_by_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Volatile per-run allowlist should permit one restricted backend."""
    monkeypatch.setenv("SER_ALLOWED_RESTRICTED_BACKENDS", "emotion2vec")
    decision = evaluate_backend_access(
        backend_id="emotion2vec",
        restricted_backends_enabled=False,
        allowed_restricted_backends=parse_allowed_restricted_backends_env(),
    )
    assert decision.allowed is True
    assert decision.access_source == "env_allowlist"


def test_restricted_backend_allows_execution_when_persisted_consent_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persisted consent should permit restricted backend without env/global override."""
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
    persisted_consents = load_persisted_backend_consents(settings=settings)
    decision = evaluate_backend_access(
        backend_id="emotion2vec",
        restricted_backends_enabled=False,
        persisted_consents=persisted_consents,
    )
    assert decision.allowed is True
    assert decision.access_source == "persisted_consent"
    assert decision.consent_record is not None
    assert decision.consent_record.consent_source == "unit_test"


def test_unknown_backend_is_denied_by_default() -> None:
    """Unknown backend ids should be blocked until policy metadata is defined."""
    decision = evaluate_backend_access(
        backend_id="unknown_backend",
        restricted_backends_enabled=True,
    )
    assert decision.allowed is False
    assert "undefined" in decision.reason


def test_backend_hooks_raise_when_license_gate_denies_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime hook registration should surface policy violations explicitly."""
    monkeypatch.setenv("SER_ENABLE_ACCURATE_PROFILE", "true")
    monkeypatch.setattr(backend_hooks, "_missing_optional_modules", lambda _mods: ())

    def fake_runner(
        request: InferenceRequest,
        settings: config.AppConfig,
    ) -> InferenceResult:
        del request, settings
        return InferenceResult(
            schema_version=OUTPUT_SCHEMA_VERSION,
            segments=[],
            frames=[],
        )

    monkeypatch.setattr(
        backend_hooks,
        "_load_accurate_inference_runner",
        lambda: fake_runner,
    )
    monkeypatch.setattr(
        backend_hooks,
        "ensure_backend_access",
        lambda **_kwargs: (_ for _ in ()).throw(BackendLicensePolicyError("blocked")),
    )

    settings = config.reload_settings()
    with pytest.raises(BackendLicensePolicyError, match="blocked"):
        backend_hooks.build_backend_hooks(settings)


def test_accurate_research_hook_requires_restricted_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accurate-research hook registration should fail without explicit opt-in."""
    monkeypatch.setenv(
        "SER_RESTRICTED_BACKENDS_CONSENT_FILE",
        str(tmp_path / "restricted-backend-consent.json"),
    )
    monkeypatch.delenv("SER_ALLOWED_RESTRICTED_BACKENDS", raising=False)
    monkeypatch.delenv("SER_ENABLE_RESTRICTED_BACKENDS", raising=False)
    monkeypatch.setenv("SER_ENABLE_ACCURATE_RESEARCH_PROFILE", "true")
    monkeypatch.setattr(backend_hooks, "_missing_optional_modules", lambda _mods: ())
    monkeypatch.setattr(
        backend_hooks,
        "_load_accurate_research_inference_runner",
        lambda: (
            lambda _request, _settings: InferenceResult(
                schema_version=OUTPUT_SCHEMA_VERSION,
                segments=[],
                frames=[],
            )
        ),
    )

    settings = config.reload_settings()
    with pytest.raises(
        BackendLicensePolicyError, match="SER_ENABLE_RESTRICTED_BACKENDS=true"
    ):
        backend_hooks.build_backend_hooks(settings)
