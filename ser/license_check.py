"""Backend license policy gates and artifact provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ser.config import AppConfig


class BackendLicensePolicyError(RuntimeError):
    """Raised when runtime policy denies access to a backend."""


@dataclass(frozen=True)
class BackendLicensePolicy:
    """License policy metadata for one backend id."""

    backend_id: str
    license_id: str
    restricted: bool
    source_url: str


@dataclass(frozen=True)
class LicenseDecision:
    """Authorization decision for backend access under current policy flags."""

    allowed: bool
    policy: BackendLicensePolicy
    reason: str
    access_source: str
    consent_record: BackendConsentRecord | None = None


@dataclass(frozen=True)
class BackendConsentRecord:
    """Persisted user acknowledgement for one restricted backend."""

    backend_id: str
    license_id: str
    source_url: str
    policy_fingerprint: str
    consent_source: str
    accepted_at_utc: str


@dataclass(frozen=True)
class BackendAccessContext:
    """Resolved backend-access inputs from runtime settings and environment."""

    restricted_backends_enabled: bool
    allowed_restricted_backends: frozenset[str]
    persisted_consents: Mapping[str, BackendConsentRecord]


_BACKEND_POLICIES: dict[str, BackendLicensePolicy] = {
    "handcrafted": BackendLicensePolicy(
        backend_id="handcrafted",
        license_id="ISC",
        restricted=False,
        source_url="https://github.com/librosa/librosa/blob/main/LICENSE.md",
    ),
    "hf_xlsr": BackendLicensePolicy(
        backend_id="hf_xlsr",
        license_id="Apache-2.0",
        restricted=False,
        source_url="https://huggingface.co/facebook/wav2vec2-xls-r-300m",
    ),
    "hf_whisper": BackendLicensePolicy(
        backend_id="hf_whisper",
        license_id="MIT",
        restricted=False,
        source_url="https://github.com/openai/whisper/blob/main/LICENSE",
    ),
    "emotion2vec": BackendLicensePolicy(
        backend_id="emotion2vec",
        license_id="other",
        restricted=True,
        source_url="https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE",
    ),
}
_ALLOWED_RESTRICTED_BACKENDS_ENV = "SER_ALLOWED_RESTRICTED_BACKENDS"
_CONSENT_STORE_ENV = "SER_RESTRICTED_BACKENDS_CONSENT_FILE"
_CONSENT_STORE_FILE_NAME = "restricted_backend_consents.json"
_CONSENT_SCHEMA_VERSION = 1


def _policy_fingerprint(policy: BackendLicensePolicy) -> str:
    """Builds stable fingerprint for backend policy validation."""
    hasher = hashlib.sha256()
    hasher.update(policy.backend_id.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(policy.license_id.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(policy.source_url.encode("utf-8"))
    return hasher.hexdigest()


def _consent_store_path(settings: AppConfig) -> Path:
    """Resolves on-disk path for persisted restricted-backend consents."""
    explicit_path = os.getenv(_CONSENT_STORE_ENV, "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser()
    return settings.models.folder.parent / _CONSENT_STORE_FILE_NAME


def _read_consent_payload(path: Path) -> dict[str, object]:
    """Reads consent payload from disk, defaulting to an empty valid payload."""
    if not path.is_file():
        return {"schema_version": _CONSENT_SCHEMA_VERSION, "consents": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as err:
        raise RuntimeError(
            f"Restricted backend consent store at {path} is unreadable: {err}"
        ) from err
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Restricted backend consent store at {path} must be a JSON object."
        )
    schema_version = payload.get("schema_version")
    if schema_version != _CONSENT_SCHEMA_VERSION:
        raise RuntimeError(
            "Restricted backend consent store schema mismatch. "
            f"Expected {_CONSENT_SCHEMA_VERSION}, got {schema_version!r}."
        )
    raw_consents = payload.get("consents")
    if raw_consents is None:
        payload["consents"] = {}
        return payload
    if not isinstance(raw_consents, dict):
        raise RuntimeError(
            f"Restricted backend consent store at {path} has invalid 'consents'."
        )
    return payload


def _write_consent_payload(path: Path, payload: dict[str, object]) -> None:
    """Persists consent payload atomically to avoid partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def restricted_backend_policies() -> tuple[BackendLicensePolicy, ...]:
    """Returns declared policy entries for restricted backends."""
    return tuple(policy for policy in _BACKEND_POLICIES.values() if policy.restricted)


def get_backend_policy(backend_id: str) -> BackendLicensePolicy | None:
    """Returns policy metadata for one backend id, when declared."""
    return _BACKEND_POLICIES.get(backend_id)


def parse_allowed_restricted_backends_env() -> frozenset[str]:
    """Parses volatile per-run restricted backend allowlist from environment."""
    raw_value = os.getenv(_ALLOWED_RESTRICTED_BACKENDS_ENV, "").strip()
    if not raw_value:
        return frozenset()
    tokens = {token.strip() for token in raw_value.split(",") if token.strip()}
    restricted_ids = {policy.backend_id for policy in restricted_backend_policies()}
    if "*" in tokens:
        return frozenset(restricted_ids)
    return frozenset(token for token in tokens if token in restricted_ids)


def load_persisted_backend_consents(
    *,
    settings: AppConfig,
) -> dict[str, BackendConsentRecord]:
    """Loads persisted consent records for restricted backends."""
    path = _consent_store_path(settings)
    payload = _read_consent_payload(path)
    raw_consents = payload.get("consents")
    if not isinstance(raw_consents, dict):
        raise RuntimeError(
            f"Restricted backend consent store at {path} has invalid 'consents'."
        )

    resolved: dict[str, BackendConsentRecord] = {}
    for backend_id, raw_record in raw_consents.items():
        if not isinstance(backend_id, str) or not backend_id.strip():
            continue
        policy = _BACKEND_POLICIES.get(backend_id)
        if policy is None or not policy.restricted:
            continue
        if not isinstance(raw_record, dict):
            raise RuntimeError(
                f"Consent record for backend {backend_id!r} in {path} must be an object."
            )
        accepted_at_utc = raw_record.get("accepted_at_utc")
        consent_source = raw_record.get("consent_source")
        policy_fingerprint = raw_record.get("policy_fingerprint")
        license_id = raw_record.get("license_id")
        source_url = raw_record.get("source_url")
        if (
            not isinstance(accepted_at_utc, str)
            or not accepted_at_utc.strip()
            or not isinstance(consent_source, str)
            or not consent_source.strip()
            or not isinstance(policy_fingerprint, str)
            or not policy_fingerprint.strip()
            or not isinstance(license_id, str)
            or not license_id.strip()
            or not isinstance(source_url, str)
            or not source_url.strip()
        ):
            raise RuntimeError(
                f"Consent record for backend {backend_id!r} in {path} is invalid."
            )
        # Ignore stale consents when backend license policy changed.
        if policy_fingerprint != _policy_fingerprint(policy):
            continue
        resolved[backend_id] = BackendConsentRecord(
            backend_id=backend_id,
            license_id=license_id,
            source_url=source_url,
            policy_fingerprint=policy_fingerprint,
            consent_source=consent_source,
            accepted_at_utc=accepted_at_utc,
        )
    return resolved


def persist_backend_consent(
    *,
    settings: AppConfig,
    backend_id: str,
    consent_source: str,
) -> BackendConsentRecord:
    """Persists user opt-in consent for one restricted backend id."""
    policy = _BACKEND_POLICIES.get(backend_id)
    if policy is None:
        raise RuntimeError(
            f"Cannot persist consent for undefined backend {backend_id!r}."
        )
    if not policy.restricted:
        raise RuntimeError(
            f"Backend {backend_id!r} is not restricted and does not require consent."
        )
    resolved_source = consent_source.strip() or "unspecified"
    consent_record = BackendConsentRecord(
        backend_id=backend_id,
        license_id=policy.license_id,
        source_url=policy.source_url,
        policy_fingerprint=_policy_fingerprint(policy),
        consent_source=resolved_source,
        accepted_at_utc=datetime.now(tz=UTC).isoformat(),
    )
    path = _consent_store_path(settings)
    payload = _read_consent_payload(path)
    raw_consents = payload.get("consents")
    if not isinstance(raw_consents, dict):
        raise RuntimeError(
            f"Restricted backend consent store at {path} has invalid 'consents'."
        )
    raw_consents[backend_id] = {
        "accepted_at_utc": consent_record.accepted_at_utc,
        "consent_source": consent_record.consent_source,
        "policy_fingerprint": consent_record.policy_fingerprint,
        "license_id": consent_record.license_id,
        "source_url": consent_record.source_url,
    }
    payload["schema_version"] = _CONSENT_SCHEMA_VERSION
    payload["consents"] = raw_consents
    _write_consent_payload(path, payload)
    return consent_record


def persist_all_restricted_backend_consents(
    *,
    settings: AppConfig,
    consent_source: str,
) -> list[BackendConsentRecord]:
    """Persists opt-in consent for all currently known restricted backends."""
    records: list[BackendConsentRecord] = []
    for policy in restricted_backend_policies():
        records.append(
            persist_backend_consent(
                settings=settings,
                backend_id=policy.backend_id,
                consent_source=consent_source,
            )
        )
    return records


def resolve_backend_access_context(*, settings: AppConfig) -> BackendAccessContext:
    """Resolves global/env/persisted policy controls used for access decisions."""
    return BackendAccessContext(
        restricted_backends_enabled=bool(settings.runtime_flags.restricted_backends),
        allowed_restricted_backends=parse_allowed_restricted_backends_env(),
        persisted_consents=load_persisted_backend_consents(settings=settings),
    )


def evaluate_backend_access(
    *,
    backend_id: str,
    restricted_backends_enabled: bool,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> LicenseDecision:
    """Evaluates whether the requested backend is allowed by runtime policy."""
    policy = _BACKEND_POLICIES.get(backend_id)
    if policy is None:
        return LicenseDecision(
            allowed=False,
            policy=BackendLicensePolicy(
                backend_id=backend_id,
                license_id="undefined",
                restricted=True,
                source_url="https://example.invalid/undefined-backend-policy",
            ),
            reason=(
                f"Backend {backend_id!r} is undefined in license policy manifest; "
                "execution is denied by default."
            ),
            access_source="undefined_backend",
        )

    if not policy.restricted:
        return LicenseDecision(
            allowed=True,
            policy=policy,
            reason="allowed",
            access_source="unrestricted",
        )

    if restricted_backends_enabled:
        return LicenseDecision(
            allowed=True,
            policy=policy,
            reason="allowed by SER_ENABLE_RESTRICTED_BACKENDS",
            access_source="global_override",
        )

    resolved_allowlist = (
        frozenset()
        if allowed_restricted_backends is None
        else allowed_restricted_backends
    )
    if backend_id in resolved_allowlist:
        return LicenseDecision(
            allowed=True,
            policy=policy,
            reason="allowed by SER_ALLOWED_RESTRICTED_BACKENDS",
            access_source="env_allowlist",
        )

    resolved_consents: Mapping[str, BackendConsentRecord] = (
        {} if persisted_consents is None else persisted_consents
    )
    consent_record = resolved_consents.get(backend_id)
    if consent_record is not None:
        return LicenseDecision(
            allowed=True,
            policy=policy,
            reason="allowed by persisted restricted-backend consent",
            access_source="persisted_consent",
            consent_record=consent_record,
        )

    if policy.restricted:
        return LicenseDecision(
            allowed=False,
            policy=policy,
            reason=(
                f"Backend {backend_id!r} is restricted by policy. "
                "Allow with one of: SER_ENABLE_RESTRICTED_BACKENDS=true "
                "(global run), SER_ALLOWED_RESTRICTED_BACKENDS=<backend_id> "
                "(per-backend run), `ser --accept-restricted-backends` "
                "(persist for active profile), or "
                "`ser --accept-all-restricted-backends` "
                "(persist all known restricted backends)."
            ),
            access_source="restricted_denied",
        )

    raise RuntimeError("Unreachable backend access policy state.")


def ensure_backend_access(
    *,
    backend_id: str,
    restricted_backends_enabled: bool,
    allowed_restricted_backends: frozenset[str] | None = None,
    persisted_consents: Mapping[str, BackendConsentRecord] | None = None,
) -> LicenseDecision:
    """Ensures backend access is permitted and raises on policy violation."""
    decision = evaluate_backend_access(
        backend_id=backend_id,
        restricted_backends_enabled=restricted_backends_enabled,
        allowed_restricted_backends=allowed_restricted_backends,
        persisted_consents=persisted_consents,
    )
    if not decision.allowed:
        raise BackendLicensePolicyError(decision.reason)
    return decision


def _dependency_manifest_fingerprint() -> str:
    """Builds deterministic fingerprint from dependency manifests when present."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_candidates = (
        repo_root / "uv.lock",
        repo_root / "pyproject.toml",
        repo_root / "requirements.txt",
    )
    hasher = hashlib.sha256()
    included = 0
    for manifest in manifest_candidates:
        if not manifest.is_file():
            continue
        hasher.update(manifest.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(manifest.read_bytes())
        hasher.update(b"\0")
        included += 1
    if included == 0:
        return "unavailable"
    return hasher.hexdigest()


def build_provenance_metadata(
    *,
    settings: AppConfig,
    backend_id: str,
    profile: str,
) -> dict[str, object]:
    """Builds machine-readable provenance payload for artifact/report metadata."""
    access_context = resolve_backend_access_context(settings=settings)
    decision = evaluate_backend_access(
        backend_id=backend_id,
        restricted_backends_enabled=access_context.restricted_backends_enabled,
        allowed_restricted_backends=access_context.allowed_restricted_backends,
        persisted_consents=access_context.persisted_consents,
    )
    code_revision = os.getenv("SER_CODE_REVISION", "").strip() or "unknown"
    provenance: dict[str, object] = {
        "code_revision": code_revision,
        "dependency_manifest_fingerprint": _dependency_manifest_fingerprint(),
        "backend_id": backend_id,
        "backend_license_id": decision.policy.license_id,
        "profile": profile,
        "dataset_glob_pattern": settings.dataset.glob_pattern,
        "runtime_restricted_backends_enabled": (
            access_context.restricted_backends_enabled
        ),
        "backend_is_restricted": decision.policy.restricted,
        "backend_access_allowed": decision.allowed,
        "backend_access_source": decision.access_source,
        "restricted_backend_policy_fingerprint": _policy_fingerprint(decision.policy),
        "license_source_url": decision.policy.source_url,
    }
    if decision.policy.restricted:
        provenance["restricted_backend_consent_source"] = (
            decision.consent_record.consent_source
            if decision.consent_record is not None
            else decision.access_source
        )
        if decision.consent_record is not None:
            provenance["restricted_backend_consent_accepted_at_utc"] = (
                decision.consent_record.accepted_at_utc
            )
    return provenance
