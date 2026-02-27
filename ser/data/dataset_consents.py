"""Dataset policy/license consent enforcement.

This project supports training on multiple datasets with potentially restrictive
policies (e.g., academic-only) and/or licenses.

Expected behavior:
  - Without prior consent: training must refuse and explain how to consent.
  - With consent: training proceeds.

Consent is persisted locally under the SER data root.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ser.config import AppConfig
from ser.data.manifest import Utterance

_CONSENT_STORE_ENV = "SER_DATASET_CONSENTS_FILE"
_CONSENT_SCHEMA_VERSION = 1
_CONSENT_FILE_NAME = "dataset_consents.json"
_RESTRICTED_POLICY_IDS = frozenset({"academic_only", "share_alike", "noncommercial"})


class DatasetConsentError(RuntimeError):
    """Raised when required dataset policy/license consents are missing."""


@dataclass(frozen=True)
class PersistedDatasetConsents:
    policy_consents: dict[str, str]
    license_consents: dict[str, str]


def _consents_path(settings: AppConfig) -> Path:
    explicit_path = os.getenv(_CONSENT_STORE_ENV, "").strip()
    if explicit_path:
        return Path(explicit_path).expanduser()
    return settings.models.folder.parent / ".ser" / _CONSENT_FILE_NAME


def is_policy_restricted(policy_id: str | None) -> bool:
    """Whether a dataset policy requires explicit acknowledgement."""

    normalized = (policy_id or "").strip().lower()
    return normalized in _RESTRICTED_POLICY_IDS


def load_persisted_dataset_consents(*, settings: AppConfig) -> PersistedDatasetConsents:
    path = _consents_path(settings)
    if not path.is_file():
        return PersistedDatasetConsents(policy_consents={}, license_consents={})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as err:
        raise RuntimeError(
            f"Dataset consent store at {path} is unreadable: {err}"
        ) from err
    if not isinstance(raw, dict):
        raise RuntimeError(f"Dataset consent store at {path} must be a JSON object.")

    schema_version = raw.get("schema_version")
    if schema_version is not None and schema_version != _CONSENT_SCHEMA_VERSION:
        raise RuntimeError(
            "Dataset consent store schema mismatch. "
            f"Expected {_CONSENT_SCHEMA_VERSION}, got {schema_version!r}."
        )
    policy_consents = raw.get("policy_consents", {})
    license_consents = raw.get("license_consents", {})
    if not isinstance(policy_consents, dict):
        policy_consents = {}
    if not isinstance(license_consents, dict):
        license_consents = {}
    return PersistedDatasetConsents(
        policy_consents={str(k): str(v) for k, v in policy_consents.items()},
        license_consents={str(k): str(v) for k, v in license_consents.items()},
    )


def persist_dataset_consents(
    *,
    settings: AppConfig,
    accept_policy_ids: list[str] | None = None,
    accept_license_ids: list[str] | None = None,
    source: str,
) -> None:
    """Persist dataset policy/license acknowledgements locally."""

    accept_policy_ids = accept_policy_ids or []
    accept_license_ids = accept_license_ids or []
    existing = load_persisted_dataset_consents(settings=settings)
    policy_consents = dict(existing.policy_consents)
    license_consents = dict(existing.license_consents)

    for policy_id in accept_policy_ids:
        normalized = policy_id.strip().lower()
        if normalized:
            policy_consents[normalized] = source
    for license_id in accept_license_ids:
        normalized = license_id.strip().lower()
        if normalized:
            license_consents[normalized] = source

    path = _consents_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _CONSENT_SCHEMA_VERSION,
        "policy_consents": dict(sorted(policy_consents.items())),
        "license_consents": dict(sorted(license_consents.items())),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
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


def compute_missing_dataset_consents(
    *,
    settings: AppConfig,
    utterances: list[Utterance],
) -> tuple[set[str], set[str]]:
    """Returns missing (policy_ids, license_ids) needed for training."""

    persisted = load_persisted_dataset_consents(settings=settings)
    required_policies: set[str] = set()
    required_licenses: set[str] = set()
    for utterance in utterances:
        policy_id = (utterance.dataset_policy_id or "").strip().lower()
        if not policy_id:
            continue
        if not is_policy_restricted(policy_id):
            continue
        required_policies.add(policy_id)
        license_id = (utterance.dataset_license_id or "").strip().lower()
        if license_id:
            required_licenses.add(license_id)

    missing_policies = required_policies - set(persisted.policy_consents)
    missing_licenses = required_licenses - set(persisted.license_consents)
    return missing_policies, missing_licenses


def ensure_dataset_consents(
    *, settings: AppConfig, utterances: list[Utterance]
) -> None:
    """Raises if the training set requires policy/license acknowledgements."""

    missing_policies, missing_licenses = compute_missing_dataset_consents(
        settings=settings,
        utterances=utterances,
    )
    if not missing_policies and not missing_licenses:
        return

    parts: list[str] = [
        "Missing dataset acknowledgements.",
        "To proceed, persist consent(s) locally via:",
    ]
    if missing_policies:
        parts.append(
            "  ser configure --accept-dataset-policy "
            + " ".join(sorted(missing_policies))
            + " --persist"
        )
    if missing_licenses:
        parts.append(
            "  ser configure --accept-dataset-license "
            + " ".join(sorted(missing_licenses))
            + " --persist"
        )
    raise DatasetConsentError("\n".join(parts))
