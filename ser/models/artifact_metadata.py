"""Model artifact metadata validation and normalization helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def read_positive_int(metadata: dict[str, object], field_name: str) -> int:
    """Returns a required positive integer field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, int) or raw_value <= 0:
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return raw_value


def read_positive_float(metadata: dict[str, object], field_name: str) -> float:
    """Returns a required positive float field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, int | float) or float(raw_value) <= 0.0:
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return float(raw_value)


def read_non_empty_text(metadata: dict[str, object], field_name: str) -> str:
    """Returns a required non-empty text field from metadata."""
    raw_value = metadata.get(field_name)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"Model artifact metadata contains invalid {field_name!r} value."
        )
    return raw_value


def read_labels(metadata: dict[str, object]) -> list[str]:
    """Returns validated class labels from metadata."""
    raw_labels = metadata.get("labels")
    if not isinstance(raw_labels, list):
        raise ValueError("Model artifact metadata contains invalid 'labels' value.")
    labels: list[str] = []
    for raw_label in raw_labels:
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ValueError("Model artifact metadata contains invalid 'labels' value.")
        labels.append(raw_label)
    if not labels:
        raise ValueError("Model artifact metadata contains invalid 'labels' value.")
    return sorted(set(labels))


def normalize_provenance_metadata(raw_value: object) -> dict[str, object]:
    """Validates optional provenance metadata, defaulting to an empty payload."""
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError("Model artifact metadata contains invalid 'provenance' value.")

    allowed_text_fields = (
        "code_revision",
        "dependency_manifest_fingerprint",
        "backend_id",
        "backend_license_id",
        "profile",
        "dataset_glob_pattern",
        "backend_access_source",
        "restricted_backend_consent_source",
        "restricted_backend_consent_accepted_at_utc",
        "restricted_backend_policy_fingerprint",
        "license_source_url",
    )
    normalized: dict[str, object] = {}
    for field_name in allowed_text_fields:
        field_value = raw_value.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(
                "Model artifact metadata contains invalid provenance field "
                f"{field_name!r}."
            )
        normalized[field_name] = field_value

    allowed_bool_fields = (
        "runtime_restricted_backends_enabled",
        "backend_is_restricted",
        "backend_access_allowed",
    )
    for field_name in allowed_bool_fields:
        field_value = raw_value.get(field_name)
        if field_value is None:
            continue
        if not isinstance(field_value, bool):
            raise ValueError(
                "Model artifact metadata contains invalid provenance field "
                f"{field_name!r}."
            )
        normalized[field_name] = field_value
    return normalized


def normalize_v2_artifact_metadata(
    metadata: dict[str, object],
    *,
    artifact_version: int,
) -> dict[str, object]:
    """Validates and normalizes artifact metadata to v2 shape."""
    resolved_artifact_version = read_positive_int(metadata, "artifact_version")
    if resolved_artifact_version != artifact_version:
        raise ValueError(
            "Model artifact metadata contains unsupported 'artifact_version' value."
        )
    feature_vector_size = read_positive_int(metadata, "feature_vector_size")
    feature_dim = read_positive_int(metadata, "feature_dim")
    if feature_dim != feature_vector_size:
        raise ValueError(
            "Model artifact metadata 'feature_dim' must match 'feature_vector_size'."
        )
    normalized: dict[str, object] = {
        "artifact_version": artifact_version,
        "artifact_schema_version": read_non_empty_text(
            metadata, "artifact_schema_version"
        ),
        "created_at_utc": read_non_empty_text(metadata, "created_at_utc"),
        "feature_vector_size": feature_vector_size,
        "training_samples": read_positive_int(metadata, "training_samples"),
        "labels": read_labels(metadata),
        "backend_id": read_non_empty_text(metadata, "backend_id"),
        "profile": read_non_empty_text(metadata, "profile"),
        "feature_dim": feature_dim,
        "frame_size_seconds": read_positive_float(metadata, "frame_size_seconds"),
        "frame_stride_seconds": read_positive_float(metadata, "frame_stride_seconds"),
        "pooling_strategy": read_non_empty_text(metadata, "pooling_strategy"),
        "provenance": normalize_provenance_metadata(metadata.get("provenance")),
    }
    backend_model_id = metadata.get("backend_model_id")
    if backend_model_id is not None:
        if not isinstance(backend_model_id, str) or not backend_model_id.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'backend_model_id' value."
            )
        normalized["backend_model_id"] = backend_model_id.strip()
    torch_device = metadata.get("torch_device")
    if torch_device is not None:
        if not isinstance(torch_device, str) or not torch_device.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_device' value."
            )
        normalized["torch_device"] = torch_device.strip().lower()
    torch_dtype = metadata.get("torch_dtype")
    if torch_dtype is not None:
        if not isinstance(torch_dtype, str) or not torch_dtype.strip():
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_dtype' value."
            )
        normalized["torch_dtype"] = torch_dtype.strip().lower()
    return normalized


def build_v2_artifact_metadata(
    *,
    artifact_version: int,
    artifact_schema_version: str,
    feature_vector_size: int,
    training_samples: int,
    labels: list[str],
    backend_id: str,
    profile: str,
    feature_dim: int | None,
    frame_size_seconds: float,
    frame_stride_seconds: float,
    pooling_strategy: str,
    backend_model_id: str | None,
    torch_device: str | None,
    torch_dtype: str | None,
    provenance: dict[str, object] | None,
) -> dict[str, object]:
    """Builds normalized v2 artifact metadata for persisted model envelopes."""
    resolved_feature_dim = feature_vector_size if feature_dim is None else feature_dim
    payload: dict[str, object] = {
        "artifact_version": artifact_version,
        "artifact_schema_version": artifact_schema_version,
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "feature_vector_size": feature_vector_size,
        "training_samples": training_samples,
        "labels": labels,
        "backend_id": backend_id,
        "profile": profile,
        "feature_dim": resolved_feature_dim,
        "frame_size_seconds": frame_size_seconds,
        "frame_stride_seconds": frame_stride_seconds,
        "pooling_strategy": pooling_strategy,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    if backend_model_id is not None:
        resolved_backend_model_id = backend_model_id.strip()
        if not resolved_backend_model_id:
            raise ValueError(
                "Model artifact metadata contains invalid 'backend_model_id' value."
            )
        payload["backend_model_id"] = resolved_backend_model_id
    if torch_device is not None:
        resolved_torch_device = torch_device.strip().lower()
        if not resolved_torch_device:
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_device' value."
            )
        payload["torch_device"] = resolved_torch_device
    if torch_dtype is not None:
        resolved_torch_dtype = torch_dtype.strip().lower()
        if not resolved_torch_dtype:
            raise ValueError(
                "Model artifact metadata contains invalid 'torch_dtype' value."
            )
        payload["torch_dtype"] = resolved_torch_dtype
    return normalize_v2_artifact_metadata(
        payload,
        artifact_version=artifact_version,
    )
