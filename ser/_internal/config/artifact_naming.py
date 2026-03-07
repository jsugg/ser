"""Artifact naming helpers shared by environment-driven config assembly."""

from __future__ import annotations

import re
from hashlib import sha1
from typing import Literal

type ArtifactProfileName = Literal["fast", "medium", "accurate", "accurate-research"]


def artifact_profile_from_runtime_flags(
    *,
    medium_profile: bool,
    accurate_profile: bool,
    accurate_research_profile: bool,
) -> ArtifactProfileName:
    """Resolves artifact profile from runtime flags using runtime precedence."""
    if accurate_research_profile:
        return "accurate-research"
    if accurate_profile:
        return "accurate"
    if medium_profile:
        return "medium"
    return "fast"


def artifact_model_id_suffix(model_id: str) -> str:
    """Builds a stable, filename-safe suffix for backend model ids."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip().lower())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "model"
    trimmed = cleaned[:48]
    digest = sha1(model_id.encode("utf-8")).hexdigest()[:10]
    return f"{trimmed}_{digest}"


def profile_artifact_file_names(
    *,
    profile: ArtifactProfileName,
    medium_model_id: str,
    accurate_model_id: str,
    accurate_research_model_id: str,
    default_fast_model_file_name: str,
    default_fast_secure_model_file_name: str,
    default_fast_training_report_file_name: str,
) -> tuple[str, str, str]:
    """Returns default artifact filenames for one profile/backend-model tuple."""
    if profile == "fast":
        return (
            default_fast_model_file_name,
            default_fast_secure_model_file_name,
            default_fast_training_report_file_name,
        )

    if profile == "medium":
        backend_model_id = medium_model_id
    elif profile == "accurate":
        backend_model_id = accurate_model_id
    else:
        backend_model_id = accurate_research_model_id

    model_suffix = artifact_model_id_suffix(backend_model_id)
    profile_token = profile.replace("-", "_")
    model_stem = f"ser_model_{profile_token}_{model_suffix}"
    report_stem = f"training_report_{profile_token}_{model_suffix}"
    return (f"{model_stem}.pkl", f"{model_stem}.skops", f"{report_stem}.json")


__all__ = [
    "artifact_model_id_suffix",
    "artifact_profile_from_runtime_flags",
    "profile_artifact_file_names",
]
