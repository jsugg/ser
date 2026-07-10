"""Local dataset registry.

The registry is a simple JSON file that maps dataset ids to:
  - dataset_root (where audio/labels live)
  - manifest_path (where the JSONL manifest is/will be written)
  - options (dataset-specific metadata needed to rebuild manifests)

The goal is to make training discover manifests automatically without requiring
manual `build-manifest` invocations.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ser.config import AppConfig


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    options: dict[str, str]


@dataclass(frozen=True)
class DatasetRegistryOptions:
    """Typed known registry options with immutable passthrough extras."""

    labels_csv_path: str | None
    audio_base_dir: str | None
    source_repo_id: str | None
    source_revision: str | None
    source_commit_sha: str | None
    default_language: str | None
    extras: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        """Returns one normalized dict representation for persistence."""
        normalized: dict[str, str] = dict(self.extras)
        if self.labels_csv_path is not None:
            normalized["labels_csv_path"] = self.labels_csv_path
        if self.audio_base_dir is not None:
            normalized["audio_base_dir"] = self.audio_base_dir
        if self.source_repo_id is not None:
            normalized["source_repo_id"] = self.source_repo_id
        if self.source_revision is not None:
            normalized["source_revision"] = self.source_revision
        if self.source_commit_sha is not None:
            normalized["source_commit_sha"] = self.source_commit_sha
        if self.default_language is not None:
            normalized["default_language"] = self.default_language
        return normalized


def parse_dataset_registry_options(
    options: Mapping[str, str] | None,
) -> DatasetRegistryOptions:
    """Parses and validates typed registry options."""

    raw: dict[str, str] = {}
    if options is not None:
        raw = {str(key): str(value) for key, value in options.items()}

    def _read_optional_str(key: str) -> str | None:
        value = raw.get(key)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    labels_csv_path = _read_optional_str("labels_csv_path")
    audio_base_dir = _read_optional_str("audio_base_dir")
    source_repo_id = _read_optional_str("source_repo_id")
    source_revision = _read_optional_str("source_revision")
    source_commit_sha = _read_optional_str("source_commit_sha")
    default_language = _read_optional_str("default_language")

    if source_repo_id is not None:
        if "/" not in source_repo_id or any(char.isspace() for char in source_repo_id):
            raise ValueError(
                "Invalid source_repo_id in dataset registry. Expected Hugging Face "
                "dataset id like `namespace/name`."
            )
    if source_revision is not None and any(char.isspace() for char in source_revision):
        raise ValueError("Invalid source_revision in dataset registry: whitespace is not allowed.")
    if source_commit_sha is not None:
        normalized_source_commit_sha = source_commit_sha.lower()
        if (
            any(char.isspace() for char in normalized_source_commit_sha)
            or len(normalized_source_commit_sha) < 7
            or len(normalized_source_commit_sha) > 64
            or any(char not in "0123456789abcdef" for char in normalized_source_commit_sha)
        ):
            raise ValueError(
                "Invalid source_commit_sha in dataset registry: expected 7-64 hex characters."
            )
        source_commit_sha = normalized_source_commit_sha

    known_keys = {
        "labels_csv_path",
        "audio_base_dir",
        "source_repo_id",
        "source_revision",
        "source_commit_sha",
        "default_language",
    }
    extras = tuple(sorted((key, value) for key, value in raw.items() if key not in known_keys))
    return DatasetRegistryOptions(
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        source_repo_id=source_repo_id,
        source_revision=source_revision,
        source_commit_sha=source_commit_sha,
        default_language=default_language,
        extras=extras,
    )


def _registry_path(settings: AppConfig) -> Path:
    data_root = settings.models.folder.parent
    return data_root / ".ser" / "dataset_registry.json"


def load_dataset_registry(*, settings: AppConfig) -> dict[str, DatasetRegistryEntry]:
    path = _registry_path(settings)
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    registry: dict[str, DatasetRegistryEntry] = {}
    for dataset_id, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        dataset_root = Path(str(payload.get("dataset_root", ""))).expanduser()
        manifest_path = Path(str(payload.get("manifest_path", ""))).expanduser()
        options = payload.get("options", {})
        if not isinstance(options, dict):
            options = {}
        registry[str(dataset_id)] = DatasetRegistryEntry(
            dataset_id=str(dataset_id),
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            options={str(k): str(v) for k, v in options.items()},
        )
    return registry


def save_dataset_registry(
    *, settings: AppConfig, registry: dict[str, DatasetRegistryEntry]
) -> None:
    path = _registry_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        dataset_id: {
            "dataset_root": str(entry.dataset_root),
            "manifest_path": str(entry.manifest_path),
            "options": dict(entry.options),
        }
        for dataset_id, entry in registry.items()
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def upsert_dataset_registry_entry(
    *,
    settings: AppConfig,
    dataset_id: str,
    dataset_root: Path,
    manifest_path: Path,
    options: dict[str, str] | None = None,
) -> None:
    registry = load_dataset_registry(settings=settings)
    normalized = dataset_id.strip().lower()
    parsed_options = parse_dataset_registry_options(options)
    registry[normalized] = DatasetRegistryEntry(
        dataset_id=normalized,
        dataset_root=dataset_root.expanduser(),
        manifest_path=manifest_path.expanduser(),
        options=parsed_options.as_dict(),
    )
    save_dataset_registry(settings=settings, registry=registry)


def remove_dataset_registry_entry(
    *,
    settings: AppConfig,
    dataset_id: str,
) -> DatasetRegistryEntry | None:
    """Removes one dataset registry entry when present."""

    registry = load_dataset_registry(settings=settings)
    normalized = dataset_id.strip().lower()
    existing = registry.pop(normalized, None)
    if existing is None:
        return None
    save_dataset_registry(settings=settings, registry=registry)
    return existing


def registered_manifest_paths(*, settings: AppConfig) -> tuple[Path, ...]:
    registry = load_dataset_registry(settings=settings)
    manifests = [
        entry.manifest_path for entry in registry.values() if entry.manifest_path.is_file()
    ]
    return tuple(sorted(set(manifests)))
