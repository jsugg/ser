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
from dataclasses import dataclass
from pathlib import Path

from ser.config import AppConfig


@dataclass(frozen=True)
class DatasetRegistryEntry:
    dataset_id: str
    dataset_root: Path
    manifest_path: Path
    options: dict[str, str]


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
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
    registry[normalized] = DatasetRegistryEntry(
        dataset_id=normalized,
        dataset_root=dataset_root.expanduser(),
        manifest_path=manifest_path.expanduser(),
        options=dict(options or {}),
    )
    save_dataset_registry(settings=settings, registry=registry)


def registered_manifest_paths(*, settings: AppConfig) -> tuple[Path, ...]:
    registry = load_dataset_registry(settings=settings)
    manifests = [
        entry.manifest_path
        for entry in registry.values()
        if entry.manifest_path.is_file()
    ]
    return tuple(sorted(set(manifests)))
