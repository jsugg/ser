"""Dataset capability snapshot services."""

from __future__ import annotations

import json
from pathlib import Path

from ser._internal.data.application.models import (
    DatasetCapabilitySnapshotEntry,
    DatasetRegistrySnapshot,
)
from ser._internal.data.application.registry_snapshot import (
    collect_dataset_registry_snapshot,
)
from ser.config import AppConfig
from ser.data.dataset_capabilities import resolve_dataset_capability_profile
from ser.data.dataset_prepare import SUPPORTED_DATASETS


def _resolve_manifest_audio_path(
    *,
    raw_audio_path: str,
    dataset_root: Path | None,
) -> Path:
    candidate = Path(raw_audio_path).expanduser()
    if candidate.is_absolute():
        return candidate
    if dataset_root is not None:
        return (dataset_root / candidate).expanduser()
    return candidate


def _collect_manifest_audio_stats(
    *,
    manifest_path: Path | None,
    dataset_root: Path | None,
) -> tuple[int, int, int, int]:
    if manifest_path is None or not manifest_path.is_file():
        return (0, 0, 0, 0)

    referenced = 0
    present = 0
    nonempty = 0
    total_bytes = 0
    seen_paths: set[Path] = set()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            raw_audio_path = payload.get("audio_path")
            if not isinstance(raw_audio_path, str) or not raw_audio_path.strip():
                continue
            resolved_path = _resolve_manifest_audio_path(
                raw_audio_path=raw_audio_path,
                dataset_root=dataset_root,
            )
            if resolved_path in seen_paths:
                continue
            seen_paths.add(resolved_path)
            referenced += 1
            if not resolved_path.is_file():
                continue
            present += 1
            size_bytes = resolved_path.stat().st_size
            total_bytes += size_bytes
            if size_bytes > 0:
                nonempty += 1
    return (referenced, present, nonempty, total_bytes)


def collect_dataset_capability_snapshot(
    *,
    settings: AppConfig,
    include_uninstalled: bool = False,
) -> tuple[DatasetCapabilitySnapshotEntry, ...]:
    """Collects capability records for installed datasets (or all supported)."""

    registry_snapshot: DatasetRegistrySnapshot = collect_dataset_registry_snapshot(
        settings=settings
    )
    registry_by_id = {entry.dataset_id: entry for entry in registry_snapshot.entries}
    rows: list[DatasetCapabilitySnapshotEntry] = []
    for dataset_id in sorted(SUPPORTED_DATASETS):
        descriptor = SUPPORTED_DATASETS[dataset_id]
        registry_entry = registry_by_id.get(dataset_id)
        registered = registry_entry is not None
        manifest_exists = (
            registry_entry.manifest_path.is_file() if registry_entry is not None else False
        )
        (
            referenced_audio_files,
            present_audio_files,
            nonempty_audio_files,
            dataset_size_bytes,
        ) = _collect_manifest_audio_stats(
            manifest_path=registry_entry.manifest_path if registry_entry else None,
            dataset_root=registry_entry.dataset_root if registry_entry else None,
        )
        installed = registered and manifest_exists and nonempty_audio_files > 0
        if not include_uninstalled and not installed:
            continue
        profile = resolve_dataset_capability_profile(dataset_id)
        rows.append(
            DatasetCapabilitySnapshotEntry(
                dataset_id=dataset_id,
                display_name=descriptor.display_name,
                registered=registered,
                installed=installed,
                manifest_exists=manifest_exists,
                dataset_root=registry_entry.dataset_root if registry_entry else None,
                manifest_path=registry_entry.manifest_path if registry_entry else None,
                referenced_audio_files=referenced_audio_files,
                present_audio_files=present_audio_files,
                nonempty_audio_files=nonempty_audio_files,
                dataset_size_bytes=dataset_size_bytes,
                source_url=descriptor.source_url,
                policy_id=descriptor.policy_id,
                license_id=descriptor.license_id,
                modalities=profile.modalities,
                label_schema=profile.label_schema,
                has_label_mapping=profile.has_label_mapping,
                supervised_ser_candidate=profile.supervised_ser_candidate,
                ssl_candidate=profile.ssl_candidate,
                multimodal_candidate=profile.multimodal_candidate,
                mergeable_with_emotion_ontology=profile.mergeable_with_emotion_ontology,
                recommended_uses=profile.recommended_uses,
                notes=profile.notes,
            )
        )
    return tuple(rows)


def build_dataset_capability_snapshot_json_payload(
    rows: tuple[DatasetCapabilitySnapshotEntry, ...],
) -> dict[str, object]:
    """Builds JSON-serializable payload for dataset capability snapshot output."""

    entries: list[dict[str, object]] = []
    for row in rows:
        entries.append(
            {
                "dataset_id": row.dataset_id,
                "display_name": row.display_name,
                "registered": row.registered,
                "installed": row.installed,
                "manifest_exists": row.manifest_exists,
                "dataset_root": (str(row.dataset_root) if row.dataset_root is not None else None),
                "manifest_path": (
                    str(row.manifest_path) if row.manifest_path is not None else None
                ),
                "referenced_audio_files": row.referenced_audio_files,
                "present_audio_files": row.present_audio_files,
                "nonempty_audio_files": row.nonempty_audio_files,
                "dataset_size_bytes": row.dataset_size_bytes,
                "source_url": row.source_url,
                "policy_id": row.policy_id,
                "license_id": row.license_id,
                "modalities": list(row.modalities),
                "label_schema": row.label_schema,
                "has_label_mapping": row.has_label_mapping,
                "supervised_ser_candidate": row.supervised_ser_candidate,
                "ssl_candidate": row.ssl_candidate,
                "multimodal_candidate": row.multimodal_candidate,
                "mergeable_with_emotion_ontology": row.mergeable_with_emotion_ontology,
                "recommended_uses": list(row.recommended_uses),
                "notes": list(row.notes),
            }
        )
    return {"entries": entries}
