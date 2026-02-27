"""Dataset download/prepare helpers.

This module provides:
  - Best-effort dataset download/install commands.
  - Manifest building (JSONL) for supported corpora.
  - Registry updates so training can auto-discover manifests.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ser.config import AppConfig
from ser.data.adapters.biic_podcast import build_biic_podcast_manifest_jsonl
from ser.data.adapters.crema_d import build_crema_d_manifest_jsonl
from ser.data.adapters.msp_podcast import build_msp_podcast_manifest_jsonl
from ser.data.adapters.ravdess import build_ravdess_manifest_jsonl
from ser.data.dataset_registry import (
    DatasetRegistryEntry,
    upsert_dataset_registry_entry,
)
from ser.data.ontology import LabelOntology
from ser.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetDescriptor:
    """Static metadata for a supported dataset."""

    dataset_id: str
    display_name: str
    policy_id: str
    license_id: str
    source_url: str
    requires_manual_download: bool


SUPPORTED_DATASETS: dict[str, DatasetDescriptor] = {
    "ravdess": DatasetDescriptor(
        dataset_id="ravdess",
        display_name="RAVDESS",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://zenodo.org/records/1188976",
        requires_manual_download=True,
    ),
    "crema-d": DatasetDescriptor(
        dataset_id="crema-d",
        display_name="CREMA-D",
        policy_id="share_alike",
        license_id="odbl-1.0",
        source_url="https://github.com/CheyneyComputerScience/CREMA-D",
        requires_manual_download=False,
    ),
    "msp-podcast": DatasetDescriptor(
        dataset_id="msp-podcast",
        display_name="MSP-Podcast",
        policy_id="academic_only",
        license_id="msp-academic-license",
        source_url="https://lab-msp.com/MSP/MSP-Podcast.html",
        requires_manual_download=True,
    ),
    "biic-podcast": DatasetDescriptor(
        dataset_id="biic-podcast",
        display_name="BIIC-Podcast",
        policy_id="academic_only",
        license_id="biic-academic-license",
        source_url="https://biic.ee.nthu.edu.tw/",
        requires_manual_download=True,
    ),
}


def resolve_dataset_descriptor(dataset_id: str) -> DatasetDescriptor:
    """Resolves a supported dataset descriptor."""

    normalized = dataset_id.strip().lower()
    if not normalized or normalized not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset_id!r}. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}."
        )
    return SUPPORTED_DATASETS[normalized]


def default_dataset_root(settings: AppConfig, dataset_id: str) -> Path:
    """Default dataset directory under the SER data root."""

    resolved = resolve_dataset_descriptor(dataset_id)
    return settings.models.folder.parent / "datasets" / resolved.dataset_id


def default_manifest_path(settings: AppConfig, dataset_id: str) -> Path:
    """Default manifest path under the SER data root."""

    resolved = resolve_dataset_descriptor(dataset_id)
    return settings.models.folder.parent / "manifests" / f"{resolved.dataset_id}.jsonl"


def download_dataset(
    *,
    settings: AppConfig,
    dataset_id: str,
    dataset_root: Path,
) -> None:
    """Best-effort dataset download.

    Some corpora require manual access requests; this function will not bypass
    those requirements.
    """

    descriptor = resolve_dataset_descriptor(dataset_id)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if descriptor.dataset_id == "crema-d":
        if any(dataset_root.iterdir()):
            logger.info("CREMA-D target directory is not empty; skipping download.")
            return
        git_bin = shutil.which("git")
        if git_bin is None:
            raise RuntimeError(
                "git is required to download CREMA-D (git-lfs recommended)."
            )
        logger.info("Cloning CREMA-D into %s", dataset_root)
        subprocess.run(
            [
                git_bin,
                "clone",
                "--depth",
                "1",
                descriptor.source_url,
                str(dataset_root),
            ],
            check=True,
        )
        git_lfs = shutil.which("git-lfs")
        if git_lfs is not None:
            logger.info("Running git-lfs pull for CREMA-D")
            subprocess.run([git_lfs, "pull"], check=False, cwd=str(dataset_root))
        return

    instructions = [
        f"Dataset {descriptor.display_name} requires manual access/download.",
        f"Source: {descriptor.source_url}",
        f"Expected install directory: {dataset_root}",
    ]
    if descriptor.dataset_id == "msp-podcast":
        instructions.append(
            "MSP-Podcast is distributed under an academic access agreement. "
            "Follow the instructions on the MSP site to request and download the corpus."
        )
    if descriptor.dataset_id == "biic-podcast":
        instructions.append(
            "BIIC-Podcast access is typically granted by request (often academic-only). "
            "Obtain the corpus from the BIIC group and place it under the install directory."
        )
    logger.warning("\n".join(instructions))


def prepare_dataset_manifest(
    *,
    settings: AppConfig,
    dataset_id: str,
    dataset_root: Path,
    ontology: LabelOntology,
    manifest_path: Path,
    labels_csv_path: Path | None = None,
    audio_base_dir: Path | None = None,
    default_language: str | None = None,
) -> list[Path]:
    """Builds a dataset manifest and updates the registry.

    Returns:
        A list of manifest paths written.
    """

    descriptor = resolve_dataset_descriptor(dataset_id)
    language = default_language or settings.default_language
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    options: dict[str, str] = {}
    if labels_csv_path is not None:
        options["labels_csv_path"] = str(labels_csv_path)
    if audio_base_dir is not None:
        options["audio_base_dir"] = str(audio_base_dir)
    if default_language is not None:
        options["default_language"] = default_language

    built: list[Path] = []
    if descriptor.dataset_id == "ravdess":
        utterances = build_ravdess_manifest_jsonl(
            dataset_root=dataset_root,
            dataset_glob_pattern=str(
                dataset_root
                / settings.dataset.subfolder_prefix
                / settings.dataset.extension
            ),
            emotion_code_map=dict(settings.emotions),
            default_language=language,
            ontology=ontology,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return []
        built.append(manifest_path)
    elif descriptor.dataset_id == "crema-d":
        crema_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }
        pattern = (
            "AudioWAV/**/*.wav" if (dataset_root / "AudioWAV").exists() else "**/*.wav"
        )
        utterances = build_crema_d_manifest_jsonl(
            dataset_root=dataset_root,
            dataset_glob_pattern=pattern,
            emotion_code_map=crema_map,
            default_language=language,
            ontology=ontology,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return []
        built.append(manifest_path)
    elif descriptor.dataset_id == "msp-podcast":
        if labels_csv_path is None:
            raise ValueError(
                "MSP-Podcast manifest build requires --labels-csv-path pointing to labels_consensus.csv."
            )
        utterances = build_msp_podcast_manifest_jsonl(
            dataset_root=dataset_root,
            labels_csv_path=labels_csv_path,
            audio_base_dir=audio_base_dir,
            ontology=ontology,
            default_language=language,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return []
        built.append(manifest_path)
    elif descriptor.dataset_id == "biic-podcast":
        if labels_csv_path is None:
            raise ValueError(
                "BIIC-Podcast manifest build requires --labels-csv-path for the corpus label index."
            )
        utterances = build_biic_podcast_manifest_jsonl(
            dataset_root=dataset_root,
            labels_csv_path=labels_csv_path,
            audio_base_dir=audio_base_dir,
            ontology=ontology,
            default_language=language,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return []
        built.append(manifest_path)
    else:
        raise ValueError(f"Unsupported dataset {descriptor.dataset_id!r}.")

    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id=descriptor.dataset_id,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options=options,
    )
    return built


def prepare_from_registry_entry(
    *,
    settings: AppConfig,
    entry: DatasetRegistryEntry,
    ontology: LabelOntology,
) -> list[Path]:
    """Ensures a registered dataset has a manifest on disk."""

    if entry.manifest_path.is_file():
        return [entry.manifest_path]
    labels_csv = entry.options.get("labels_csv_path")
    audio_base = entry.options.get("audio_base_dir")
    default_language = entry.options.get("default_language")
    return prepare_dataset_manifest(
        settings=settings,
        dataset_id=entry.dataset_id,
        dataset_root=entry.dataset_root,
        ontology=ontology,
        manifest_path=entry.manifest_path,
        labels_csv_path=Path(labels_csv).expanduser() if labels_csv else None,
        audio_base_dir=Path(audio_base).expanduser() if audio_base else None,
        default_language=default_language,
    )
