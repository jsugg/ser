"""Default dataset strategy implementations."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ser.config import AppConfig
from ser.data.adapters.biic_podcast import build_biic_podcast_manifest_jsonl
from ser.data.adapters.crema_d import build_crema_d_manifest_jsonl
from ser.data.adapters.msp_podcast import build_msp_podcast_manifest_jsonl
from ser.data.adapters.ravdess import build_ravdess_manifest_jsonl
from ser.data.msp_podcast_mirror import (
    DEFAULT_MSP_MIRROR_AUDIO_SUBDIR,
    DEFAULT_MSP_MIRROR_LABELS_FILE,
    DEFAULT_MSP_MIRROR_REPO_ID,
    DEFAULT_MSP_MIRROR_REVISION,
    prepare_msp_podcast_from_hf_mirror,
)
from ser.data.ontology import LabelOntology
from ser.data.public_dataset_downloads import (
    prepare_ravdess_from_zenodo,
)
from ser.utils.logger import get_logger

from .auto_csv import (
    AsvpEsdDatasetStrategy,
    AttHackDatasetStrategy,
    CafeDatasetStrategy,
    CoraaSerDatasetStrategy,
    Emodb2DatasetStrategy,
    EmovDbDatasetStrategy,
    EscorpusPeDatasetStrategy,
    JlCorpusDatasetStrategy,
    MesdDatasetStrategy,
    OreauFrenchEsdDatasetStrategy,
    PavoqueDatasetStrategy,
    SpanishMeacorpus2023DatasetStrategy,
)
from .base import DatasetStrategy, PreparedManifestResult

if TYPE_CHECKING:
    from ser.data.dataset_prepare import DatasetDescriptor

logger = get_logger(__name__)


def _log_manual_download_instructions(
    *,
    descriptor: DatasetDescriptor,
    dataset_root: Path,
) -> tuple[None, None]:
    instructions = [
        f"Dataset {descriptor.display_name} requires manual access/download.",
        f"Source: {descriptor.source_url}",
        f"Expected install directory: {dataset_root}",
    ]
    if descriptor.dataset_id == "biic-podcast":
        instructions.append(
            "BIIC-Podcast access is typically granted by request (often academic-only). "
            "Obtain the corpus from the BIIC group and place it under the install directory."
        )
    logger.warning("\n".join(instructions))
    return (None, None)


class RavdessDatasetStrategy:
    """Strategy for RAVDESS dataset operations."""

    supports_source_overrides = False

    def download(
        self,
        *,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        source_repo_id: str | None,
        source_revision: str | None,
    ) -> tuple[str | None, str | None]:
        del descriptor, source_repo_id, source_revision
        artifacts = prepare_ravdess_from_zenodo(dataset_root=dataset_root)
        logger.info(
            "RAVDESS prepared at %s (files_seen=%s source_manifest=%s)",
            artifacts.dataset_root,
            artifacts.files_seen,
            artifacts.source_manifest_path,
        )
        return (None, None)

    def prepare_manifest(
        self,
        *,
        settings: AppConfig,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        ontology: LabelOntology,
        manifest_path: Path,
        language: str,
        labels_csv_path: Path | None,
        audio_base_dir: Path | None,
        options: dict[str, str],
    ) -> PreparedManifestResult:
        del descriptor, labels_csv_path, audio_base_dir
        utterances = build_ravdess_manifest_jsonl(
            dataset_root=dataset_root,
            dataset_glob_pattern=str(
                dataset_root / settings.dataset.subfolder_prefix / settings.dataset.extension
            ),
            emotion_code_map=dict(settings.emotions),
            default_language=language,
            ontology=ontology,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return PreparedManifestResult(manifest_paths=(), options=options)
        return PreparedManifestResult(
            manifest_paths=(manifest_path,),
            options=options,
        )


class CremaDDatasetStrategy:
    """Strategy for CREMA-D dataset operations."""

    supports_source_overrides = False

    def download(
        self,
        *,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        source_repo_id: str | None,
        source_revision: str | None,
    ) -> tuple[str | None, str | None]:
        del source_repo_id, source_revision
        if any(dataset_root.iterdir()):
            logger.info("CREMA-D target directory is not empty; skipping download.")
            return (None, None)
        git_bin = shutil.which("git")
        if git_bin is None:
            raise RuntimeError("git is required to download CREMA-D (git-lfs recommended).")
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
        return (None, None)

    def prepare_manifest(
        self,
        *,
        settings: AppConfig,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        ontology: LabelOntology,
        manifest_path: Path,
        language: str,
        labels_csv_path: Path | None,
        audio_base_dir: Path | None,
        options: dict[str, str],
    ) -> PreparedManifestResult:
        del descriptor, labels_csv_path, audio_base_dir
        crema_map = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }
        pattern = "AudioWAV/**/*.wav" if (dataset_root / "AudioWAV").exists() else "**/*.wav"
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
            return PreparedManifestResult(manifest_paths=(), options=options)
        return PreparedManifestResult(
            manifest_paths=(manifest_path,),
            options=options,
        )


class MspPodcastDatasetStrategy:
    """Strategy for MSP-Podcast dataset operations."""

    supports_source_overrides = True

    def download(
        self,
        *,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        source_repo_id: str | None,
        source_revision: str | None,
    ) -> tuple[str | None, str | None]:
        del descriptor
        resolved_repo_id = source_repo_id or DEFAULT_MSP_MIRROR_REPO_ID
        resolved_revision = source_revision or DEFAULT_MSP_MIRROR_REVISION
        artifacts = prepare_msp_podcast_from_hf_mirror(
            dataset_root=dataset_root,
            repo_id=resolved_repo_id,
            revision=resolved_revision,
        )
        logger.info(
            "MSP-Podcast mirror prepared at %s (source=%s@%s labels=%s audio=%s rows=%s labels_written=%s)",
            artifacts.dataset_root,
            resolved_repo_id,
            resolved_revision,
            artifacts.labels_csv_path,
            artifacts.audio_dir,
            artifacts.rows_seen,
            artifacts.labels_written,
        )
        return (resolved_repo_id, resolved_revision)

    def prepare_manifest(
        self,
        *,
        settings: AppConfig,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        ontology: LabelOntology,
        manifest_path: Path,
        language: str,
        labels_csv_path: Path | None,
        audio_base_dir: Path | None,
        options: dict[str, str],
    ) -> PreparedManifestResult:
        del descriptor
        resolved_labels_csv_path = labels_csv_path
        if resolved_labels_csv_path is None:
            generated_labels_path = dataset_root / DEFAULT_MSP_MIRROR_LABELS_FILE
            if generated_labels_path.is_file():
                resolved_labels_csv_path = generated_labels_path
        if resolved_labels_csv_path is None:
            raise ValueError(
                "MSP-Podcast manifest build requires labels CSV. "
                "Provide --labels-csv-path, or run `ser data download --dataset msp-podcast` "
                "to generate dataset_root/labels.csv first."
            )
        resolved_audio_base_dir = audio_base_dir
        if resolved_audio_base_dir is None:
            generated_audio_dir = dataset_root / DEFAULT_MSP_MIRROR_AUDIO_SUBDIR
            if generated_audio_dir.is_dir():
                resolved_audio_base_dir = generated_audio_dir
        options["labels_csv_path"] = str(resolved_labels_csv_path)
        if resolved_audio_base_dir is not None:
            options["audio_base_dir"] = str(resolved_audio_base_dir)
        utterances = build_msp_podcast_manifest_jsonl(
            dataset_root=dataset_root,
            labels_csv_path=resolved_labels_csv_path,
            audio_base_dir=resolved_audio_base_dir,
            ontology=ontology,
            default_language=language,
            max_failed_file_ratio=settings.data_loader.max_failed_file_ratio,
            output_path=manifest_path,
        )
        if utterances is None:
            return PreparedManifestResult(manifest_paths=(), options=options)
        return PreparedManifestResult(
            manifest_paths=(manifest_path,),
            options=options,
        )


class BiicPodcastDatasetStrategy:
    """Strategy for BIIC-Podcast dataset operations."""

    supports_source_overrides = False

    def download(
        self,
        *,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        source_repo_id: str | None,
        source_revision: str | None,
    ) -> tuple[str | None, str | None]:
        del source_repo_id, source_revision
        return _log_manual_download_instructions(
            descriptor=descriptor,
            dataset_root=dataset_root,
        )

    def prepare_manifest(
        self,
        *,
        settings: AppConfig,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        ontology: LabelOntology,
        manifest_path: Path,
        language: str,
        labels_csv_path: Path | None,
        audio_base_dir: Path | None,
        options: dict[str, str],
    ) -> PreparedManifestResult:
        del descriptor
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
            return PreparedManifestResult(manifest_paths=(), options=options)
        return PreparedManifestResult(
            manifest_paths=(manifest_path,),
            options=options,
        )


def build_default_dataset_strategies() -> dict[str, DatasetStrategy]:
    """Builds the default dataset-id to strategy mapping."""

    return {
        "ravdess": RavdessDatasetStrategy(),
        "crema-d": CremaDDatasetStrategy(),
        "msp-podcast": MspPodcastDatasetStrategy(),
        "emodb-2.0": Emodb2DatasetStrategy(),
        "escorpus-pe": EscorpusPeDatasetStrategy(),
        "mesd": MesdDatasetStrategy(),
        "oreau-french-esd": OreauFrenchEsdDatasetStrategy(),
        "jl-corpus": JlCorpusDatasetStrategy(),
        "cafe": CafeDatasetStrategy(),
        "asvp-esd": AsvpEsdDatasetStrategy(),
        "emov-db": EmovDbDatasetStrategy(),
        "pavoque": PavoqueDatasetStrategy(),
        "att-hack": AttHackDatasetStrategy(),
        "coraa-ser": CoraaSerDatasetStrategy(),
        "spanish-meacorpus-2023": SpanishMeacorpus2023DatasetStrategy(),
        "biic-podcast": BiicPodcastDatasetStrategy(),
    }
