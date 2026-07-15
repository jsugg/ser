"""Default dataset strategy implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ser._internal.data.adapters.biic_podcast import build_biic_podcast_manifest_jsonl
from ser._internal.data.adapters.crema_d import (
    CremaDDatasetIntegrityError,
    build_crema_d_manifest_jsonl,
    validate_crema_d_audio_files,
)
from ser._internal.data.adapters.msp_podcast import build_msp_podcast_manifest_jsonl
from ser._internal.data.adapters.ravdess import build_ravdess_manifest_jsonl
from ser._internal.data.msp_podcast_mirror import (
    DEFAULT_MSP_MIRROR_AUDIO_SUBDIR,
    DEFAULT_MSP_MIRROR_LABELS_FILE,
    DEFAULT_MSP_MIRROR_REPO_ID,
    DEFAULT_MSP_MIRROR_REVISION,
    prepare_msp_podcast_from_hf_mirror,
)
from ser._internal.data.ontology import LabelOntology
from ser._internal.data.public_dataset_downloads import (
    prepare_ravdess_from_zenodo,
)
from ser._internal.utils.logger import get_logger
from ser.config import AppConfig

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
    from ser._internal.data.dataset_prepare import DatasetDescriptor

logger = get_logger(__name__)

_CREMA_D_AUDIO_PATTERN = "AudioWAV/**/*.wav"
_SUBPROCESS_ERROR_DETAIL_LIMIT = 2_000


def _crema_d_audio_pattern(dataset_root: Path) -> str:
    """Returns the narrowest useful audio glob for one CREMA-D tree."""
    return _CREMA_D_AUDIO_PATTERN if (dataset_root / "AudioWAV").exists() else "**/*.wav"


def _require_crema_d_git_tools() -> str:
    """Resolves the mandatory Git and Git LFS executables for CREMA-D."""
    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError(
            "git is required to download CREMA-D. Install Git and retry dataset preparation."
        )
    if shutil.which("git-lfs") is None:
        raise RuntimeError(
            "git-lfs is required to download CREMA-D audio. Install it with "
            "`brew install git-lfs` or your operating system package manager, then retry."
        )
    return git_bin


def _run_crema_d_git_command(
    command: list[str],
    *,
    operation: str,
    cwd: Path | None = None,
) -> None:
    """Runs one mandatory CREMA-D Git command with bounded diagnostics."""
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except (OSError, subprocess.CalledProcessError) as err:
        stderr = getattr(err, "stderr", None)
        detail = stderr.strip() if isinstance(stderr, str) else str(err)
        if len(detail) > _SUBPROCESS_ERROR_DETAIL_LIMIT:
            detail = f"...{detail[-_SUBPROCESS_ERROR_DETAIL_LIMIT:]}"
        raise RuntimeError(f"CREMA-D {operation} failed: {detail}") from err


def _validate_crema_d_tree(dataset_root: Path) -> None:
    """Validates all CREMA-D audio without decoding complete waveforms."""
    validate_crema_d_audio_files(
        dataset_root=dataset_root,
        dataset_glob_pattern=_crema_d_audio_pattern(dataset_root),
    )


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
        dataset_root.mkdir(parents=True, exist_ok=True)
        has_existing_content = any(dataset_root.iterdir())
        if has_existing_content:
            try:
                _validate_crema_d_tree(dataset_root)
            except CremaDDatasetIntegrityError as integrity_error:
                if not (dataset_root / ".git").exists():
                    raise RuntimeError(
                        f"Existing CREMA-D directory is incomplete: {integrity_error} "
                        f"Move it aside and retry: {dataset_root}"
                    ) from integrity_error
                git_bin = _require_crema_d_git_tools()
                logger.info("Repairing incomplete CREMA-D Git LFS checkout at %s", dataset_root)
                for args, operation in (
                    (("lfs", "install", "--local"), "Git LFS initialization"),
                    (("lfs", "pull"), "Git LFS pull"),
                    (("lfs", "checkout"), "Git LFS checkout"),
                ):
                    _run_crema_d_git_command(
                        [git_bin, *args],
                        operation=operation,
                        cwd=dataset_root,
                    )
                _validate_crema_d_tree(dataset_root)
                return (None, None)
            logger.info("Existing CREMA-D dataset passed integrity checks; skipping download.")
            return (None, None)

        git_bin = _require_crema_d_git_tools()
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{dataset_root.name}.staging-", dir=dataset_root.parent)
        )
        staging_root.rmdir()
        try:
            logger.info("Cloning CREMA-D into staging directory %s", staging_root)
            _run_crema_d_git_command(
                [
                    git_bin,
                    "clone",
                    "--depth",
                    "1",
                    descriptor.source_url,
                    str(staging_root),
                ],
                operation="clone",
            )
            for args, operation in (
                (("lfs", "install", "--local"), "Git LFS initialization"),
                (("lfs", "pull"), "Git LFS pull"),
                (("lfs", "checkout"), "Git LFS checkout"),
            ):
                _run_crema_d_git_command(
                    [git_bin, *args],
                    operation=operation,
                    cwd=staging_root,
                )
            _validate_crema_d_tree(staging_root)
            dataset_root.rmdir()
            os.replace(staging_root, dataset_root)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)
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
        pattern = _crema_d_audio_pattern(dataset_root)
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
