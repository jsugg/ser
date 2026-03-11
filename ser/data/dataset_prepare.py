"""Dataset download/prepare helpers.

This module provides:
  - Best-effort dataset download/install commands.
  - Manifest building (JSONL) for supported corpora.
  - Registry updates so training can auto-discover manifests.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ser.config import AppConfig
from ser.data.dataset_registry import (
    DatasetRegistryEntry,
    load_dataset_registry,
    parse_dataset_registry_options,
    upsert_dataset_registry_entry,
)
from ser.data.msp_podcast_mirror import (
    DEFAULT_MSP_MIRROR_MANIFEST_FILE,
)
from ser.data.ontology import LabelOntology
from ser.data.strategies import (
    DatasetStrategy,
    DatasetStrategyRegistry,
    build_default_dataset_strategies,
)
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


@dataclass(frozen=True, slots=True)
class DatasetRegistryHealthIssue:
    """Typed registry health issue emitted by consistency checks."""

    dataset_id: str
    code: str
    message: str


SUPPORTED_DATASETS: dict[str, DatasetDescriptor] = {
    "ravdess": DatasetDescriptor(
        dataset_id="ravdess",
        display_name="RAVDESS",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://zenodo.org/records/1188976",
        requires_manual_download=False,
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
        source_url="https://huggingface.co/datasets/AbstractTTS/PODCAST",
        requires_manual_download=False,
    ),
    "emodb-2.0": DatasetDescriptor(
        dataset_id="emodb-2.0",
        display_name="EmoDB 2.0",
        policy_id="open",
        license_id="cc-by-4.0",
        source_url="https://zenodo.org/records/17651657",
        requires_manual_download=False,
    ),
    "escorpus-pe": DatasetDescriptor(
        dataset_id="escorpus-pe",
        display_name="ESCorpus-PE",
        policy_id="open",
        license_id="cc-by-4.0",
        source_url="https://zenodo.org/records/5793223",
        requires_manual_download=False,
    ),
    "mesd": DatasetDescriptor(
        dataset_id="mesd",
        display_name="MESD",
        policy_id="open",
        license_id="cc-by-4.0",
        source_url="https://data.mendeley.com/datasets/cy34mh68j9/5",
        requires_manual_download=False,
    ),
    "oreau-french-esd": DatasetDescriptor(
        dataset_id="oreau-french-esd",
        display_name="Oreau French ESD",
        policy_id="open",
        license_id="cc-by-4.0",
        source_url="https://zenodo.org/records/4405783",
        requires_manual_download=False,
    ),
    "jl-corpus": DatasetDescriptor(
        dataset_id="jl-corpus",
        display_name="JL-Corpus",
        policy_id="open",
        license_id="cc0-1.0",
        source_url="https://www.kaggle.com/datasets/tli725/jl-corpus",
        requires_manual_download=False,
    ),
    "cafe": DatasetDescriptor(
        dataset_id="cafe",
        display_name="CaFE",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://zenodo.org/records/1478765",
        requires_manual_download=False,
    ),
    "asvp-esd": DatasetDescriptor(
        dataset_id="asvp-esd",
        display_name="ASVP-ESD",
        policy_id="open",
        license_id="cc-by-4.0",
        source_url="https://zenodo.org/records/7132783",
        requires_manual_download=False,
    ),
    "emov-db": DatasetDescriptor(
        dataset_id="emov-db",
        display_name="EmoV-DB",
        policy_id="noncommercial",
        license_id="custom-noncommercial",
        source_url="https://www.openslr.org/115/",
        requires_manual_download=False,
    ),
    "pavoque": DatasetDescriptor(
        dataset_id="pavoque",
        display_name="PAVOQUE",
        policy_id="noncommercial",
        license_id="cc-by-nc-sa-4.0",
        source_url="https://github.com/marytts/pavoque-data/releases",
        requires_manual_download=False,
    ),
    "att-hack": DatasetDescriptor(
        dataset_id="att-hack",
        display_name="Att-HACK",
        policy_id="noncommercial",
        license_id="cc-by-nc-nd-4.0",
        source_url="https://www.openslr.org/88/",
        requires_manual_download=False,
    ),
    "coraa-ser": DatasetDescriptor(
        dataset_id="coraa-ser",
        display_name="CORAA SER",
        policy_id="research_only",
        license_id="custom-research-only",
        source_url="https://github.com/rmarcacini/ser-coraa-pt-br",
        requires_manual_download=False,
    ),
    "spanish-meacorpus-2023": DatasetDescriptor(
        dataset_id="spanish-meacorpus-2023",
        display_name="Spanish MEACorpus 2023",
        policy_id="noncommercial",
        license_id="cc-by-nc-4.0",
        source_url="https://zenodo.org/records/18606423",
        requires_manual_download=False,
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


def _normalize_source_overrides(
    *,
    source_repo_id: str | None,
    source_revision: str | None,
) -> tuple[str | None, str | None]:
    normalized_source_repo_id = source_repo_id.strip() if source_repo_id is not None else None
    normalized_source_revision = source_revision.strip() if source_revision is not None else None
    if normalized_source_repo_id == "":
        normalized_source_repo_id = None
    if normalized_source_revision == "":
        normalized_source_revision = None
    if normalized_source_repo_id is not None:
        if "/" not in normalized_source_repo_id or any(
            char.isspace() for char in normalized_source_repo_id
        ):
            raise ValueError(
                "Invalid --source value. Expected Hugging Face dataset id like " "`namespace/name`."
            )
    if normalized_source_revision is not None and any(
        char.isspace() for char in normalized_source_revision
    ):
        raise ValueError("Invalid --source-revision value: whitespace is not allowed.")
    return normalized_source_repo_id, normalized_source_revision


def _build_registry_options(
    *,
    labels_csv_path: Path | None,
    audio_base_dir: Path | None,
    source_repo_id: str | None,
    source_revision: str | None,
    source_commit_sha: str | None,
    default_language: str | None,
) -> dict[str, str]:
    options: dict[str, str] = {}
    if labels_csv_path is not None:
        options["labels_csv_path"] = str(labels_csv_path)
    if audio_base_dir is not None:
        options["audio_base_dir"] = str(audio_base_dir)
    if source_repo_id is not None:
        options["source_repo_id"] = source_repo_id
    if source_revision is not None:
        options["source_revision"] = source_revision
    if source_commit_sha is not None:
        options["source_commit_sha"] = source_commit_sha
    if default_language is not None:
        options["default_language"] = default_language
    return options


def _run_dataset_strategy_phase[T](
    *,
    dataset_id: str,
    strategy: DatasetStrategy,
    phase: str,
    action: Callable[[], T],
) -> T:
    """Runs one strategy phase with structured success/failure telemetry."""

    started_at = perf_counter()
    strategy_name = type(strategy).__name__
    try:
        result = action()
    except Exception as err:
        duration_ms = (perf_counter() - started_at) * 1000.0
        logger.error(
            "dataset_strategy phase=%s outcome=failure dataset_id=%s strategy=%s duration_ms=%.2f error=%s",
            phase,
            dataset_id,
            strategy_name,
            duration_ms,
            type(err).__name__,
        )
        raise
    duration_ms = (perf_counter() - started_at) * 1000.0
    logger.info(
        "dataset_strategy phase=%s outcome=success dataset_id=%s strategy=%s duration_ms=%.2f",
        phase,
        dataset_id,
        strategy_name,
        duration_ms,
    )
    return result


def _build_dataset_strategy_registry(
    *,
    strategies: Mapping[str, DatasetStrategy] | None = None,
    supported_dataset_ids: Collection[str] | None = None,
) -> DatasetStrategyRegistry:
    """Builds one validated strategy registry with contextual integrity errors."""
    resolved_strategies = (
        strategies if strategies is not None else build_default_dataset_strategies()
    )
    resolved_supported_dataset_ids = (
        tuple(supported_dataset_ids)
        if supported_dataset_ids is not None
        else tuple(SUPPORTED_DATASETS.keys())
    )
    try:
        return DatasetStrategyRegistry.from_mapping(
            strategies=resolved_strategies,
            supported_dataset_ids=resolved_supported_dataset_ids,
        )
    except ValueError as err:
        raise ValueError("Dataset strategy registry initialization failed. " f"{err}") from err


_DATASET_STRATEGY_REGISTRY = _build_dataset_strategy_registry()


def _resolve_dataset_strategy(dataset_id: str) -> DatasetStrategy:
    try:
        return _DATASET_STRATEGY_REGISTRY.resolve(dataset_id)
    except ValueError as err:
        raise ValueError(
            "Dataset strategy resolution failed for " f"dataset_id={dataset_id!r}. {err}"
        ) from err


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
    source_repo_id: str | None = None,
    source_revision: str | None = None,
) -> tuple[str | None, str | None]:
    """Best-effort dataset download.

    Some corpora require manual access requests; this function will not bypass
    those requirements.

    Returns:
        Tuple ``(resolved_source_repo_id, resolved_source_revision)`` used for
        acquisition, when applicable (currently MSP-Podcast only). For datasets
        without source pinning, returns ``(None, None)``.
    """

    descriptor = resolve_dataset_descriptor(dataset_id)
    strategy = _resolve_dataset_strategy(descriptor.dataset_id)
    dataset_root.mkdir(parents=True, exist_ok=True)
    normalized_source_repo_id, normalized_source_revision = _normalize_source_overrides(
        source_repo_id=source_repo_id,
        source_revision=source_revision,
    )
    if not strategy.supports_source_overrides and (
        normalized_source_repo_id is not None or normalized_source_revision is not None
    ):
        raise ValueError(
            "Download source overrides are currently supported only for `msp-podcast`."
        )
    return _run_dataset_strategy_phase(
        dataset_id=descriptor.dataset_id,
        strategy=strategy,
        phase="download",
        action=lambda: strategy.download(
            descriptor=descriptor,
            dataset_root=dataset_root,
            source_repo_id=normalized_source_repo_id,
            source_revision=normalized_source_revision,
        ),
    )


def prepare_dataset_manifest(
    *,
    settings: AppConfig,
    dataset_id: str,
    dataset_root: Path,
    ontology: LabelOntology,
    manifest_path: Path,
    labels_csv_path: Path | None = None,
    audio_base_dir: Path | None = None,
    source_repo_id: str | None = None,
    source_revision: str | None = None,
    source_commit_sha: str | None = None,
    default_language: str | None = None,
) -> list[Path]:
    """Builds a dataset manifest and updates the registry.

    Returns:
        A list of manifest paths written.
    """

    descriptor = resolve_dataset_descriptor(dataset_id)
    strategy = _resolve_dataset_strategy(descriptor.dataset_id)
    language = default_language or settings.default_language
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_source_repo_id = source_repo_id
    resolved_source_revision = source_revision
    resolved_source_commit_sha = source_commit_sha
    if descriptor.dataset_id == "msp-podcast":
        manifest_source = _read_msp_mirror_source_provenance(dataset_root=dataset_root)
        if manifest_source is not None:
            manifest_repo_id, manifest_revision, manifest_commit_sha = manifest_source
            if resolved_source_repo_id is None:
                resolved_source_repo_id = manifest_repo_id
            if resolved_source_revision is None:
                resolved_source_revision = manifest_revision
            if resolved_source_commit_sha is None:
                resolved_source_commit_sha = manifest_commit_sha
    options = _build_registry_options(
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        source_repo_id=resolved_source_repo_id,
        source_revision=resolved_source_revision,
        source_commit_sha=resolved_source_commit_sha,
        default_language=default_language,
    )
    result = _run_dataset_strategy_phase(
        dataset_id=descriptor.dataset_id,
        strategy=strategy,
        phase="prepare_manifest",
        action=lambda: strategy.prepare_manifest(
            settings=settings,
            descriptor=descriptor,
            dataset_root=dataset_root,
            ontology=ontology,
            manifest_path=manifest_path,
            language=language,
            labels_csv_path=labels_csv_path,
            audio_base_dir=audio_base_dir,
            options=options,
        ),
    )
    if not result.manifest_paths:
        return []

    upsert_dataset_registry_entry(
        settings=settings,
        dataset_id=descriptor.dataset_id,
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        options=result.options,
    )
    return list(result.manifest_paths)


def prepare_from_registry_entry(
    *,
    settings: AppConfig,
    entry: DatasetRegistryEntry,
    ontology: LabelOntology,
) -> list[Path]:
    """Ensures a registered dataset has a manifest on disk."""

    if entry.manifest_path.is_file():
        return [entry.manifest_path]
    parsed_options = parse_dataset_registry_options(entry.options)
    _validate_registry_source_provenance(
        dataset_id=entry.dataset_id,
        dataset_root=entry.dataset_root,
        source_repo_id=parsed_options.source_repo_id,
        source_revision=parsed_options.source_revision,
        source_commit_sha=parsed_options.source_commit_sha,
    )
    _validate_registry_msp_mirror_artifacts(
        dataset_id=entry.dataset_id,
        dataset_root=entry.dataset_root,
        labels_csv_path=parsed_options.labels_csv_path,
        audio_base_dir=parsed_options.audio_base_dir,
    )
    return prepare_dataset_manifest(
        settings=settings,
        dataset_id=entry.dataset_id,
        dataset_root=entry.dataset_root,
        ontology=ontology,
        manifest_path=entry.manifest_path,
        labels_csv_path=(
            Path(parsed_options.labels_csv_path).expanduser()
            if parsed_options.labels_csv_path
            else None
        ),
        audio_base_dir=(
            Path(parsed_options.audio_base_dir).expanduser()
            if parsed_options.audio_base_dir
            else None
        ),
        source_repo_id=parsed_options.source_repo_id,
        source_revision=parsed_options.source_revision,
        source_commit_sha=parsed_options.source_commit_sha,
        default_language=parsed_options.default_language,
    )


def collect_dataset_registry_health_issues(
    *,
    settings: AppConfig,
) -> tuple[DatasetRegistryHealthIssue, ...]:
    """Collects deterministic dataset registry health issues."""

    registry = load_dataset_registry(settings=settings)
    issues: list[DatasetRegistryHealthIssue] = []
    for entry in sorted(registry.values(), key=lambda item: item.dataset_id):
        try:
            parsed_options = parse_dataset_registry_options(entry.options)
        except ValueError as err:
            issues.append(
                DatasetRegistryHealthIssue(
                    dataset_id=entry.dataset_id,
                    code="registry_options_invalid",
                    message=str(err),
                )
            )
            continue
        try:
            _validate_registry_source_provenance(
                dataset_id=entry.dataset_id,
                dataset_root=entry.dataset_root,
                source_repo_id=parsed_options.source_repo_id,
                source_revision=parsed_options.source_revision,
                source_commit_sha=parsed_options.source_commit_sha,
            )
        except ValueError as err:
            issues.append(
                DatasetRegistryHealthIssue(
                    dataset_id=entry.dataset_id,
                    code="source_provenance_mismatch",
                    message=str(err),
                )
            )
            continue
        try:
            _validate_registry_msp_mirror_artifacts(
                dataset_id=entry.dataset_id,
                dataset_root=entry.dataset_root,
                labels_csv_path=parsed_options.labels_csv_path,
                audio_base_dir=parsed_options.audio_base_dir,
            )
        except ValueError as err:
            issues.append(
                DatasetRegistryHealthIssue(
                    dataset_id=entry.dataset_id,
                    code="mirror_artifact_mismatch",
                    message=str(err),
                )
            )
    return tuple(issues)


def _validate_registry_source_provenance(
    *,
    dataset_id: str,
    dataset_root: Path,
    source_repo_id: str | None,
    source_revision: str | None,
    source_commit_sha: str | None,
) -> None:
    """Validates registry source pin against local MSP mirror provenance."""

    if dataset_id != "msp-podcast":
        return
    has_registry_source_pin = any(
        item is not None for item in (source_repo_id, source_revision, source_commit_sha)
    )
    manifest_source = _read_msp_mirror_source_provenance(dataset_root=dataset_root)
    if manifest_source is None:
        if has_registry_source_pin:
            raise ValueError(
                "MSP source provenance mismatch: mirror manifest "
                f"{dataset_root / DEFAULT_MSP_MIRROR_MANIFEST_FILE} is missing; "
                "cannot validate persisted source pin."
            )
        return

    manifest_repo_id, manifest_revision, manifest_commit_sha = manifest_source
    has_manifest_source_pin = any(
        item is not None for item in (manifest_repo_id, manifest_revision, manifest_commit_sha)
    )
    if not has_registry_source_pin:
        if has_manifest_source_pin:
            raise ValueError(
                "MSP source provenance mismatch: registry options are missing source pin "
                "while local mirror manifest contains one. Re-run "
                "`ser data download --dataset msp-podcast` to persist source provenance."
            )
        return

    if source_repo_id is not None and source_repo_id != manifest_repo_id:
        raise ValueError(
            "MSP source provenance mismatch: registry source_repo_id "
            f"{source_repo_id!r} does not match mirror manifest {manifest_repo_id!r}. "
            "Re-run `ser data download --dataset msp-podcast` to reconcile provenance."
        )
    if source_revision is not None and source_revision != manifest_revision:
        raise ValueError(
            "MSP source provenance mismatch: registry source_revision "
            f"{source_revision!r} does not match mirror manifest {manifest_revision!r}. "
            "Re-run `ser data download --dataset msp-podcast` to reconcile provenance."
        )
    if source_commit_sha is not None and source_commit_sha != manifest_commit_sha:
        raise ValueError(
            "MSP source provenance mismatch: registry source_commit_sha "
            f"{source_commit_sha!r} does not match mirror manifest {manifest_commit_sha!r}. "
            "Re-run `ser data download --dataset msp-podcast` to reconcile provenance."
        )


def _read_msp_mirror_source_provenance(
    *,
    dataset_root: Path,
) -> tuple[str | None, str | None, str | None] | None:
    """Reads repo/revision/commit source provenance from MSP mirror manifest."""

    parsed_manifest = _read_msp_mirror_manifest_payload(dataset_root=dataset_root)
    if parsed_manifest is None:
        return None
    source = parsed_manifest.get("source")
    if not isinstance(source, dict):
        return (None, None, None)

    def _normalized_source_str(key: str) -> str | None:
        raw = source.get(key)
        if not isinstance(raw, str):
            return None
        stripped = raw.strip()
        return stripped or None

    return (
        _normalized_source_str("repo_id"),
        _normalized_source_str("revision"),
        _normalized_source_str("commit_sha"),
    )


def _validate_registry_msp_mirror_artifacts(
    *,
    dataset_id: str,
    dataset_root: Path,
    labels_csv_path: str | None,
    audio_base_dir: str | None,
) -> None:
    """Validates MSP mirror artifacts consistency for registry-driven rebuilds."""

    if dataset_id != "msp-podcast":
        return
    parsed_manifest = _read_msp_mirror_manifest_payload(dataset_root=dataset_root)
    if parsed_manifest is None:
        return
    registry_labels_path: Path | None = None
    if labels_csv_path is not None:
        registry_labels_path = Path(labels_csv_path).expanduser()
        if not registry_labels_path.is_file():
            raise ValueError(
                "MSP mirror artifact mismatch: registry labels_csv_path "
                f"{registry_labels_path} is missing; re-run "
                "`ser data download --dataset msp-podcast`."
            )
    registry_audio_dir: Path | None = None
    if audio_base_dir is not None:
        registry_audio_dir = Path(audio_base_dir).expanduser()
        if not registry_audio_dir.is_dir():
            raise ValueError(
                "MSP mirror artifact mismatch: registry audio_base_dir "
                f"{registry_audio_dir} is missing; re-run "
                "`ser data download --dataset msp-podcast`."
            )

    artifacts = parsed_manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return
    manifest_labels_path = _read_manifest_artifact_path(
        artifacts=artifacts,
        key="labels_csv",
        dataset_root=dataset_root,
    )
    manifest_audio_dir = _read_manifest_artifact_path(
        artifacts=artifacts,
        key="audio_dir",
        dataset_root=dataset_root,
    )
    if (
        registry_labels_path is not None
        and manifest_labels_path is not None
        and not _paths_equivalent(registry_labels_path, manifest_labels_path)
    ):
        raise ValueError(
            "MSP mirror artifact mismatch: registry labels_csv_path "
            f"{registry_labels_path} does not match mirror manifest "
            f"{manifest_labels_path}; re-run `ser data download --dataset msp-podcast`."
        )
    if (
        registry_audio_dir is not None
        and manifest_audio_dir is not None
        and not _paths_equivalent(registry_audio_dir, manifest_audio_dir)
    ):
        raise ValueError(
            "MSP mirror artifact mismatch: registry audio_base_dir "
            f"{registry_audio_dir} does not match mirror manifest "
            f"{manifest_audio_dir}; re-run `ser data download --dataset msp-podcast`."
        )

    if manifest_labels_path is None:
        return
    if not manifest_labels_path.is_file():
        raise ValueError(
            "MSP mirror artifact mismatch: mirror manifest labels_csv "
            f"{manifest_labels_path} is missing; re-run "
            "`ser data download --dataset msp-podcast`."
        )
    stats = parsed_manifest.get("stats")
    if not isinstance(stats, dict):
        return
    labels_written = stats.get("labels_written")
    if isinstance(labels_written, int):
        labels_rows = _count_labels_csv_rows(labels_csv_path=manifest_labels_path)
        if labels_rows != labels_written:
            raise ValueError(
                "MSP mirror artifact mismatch: labels.csv row count "
                f"{labels_rows} does not match mirror manifest labels_written "
                f"{labels_written}; re-run `ser data download --dataset msp-podcast`."
            )


def _read_msp_mirror_manifest_payload(
    *,
    dataset_root: Path,
) -> dict[str, object] | None:
    """Reads and validates local MSP mirror manifest payload."""

    manifest_path = dataset_root / DEFAULT_MSP_MIRROR_MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    try:
        parsed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Invalid MSP mirror manifest JSON at {manifest_path}: {err.msg}."
        ) from err
    if not isinstance(parsed_manifest, dict):
        raise ValueError(f"Invalid MSP mirror manifest structure at {manifest_path}.")
    return parsed_manifest


def _read_manifest_artifact_path(
    *,
    artifacts: dict[str, object],
    key: str,
    dataset_root: Path,
) -> Path | None:
    """Resolves one manifest artifact path entry."""

    raw_path = artifacts.get(key)
    if not isinstance(raw_path, str):
        return None
    stripped = raw_path.strip()
    if not stripped:
        return None
    parsed_path = Path(stripped).expanduser()
    if parsed_path.is_absolute():
        return parsed_path
    return (dataset_root / parsed_path).expanduser()


def _paths_equivalent(left: Path, right: Path) -> bool:
    """Returns whether two paths resolve to the same location."""

    return left.expanduser().resolve(strict=False) == right.expanduser().resolve(strict=False)


def _count_labels_csv_rows(*, labels_csv_path: Path) -> int:
    """Counts data rows in a labels CSV using header-aware parsing."""

    try:
        with labels_csv_path.open("r", encoding="utf-8", newline="") as labels_fp:
            reader = csv.DictReader(labels_fp)
            return sum(1 for _ in reader)
    except csv.Error as err:
        raise ValueError(f"Invalid MSP labels CSV at {labels_csv_path}: {err}.") from err
