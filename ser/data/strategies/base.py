"""Strategy contracts for dataset-specific download and manifest operations."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from ser.config import AppConfig
from ser.data.ontology import LabelOntology

if TYPE_CHECKING:
    from ser.data.dataset_prepare import DatasetDescriptor


@dataclass(frozen=True, slots=True)
class PreparedManifestResult:
    """Internal manifest-preparation result for one dataset strategy."""

    manifest_paths: tuple[Path, ...]
    options: dict[str, str]


class DatasetStrategy(Protocol):
    """Contract for dataset-specific download and manifest preparation."""

    supports_source_overrides: bool

    def download(
        self,
        *,
        descriptor: DatasetDescriptor,
        dataset_root: Path,
        source_repo_id: str | None,
        source_revision: str | None,
    ) -> tuple[str | None, str | None]: ...

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
    ) -> PreparedManifestResult: ...


@dataclass(frozen=True, slots=True)
class DatasetStrategyRegistry:
    """Typed registry for dataset-id strategy mappings with invariant checks."""

    _strategies: Mapping[str, DatasetStrategy]

    @classmethod
    def from_mapping(
        cls,
        *,
        strategies: Mapping[str, DatasetStrategy],
        supported_dataset_ids: Collection[str],
    ) -> DatasetStrategyRegistry:
        """Builds one validated registry from a mapping."""

        expected_ids = set(supported_dataset_ids)
        registered_ids = set(strategies)
        missing_ids = sorted(expected_ids - registered_ids)
        extra_ids = sorted(registered_ids - expected_ids)
        if missing_ids or extra_ids:
            parts: list[str] = [
                "Invalid dataset strategy registration.",
            ]
            if missing_ids:
                parts.append(f"Missing strategy ids: {', '.join(missing_ids)}.")
            if extra_ids:
                parts.append(f"Unknown strategy ids: {', '.join(extra_ids)}.")
            raise ValueError(" ".join(parts))
        return cls(_strategies=MappingProxyType(dict(strategies)))

    def resolve(self, dataset_id: str) -> DatasetStrategy:
        """Resolves one registered strategy by dataset id."""

        strategy = self._strategies.get(dataset_id)
        if strategy is None:
            raise ValueError(f"No strategy registered for dataset {dataset_id!r}.")
        return strategy

    def as_mapping(self) -> Mapping[str, DatasetStrategy]:
        """Returns an immutable view of registered strategies."""

        return self._strategies
