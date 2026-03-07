"""Tests for dataset strategy registry and contract behavior."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from ser.data import dataset_prepare
from ser.data.strategies import (
    DatasetStrategyRegistry,
    PreparedManifestResult,
    build_default_dataset_strategies,
)


def test_build_default_dataset_strategies_contains_expected_ids() -> None:
    """Default strategy factory should register every supported dataset id."""
    strategies = build_default_dataset_strategies()
    assert set(strategies) == {
        "ravdess",
        "crema-d",
        "msp-podcast",
        "emodb-2.0",
        "escorpus-pe",
        "mesd",
        "oreau-french-esd",
        "jl-corpus",
        "cafe",
        "asvp-esd",
        "emov-db",
        "pavoque",
        "att-hack",
        "coraa-ser",
        "spanish-meacorpus-2023",
        "biic-podcast",
    }


def test_default_strategies_expose_required_contract_members() -> None:
    """Each strategy instance should expose the expected contract members."""
    strategies = build_default_dataset_strategies()
    for strategy in strategies.values():
        assert isinstance(strategy.supports_source_overrides, bool)
        assert callable(strategy.download)
        assert callable(strategy.prepare_manifest)


def test_source_override_support_is_only_enabled_for_msp() -> None:
    """Only MSP strategy should support source override pins."""
    strategies = build_default_dataset_strategies()
    assert strategies["msp-podcast"].supports_source_overrides is True
    assert strategies["ravdess"].supports_source_overrides is False
    assert strategies["crema-d"].supports_source_overrides is False
    assert strategies["emodb-2.0"].supports_source_overrides is False
    assert strategies["escorpus-pe"].supports_source_overrides is False
    assert strategies["mesd"].supports_source_overrides is False
    assert strategies["oreau-french-esd"].supports_source_overrides is False
    assert strategies["jl-corpus"].supports_source_overrides is False
    assert strategies["cafe"].supports_source_overrides is False
    assert strategies["asvp-esd"].supports_source_overrides is False
    assert strategies["emov-db"].supports_source_overrides is False
    assert strategies["pavoque"].supports_source_overrides is False
    assert strategies["att-hack"].supports_source_overrides is False
    assert strategies["coraa-ser"].supports_source_overrides is False
    assert strategies["spanish-meacorpus-2023"].supports_source_overrides is False
    assert strategies["biic-podcast"].supports_source_overrides is False


def test_dataset_strategy_registry_rejects_missing_supported_ids() -> None:
    """Registry construction should fail when supported ids are missing mappings."""
    with pytest.raises(ValueError, match="Missing strategy ids"):
        DatasetStrategyRegistry.from_mapping(
            strategies={"ravdess": build_default_dataset_strategies()["ravdess"]},
            supported_dataset_ids=("ravdess", "msp-podcast"),
        )


def test_dataset_strategy_registry_rejects_unknown_extra_ids() -> None:
    """Registry construction should fail when unknown extra ids are provided."""
    strategies = build_default_dataset_strategies()
    with pytest.raises(ValueError, match="Unknown strategy ids"):
        DatasetStrategyRegistry.from_mapping(
            strategies=strategies,
            supported_dataset_ids=("ravdess",),
        )


def test_dataset_strategy_registry_resolve_returns_expected_strategy() -> None:
    """Registry resolution should return the registered strategy instance."""
    strategies = build_default_dataset_strategies()
    registry = DatasetStrategyRegistry.from_mapping(
        strategies=strategies,
        supported_dataset_ids=tuple(strategies),
    )

    strategy = registry.resolve("msp-podcast")

    assert strategy is strategies["msp-podcast"]


def test_supported_datasets_and_default_strategy_factory_stay_in_sync() -> None:
    """Supported dataset descriptors must match default strategy registration ids."""
    strategies = build_default_dataset_strategies()
    supported_ids = set(dataset_prepare.SUPPORTED_DATASETS)

    assert set(strategies) == supported_ids
    assert set(dataset_prepare._DATASET_STRATEGY_REGISTRY.as_mapping()) == supported_ids


def test_dataset_strategy_registry_reports_missing_and_unknown_ids_together() -> None:
    """Registry error should include both missing and unknown id diagnostics."""

    class _FakeStrategy:
        supports_source_overrides = False

        def download(
            self,
            *,
            descriptor: object,
            dataset_root: Path,
            source_repo_id: str | None,
            source_revision: str | None,
        ) -> tuple[str | None, str | None]:
            del descriptor, dataset_root, source_repo_id, source_revision
            return (None, None)

        def prepare_manifest(self, **kwargs: object) -> PreparedManifestResult:
            del kwargs
            return PreparedManifestResult(manifest_paths=(), options={})

    strategies = {
        "ravdess": build_default_dataset_strategies()["ravdess"],
        "oops": _FakeStrategy(),
    }
    with pytest.raises(
        ValueError,
        match=r"Missing strategy ids: msp-podcast\..*Unknown strategy ids: oops\.",
    ):
        DatasetStrategyRegistry.from_mapping(
            strategies=strategies,
            supported_dataset_ids=("ravdess", "msp-podcast"),
        )


def test_dataset_strategy_registry_as_mapping_is_immutable() -> None:
    """Registry mapping view should be immutable and read-only."""
    strategies = build_default_dataset_strategies()
    registry = DatasetStrategyRegistry.from_mapping(
        strategies=strategies,
        supported_dataset_ids=tuple(strategies),
    )

    mapping = registry.as_mapping()
    assert isinstance(mapping, MappingProxyType)
    with pytest.raises(TypeError):
        cast(dict[str, object], mapping)["new-id"] = strategies["ravdess"]
