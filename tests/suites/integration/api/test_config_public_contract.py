"""Contract tests for the stable `ser.config` facade."""

from __future__ import annotations

import ast
from pathlib import Path

import ser.config as config

_REMOVED_LEGACY_EXPORTS = {
    "ProfileCatalogEntry",
    "TranscriptionBackendId",
    "apply_settings",
    "default_profile_model_id",
    "get_profile_catalog",
    "normalize_torch_device_selector",
    "profile_artifact_file_names",
    "resolve_profile_transcription_config",
}


def test_config_public_surface_snapshot_matches_expected_contract() -> None:
    """Stable config facade snapshot should only change through explicit review."""
    assert config.__all__ == [
        "APP_NAME",
        "DEFAULT_FAST_MODEL_FILE_NAME",
        "DEFAULT_FAST_SECURE_MODEL_FILE_NAME",
        "DEFAULT_FAST_TRAINING_REPORT_FILE_NAME",
        "AccurateResearchRuntimeConfig",
        "AccurateRuntimeConfig",
        "AppConfig",
        "ArtifactProfileName",
        "AudioReadConfig",
        "DataLoaderConfig",
        "DatasetConfig",
        "FastRuntimeConfig",
        "FeatureFlags",
        "FeatureRuntimeBackendOverride",
        "FeatureRuntimePolicyConfig",
        "MediumRuntimeConfig",
        "MediumTrainingConfig",
        "ModelsConfig",
        "NeuralNetConfig",
        "ProfileRuntimeConfig",
        "QualityGateConfig",
        "ResolvedSettingsInputs",
        "RuntimeFlags",
        "SchemaConfig",
        "SettingsInputDeps",
        "TimelineConfig",
        "TorchRuntimeConfig",
        "TrainingConfig",
        "TranscriptionConfig",
        "WhisperModelConfig",
        "get_settings",
        "reload_settings",
        "settings_override",
    ]


def test_config_does_not_expose_removed_legacy_exports() -> None:
    """Removed legacy names should stay absent from the public facade."""
    visible_names = set(dir(config))
    for export_name in _REMOVED_LEGACY_EXPORTS:
        assert export_name not in config.__all__
        assert not hasattr(config, export_name)
        assert export_name not in visible_names


def test_internal_modules_do_not_named_import_removed_legacy_config_exports(
    repo_root: Path,
) -> None:
    """Internal source should not import removed legacy config names by symbol."""
    package_root = repo_root / "ser"
    for source_path in package_root.rglob("*.py"):
        if source_path == package_root / "config.py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "ser.config":
                continue
            imported_names = {alias.name for alias in node.names}
            overlap = imported_names & _REMOVED_LEGACY_EXPORTS
            assert not overlap, (
                f"{source_path} imports removed legacy config names from ser.config: "
                f"{sorted(overlap)}"
            )
