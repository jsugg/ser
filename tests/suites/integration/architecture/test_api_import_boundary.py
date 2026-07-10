"""Import-boundary contract tests for the public API facade."""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.topology_contract

_TIER_ONE_SOURCE_FILES = frozenset(
    {
        "ser/__init__.py",
        "ser/api.py",
        "ser/config.py",
        "ser/domain.py",
        "ser/profiles.py",
        "ser/utils/__init__.py",
    }
)
_ALLOWED_NON_TIER_ONE_PUBLIC_FILES = frozenset(
    {
        "ser/__main__.py",
        "ser/diagnostics/__init__.py",
        "ser/diagnostics/domain.py",
        "ser/runtime/__init__.py",
        "ser/runtime/contracts.py",
        "ser/runtime/schema.py",
    }
)
_PACKAGE_MARKER_FILES = frozenset(
    {
        "ser/diagnostics/__init__.py",
        "ser/runtime/__init__.py",
    }
)


@dataclass(frozen=True)
class _BoundaryPolicyEntry:
    """One allowed public-to-internal dependency entry from boundary policy."""

    relative_path: str
    reason: str


def _load_boundary_policy_entries(repo_root: Path) -> tuple[_BoundaryPolicyEntry, ...]:
    """Loads the authoritative public-to-internal boundary policy entries."""
    policy_data = tomllib.loads((repo_root / "boundary_policy.toml").read_text(encoding="utf-8"))
    raw_entries = policy_data.get("public_internal_import")
    if not isinstance(raw_entries, list):
        raise AssertionError("boundary_policy.toml must define [[public_internal_import]] entries.")

    entries: list[_BoundaryPolicyEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must be a table, got {type(raw_entry)!r}."
            )
        raw_path = raw_entry.get("path")
        raw_reason = raw_entry.get("reason")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must define a non-empty path."
            )
        if not isinstance(raw_reason, str) or not raw_reason.strip():
            raise AssertionError(
                f"boundary_policy.toml entry #{index} must define a non-empty reason."
            )
        entries.append(
            _BoundaryPolicyEntry(
                relative_path=raw_path.strip(),
                reason=raw_reason.strip(),
            )
        )
    return tuple(entries)


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    """Returns one repository-relative path resolved from the root fixture."""
    return (repo_root / relative_path).resolve()


def test_boundary_policy_file_is_well_formed(repo_root: Path) -> None:
    """Boundary policy entries should be bounded, sorted, unique, and public."""
    entries = _load_boundary_policy_entries(repo_root)
    assert entries, "boundary_policy.toml must declare at least one allowed public module."
    assert len(entries) <= 10, "boundary_policy.toml must contain at most ten facade entries."

    relative_paths = [entry.relative_path for entry in entries]
    assert relative_paths == sorted(
        relative_paths
    ), "boundary_policy.toml entries must stay sorted by path for reviewability."
    assert len(set(relative_paths)) == len(
        relative_paths
    ), "boundary_policy.toml must not contain duplicate paths."
    for entry in entries:
        assert entry.relative_path.startswith("ser/")
        assert entry.relative_path.endswith(".py")
        assert "_internal" not in Path(entry.relative_path).parts
        assert _resolve_repo_path(
            repo_root, entry.relative_path
        ).exists(), f"Boundary policy path {entry.relative_path!r} does not exist."
        assert entry.reason


def test_no_first_party_module_imports_internal_api_directly(repo_root: Path) -> None:
    """All first-party imports should consume API callables through `ser.api`."""
    package_root = repo_root / "ser"
    blocked_patterns = (
        re.compile(r"^\s*from\s+ser\._internal\.api(?:\.|\s+)"),
        re.compile(r"^\s*import\s+ser\._internal\.api(?:\.|\s+|$)"),
    )
    allowed_files = {
        (package_root / "api.py").resolve(),
    }
    for source_path in package_root.rglob("*.py"):
        resolved_path = source_path.resolve()
        if resolved_path in allowed_files:
            continue
        if "_internal" in source_path.parts:
            continue
        source_text = source_path.read_text(encoding="utf-8")
        if any(pattern.search(source_text) for pattern in blocked_patterns):
            raise AssertionError(
                f"Internal API import boundary violation in {source_path}. "
                "Use `ser.api` for public imports."
            )


def test_public_python_tree_contains_only_tier_one_and_justified_contract_leaves(
    repo_root: Path,
) -> None:
    """Public package residue must stay limited to documented contract paths."""
    public_python_files = {
        source_path.relative_to(repo_root).as_posix()
        for source_path in (repo_root / "ser").rglob("*.py")
        if "_internal" not in source_path.relative_to(repo_root / "ser").parts
    }
    expected_files = _TIER_ONE_SOURCE_FILES | _ALLOWED_NON_TIER_ONE_PUBLIC_FILES
    assert public_python_files == expected_files, (
        "Public Python tree drifted. Move implementation under ser._internal or add a narrowly "
        "justified tier-one/contract-leaf rule with architecture review."
    )

    for relative_path in _PACKAGE_MARKER_FILES:
        tree = ast.parse((repo_root / relative_path).read_text(encoding="utf-8"))
        assert (
            len(tree.body) == 1
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ), f"{relative_path} must remain a package marker, not public implementation."


def test_public_to_internal_imports_match_explicit_allowlist(repo_root: Path) -> None:
    """Static and dynamic private imports should match the authoritative allowlist."""
    completed = subprocess.run(
        [sys.executable, "scripts/check_public_internal_imports.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_cli_main_uses_internal_cli_support_modules_for_runtime_policy(repo_root: Path) -> None:
    """CLI runtime policy should flow through internal CLI support modules only."""
    cli_main_source = (repo_root / "ser" / "__main__.py").read_text(encoding="utf-8")

    required_imports = (
        "from ser._internal.cli.data import run_configure_command, run_data_command",
        "from ser._internal.cli.diagnostics import (",
        "from ser._internal.cli.runtime import (",
        "run_restricted_backend_cli_gate(",
        "run_startup_preflight_cli_gate(",
        "run_transcription_runtime_calibration_command(",
        "run_training_command(",
        "run_inference_command(",
    )
    for required_import in required_imports:
        assert required_import in cli_main_source

    blocked_policy_calls = (
        "from ser.api import (",
        "prepare_restricted_backend_opt_in_state(",
        "enforce_restricted_backends_for_cli(",
        "run_training_workflow(",
        "run_inference_workflow(",
        "classify_training_exception(",
        "classify_inference_exception(",
    )
    for blocked_call in blocked_policy_calls:
        assert blocked_call not in cli_main_source


@pytest.mark.parametrize(
    "module_name",
    ("ser.api_data", "ser.api_runtime", "ser.api_diagnostics"),
)
def test_legacy_api_implementation_modules_are_not_publicly_importable(
    module_name: str,
) -> None:
    """Legacy top-level API implementation modules should stay removed."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    (
        "ser.data",
        "ser.data.application",
        "ser.features",
        "ser.heads",
        "ser.license_check",
        "ser.models",
        "ser.models.emotion_model",
        "ser.models.profile_runtime",
        "ser.models.training_entrypoints",
        "ser.pool",
        "ser.pool.stats_pool",
        "ser.repr",
        "ser.runtime.pipeline",
        "ser.runtime.registry",
        "ser.train",
        "ser.train.eval",
        "ser.transcript",
        "ser.transcript.backends.base",
        "ser.transcript.profiling",
        "ser.utils.common_utils",
        "ser.utils.logger",
    ),
)
def test_removed_public_implementation_paths_are_not_importable(module_name: str) -> None:
    """Moved implementation owners must not remain accidentally importable in public paths."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


_TIER_ONE_PUBLIC_EXPORTS: dict[str, tuple[str, ...]] = {
    "ser": (
        "EmotionSegment",
        "TimelineEntry",
        "TranscriptWord",
        "__version__",
    ),
    "ser.api": (
        "AccurateResearchRuntimeConfig",
        "AccurateRuntimeConfig",
        "AppConfig",
        "AudioReadConfig",
        "ComplianceMode",
        "DataLoaderConfig",
        "DatasetConfig",
        "DatasetConsents",
        "DatasetPrepareResult",
        "DatasetRegistryHealthIssueRecord",
        "DatasetRegistryRecord",
        "DiagnosticFinding",
        "DiagnosticReport",
        "DiagnosticSeverity",
        "EmotionSegment",
        "FastRuntimeConfig",
        "FeatureFlags",
        "FeatureRuntimeBackendOverride",
        "FeatureRuntimePolicyConfig",
        "FramePrediction",
        "InferenceExecution",
        "InferenceRequest",
        "InferenceResult",
        "MediumRuntimeConfig",
        "MediumTrainingConfig",
        "ModelsConfig",
        "NeuralNetConfig",
        "ProfileName",
        "QualityGateConfig",
        "RuntimeFlags",
        "RuntimePipeline",
        "RuntimePipelineBuilder",
        "SchemaConfig",
        "SegmentPrediction",
        "SubtitleFormat",
        "TimelineConfig",
        "TimelineEntry",
        "TorchRuntimeConfig",
        "TrainingConfig",
        "TranscriptWord",
        "TranscriptionConfig",
        "WhisperModelConfig",
        "configure_dataset_consents",
        "infer",
        "list_dataset_registry_health_issues",
        "list_datasets",
        "list_profiles",
        "list_registered_datasets",
        "load_profile",
        "prepare_dataset",
        "run_startup_preflight",
        "show_dataset_consents",
        "train",
    ),
    "ser.config": (
        "APP_NAME",
        "AccurateResearchRuntimeConfig",
        "AccurateRuntimeConfig",
        "AppConfig",
        "ArtifactProfileName",
        "AudioReadConfig",
        "DataLoaderConfig",
        "DatasetConfig",
        "FastRuntimeConfig",
        "FeatureFlags",
        "FeatureRuntimePolicyConfig",
        "MediumRuntimeConfig",
        "MediumTrainingConfig",
        "ModelsConfig",
        "NeuralNetConfig",
        "ProfileRuntimeConfig",
        "QualityGateConfig",
        "RuntimeFlags",
        "SchemaConfig",
        "TimelineConfig",
        "TorchRuntimeConfig",
        "TrainingConfig",
        "TranscriptionConfig",
        "WhisperModelConfig",
        "get_settings",
        "reload_settings",
        "settings_override",
    ),
    "ser.domain": (
        "DatasetConsents",
        "EmotionSegment",
        "TimelineEntry",
        "TranscriptWord",
    ),
    "ser.profiles": (
        "ProfileCatalogEntry",
        "ProfileEnableFlag",
        "ProfileFeatureRuntimeDefaults",
        "ProfileModelDefinition",
        "ProfileName",
        "ProfileRuntimeDefaults",
        "ProfileRuntimeEnvDefinition",
        "ProfileTranscriptionDefaults",
        "ProfileTranscriptionEnvDefinition",
        "RuntimeProfile",
        "TranscriptionBackendId",
        "available_profiles",
        "get_profile_catalog",
        "resolve_profile",
        "resolve_profile_name",
    ),
    "ser.utils": (
        "build_timeline",
        "display_elapsed_time",
        "get_logger",
        "print_timeline",
        "read_audio_file",
        "save_timeline_to_csv",
    ),
}


@pytest.mark.parametrize("module_name", sorted(_TIER_ONE_PUBLIC_EXPORTS))
def test_tier_one_public_exports_do_not_grow_without_contract_review(
    module_name: str,
) -> None:
    """Tier-1 `__all__` surfaces should never change without contract review."""
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{module_name} must declare an explicit __all__."
    assert len(set(exported)) == len(
        exported
    ), f"{module_name}.__all__ must not contain duplicates."
    assert tuple(sorted(exported)) == _TIER_ONE_PUBLIC_EXPORTS[module_name], (
        f"{module_name}.__all__ diverged from the tier-1 public contract. "
        "Shrinking or renaming requires updating this snapshot; growth requires "
        "explicit API review."
    )


def test_root_package_exposes_metadata_version() -> None:
    """Root package should expose the installed package version."""
    import ser

    assert ser.__version__ == "1.0.0"


def test_root_package_uses_exact_metadata_fallback_without_heavy_imports(repo_root: Path) -> None:
    """Source-tree imports should retain the documented metadata fallback."""
    script = """
import importlib.metadata
import sys


def _missing_distribution(_distribution_name: str) -> str:
    raise importlib.metadata.PackageNotFoundError


importlib.metadata.version = _missing_distribution
import ser

assert ser.__version__ == \"0.0.0.dev0\"
assert \"__version__\" in ser.__all__
assert \"torch\" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
