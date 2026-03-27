"""Shared pytest fixtures and suite semantics for repository-aware test infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("tests.fixtures.settings",)

_UNIT_SUITE_PREFIX = ("tests", "suites", "unit")
_INTEGRATION_SUITE_PREFIX = ("tests", "suites", "integration")
_ARCHITECTURE_SUITE_PREFIX = ("tests", "suites", "integration", "architecture")
_SMOKE_SUITE_PREFIX = ("tests", "suites", "smoke")
_RESET_AMBIENT_SETTINGS_FIXTURE = "reset_ambient_settings"


def _matches_suite_prefix(path: Path, prefix: tuple[str, ...]) -> bool:
    """Returns whether a collected path belongs to the given suite prefix."""
    parts = path.parts
    return parts[: len(prefix)] == prefix


def _has_usefixture(item: pytest.Item, fixture_name: str) -> bool:
    """Returns whether a collected item already requests one fixture by name."""
    return any(fixture_name in marker.args for marker in item.iter_markers(name="usefixtures"))


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Applies suite markers from the tests/suites directory layout."""
    root_path = Path(str(config.rootpath)).resolve()
    for item in items:
        relative_path = item.path.resolve().relative_to(root_path)
        if _matches_suite_prefix(relative_path, _ARCHITECTURE_SUITE_PREFIX):
            if "integration" not in item.keywords:
                item.add_marker(pytest.mark.integration)
            if "topology_contract" not in item.keywords:
                item.add_marker(pytest.mark.topology_contract)
            if not _has_usefixture(item, _RESET_AMBIENT_SETTINGS_FIXTURE):
                item.add_marker(pytest.mark.usefixtures(_RESET_AMBIENT_SETTINGS_FIXTURE))
            continue
        if _matches_suite_prefix(relative_path, _INTEGRATION_SUITE_PREFIX):
            if "integration" not in item.keywords:
                item.add_marker(pytest.mark.integration)
            if not _has_usefixture(item, _RESET_AMBIENT_SETTINGS_FIXTURE):
                item.add_marker(pytest.mark.usefixtures(_RESET_AMBIENT_SETTINGS_FIXTURE))
            continue
        if _matches_suite_prefix(relative_path, _SMOKE_SUITE_PREFIX):
            if "smoke" not in item.keywords:
                item.add_marker(pytest.mark.smoke)
            if not _has_usefixture(item, _RESET_AMBIENT_SETTINGS_FIXTURE):
                item.add_marker(pytest.mark.usefixtures(_RESET_AMBIENT_SETTINGS_FIXTURE))
            continue
        if _matches_suite_prefix(relative_path, _UNIT_SUITE_PREFIX) and "unit" not in item.keywords:
            item.add_marker(pytest.mark.unit)


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> Path:
    """Returns the resolved repository root discovered by pytest."""
    return Path(str(pytestconfig.rootpath)).resolve()
