"""Shared pytest fixtures for repository-aware test infrastructure."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("tests.fixtures.settings",)


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> Path:
    """Returns the resolved repository root discovered by pytest."""
    return Path(str(pytestconfig.rootpath)).resolve()
