"""Shared fixtures for ambient settings state."""

from __future__ import annotations

from collections.abc import Generator

import pytest

import ser.config as config


@pytest.fixture
def reset_ambient_settings() -> Generator[None]:
    """Resets ambient settings before and after one test."""
    config.reload_settings()
    yield
    config.reload_settings()
