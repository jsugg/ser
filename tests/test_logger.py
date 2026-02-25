"""Unit tests for log-level resolution and logging configuration."""

import logging

import pytest

import ser.utils.logger as logger_utils


def test_configure_logging_defaults_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default configuration should resolve to INFO when LOG_LEVEL is unset."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)

    assert logger_utils.configure_logging() == logging.INFO


def test_configure_logging_reads_log_level_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment LOG_LEVEL should define the default when no explicit level exists."""
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    assert logger_utils.configure_logging() == logging.WARNING


def test_configure_logging_explicit_level_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit runtime level should override LOG_LEVEL."""
    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    assert logger_utils.configure_logging("DEBUG") == logging.DEBUG


def test_get_logger_does_not_reapply_env_after_explicit_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once configured, later logger retrieval should not reset level from environment."""
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    logger_utils.configure_logging("INFO")

    logger_utils.get_logger("ser.test")

    assert logging.getLogger().level == logging.INFO
