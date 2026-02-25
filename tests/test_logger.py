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


def test_configure_logging_sets_handler_level_during_first_initialization() -> None:
    """First-time basicConfig path should align handler level with resolved level."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_configured = logger_utils._LOGGING_CONFIGURED

    for handler in original_handlers:
        root_logger.removeHandler(handler)
    logger_utils._LOGGING_CONFIGURED = False

    try:
        applied_level = logger_utils.configure_logging("INFO")

        assert applied_level == logging.INFO
        assert root_logger.handlers
        assert all(handler.level == logging.INFO for handler in root_logger.handlers)
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
        logger_utils._LOGGING_CONFIGURED = original_configured


def test_scoped_dependency_log_policy_hides_demoted_logs_at_info_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Demoted dependency DEBUG entries should stay hidden in INFO-level runs."""
    dependency_logger = logging.getLogger("faster_whisper")
    policy = logger_utils.DependencyLogPolicy(logger_prefixes=frozenset({"faster_whisper"}))

    with caplog.at_level(logging.INFO):
        with logger_utils.scoped_dependency_log_policy(
            policy=policy,
            keep_demoted=True,
        ):
            dependency_logger.info("dependency info")

    assert "dependency info" not in caplog.text


def test_scoped_dependency_log_policy_keeps_demoted_logs_at_debug_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Demoted dependency entries should remain visible when DEBUG is enabled."""
    dependency_logger = logging.getLogger("faster_whisper")
    policy = logger_utils.DependencyLogPolicy(logger_prefixes=frozenset({"faster_whisper"}))

    with caplog.at_level(logging.DEBUG):
        with logger_utils.scoped_dependency_log_policy(
            policy=policy,
            keep_demoted=True,
        ):
            dependency_logger.info("dependency info")

    assert "dependency info" in caplog.text
