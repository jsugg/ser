"""Logging helpers used across the SER package."""

from __future__ import annotations

import io
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_LOGGING_CONFIGURED = False


@dataclass(frozen=True, slots=True)
class DependencyLogPolicy:
    """Scoped policy that targets noisy dependency log records."""

    logger_prefixes: frozenset[str]
    root_path_markers: frozenset[str] = frozenset()
    demote_from_level: int = logging.INFO
    demote_to_level: int = logging.DEBUG

    def __post_init__(self) -> None:
        """Normalizes dependency logger prefixes and root path markers."""
        normalized_prefixes = frozenset(
            prefix.strip().lower()
            for prefix in self.logger_prefixes
            if prefix.strip()
        )
        normalized_markers = frozenset(
            marker.replace("\\", "/").strip().lower()
            for marker in self.root_path_markers
            if marker.strip()
        )
        object.__setattr__(self, "logger_prefixes", normalized_prefixes)
        object.__setattr__(self, "root_path_markers", normalized_markers)


def _record_matches_policy(record: logging.LogRecord, policy: DependencyLogPolicy) -> bool:
    """Checks whether a log record matches one dependency policy."""
    logger_name = record.name.strip().lower()
    for prefix in policy.logger_prefixes:
        if logger_name == prefix or logger_name.startswith(f"{prefix}."):
            return True
    if logger_name != "root":
        return False
    if not policy.root_path_markers:
        return False
    normalized_path = record.pathname.replace("\\", "/").lower()
    return any(marker in normalized_path for marker in policy.root_path_markers)


def is_dependency_log_record(
    record: logging.LogRecord,
    *,
    policy: DependencyLogPolicy,
) -> bool:
    """Returns whether a record belongs to one dependency policy."""
    return _record_matches_policy(record, policy)


class DependencyLogFilter(logging.Filter):
    """Demotes matching dependency records and can drop demoted output."""

    def __init__(
        self,
        *,
        policy: DependencyLogPolicy,
        keep_demoted: bool = True,
    ) -> None:
        """Initializes filter behavior for one scoped dependency policy."""
        super().__init__()
        self._policy = policy
        self._keep_demoted = keep_demoted

    def filter(self, record: logging.LogRecord) -> bool:
        """Demotes matching records and optionally suppresses demoted output."""
        if (
            _record_matches_policy(record, self._policy)
            and record.levelno == self._policy.demote_from_level
        ):
            record.levelno = self._policy.demote_to_level
            record.levelname = logging.getLevelName(self._policy.demote_to_level)
            return self._keep_demoted
        return True


@contextmanager
def scoped_dependency_log_policy(
    *,
    policy: DependencyLogPolicy,
    keep_demoted: bool = True,
    handlers: Sequence[logging.Handler] | None = None,
    capture_std_streams: bool = False,
    suppressed_output_logger: logging.Logger | None = None,
    suppressed_output_level: int = logging.DEBUG,
) -> Iterator[None]:
    """Applies one dependency logging policy for the current code path scope."""
    active_handlers = (
        tuple(handlers) if handlers is not None else tuple(logging.getLogger().handlers)
    )
    dependency_filter = DependencyLogFilter(policy=policy, keep_demoted=keep_demoted)
    for handler in active_handlers:
        handler.addFilter(dependency_filter)

    stdout_buffer: io.StringIO | None = None
    stderr_buffer: io.StringIO | None = None
    try:
        if capture_std_streams:
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                yield
            return
        yield
    finally:
        for handler in active_handlers:
            handler.removeFilter(dependency_filter)
        if (
            suppressed_output_logger is not None
            and stdout_buffer is not None
            and stderr_buffer is not None
        ):
            for stream_name, stream_value in (
                ("stdout", stdout_buffer.getvalue()),
                ("stderr", stderr_buffer.getvalue()),
            ):
                for line in stream_value.splitlines():
                    message = line.strip()
                    if not message:
                        continue
                    suppressed_output_logger.log(
                        suppressed_output_level,
                        "Suppressed dependency %s line: %s",
                        stream_name,
                        message,
                    )


def _resolve_level(level: str | int | None = None) -> int:
    """Resolves one log-level value from explicit input or environment."""
    raw_level: str | int | None = level
    if raw_level is None:
        raw_level = os.getenv("LOG_LEVEL", "INFO")
    if isinstance(raw_level, int):
        return raw_level
    normalized = raw_level.strip()
    if not normalized:
        normalized = "INFO"
    if normalized.lstrip("+-").isdigit():
        return int(normalized)
    candidate = getattr(logging, normalized.upper(), None)
    if isinstance(candidate, int):
        return candidate
    return logging.INFO


def configure_logging(level: str | int | None = None) -> int:
    """Configures root logging and returns the applied numeric level."""
    global _LOGGING_CONFIGURED
    resolved_level = _resolve_level(level)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(format=_LOG_FORMAT, level=resolved_level)
        _LOGGING_CONFIGURED = True
        return resolved_level

    root_logger.setLevel(resolved_level)
    formatter = logging.Formatter(_LOG_FORMAT)
    for handler in root_logger.handlers:
        handler.setLevel(resolved_level)
        handler.setFormatter(formatter)
    _LOGGING_CONFIGURED = True
    return resolved_level


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Creates or retrieves one logger without repeated global reconfiguration."""
    if level is not None or not _LOGGING_CONFIGURED:
        configure_logging(level)
    return logging.getLogger(name)
