"""Logging helpers used across the SER package."""

from __future__ import annotations

import io
import logging
import os
import re
import warnings
from collections.abc import Iterator, Sequence
from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
    redirect_stderr,
    redirect_stdout,
)
from dataclasses import dataclass
from typing import Literal

_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_LOGGING_CONFIGURED = False


def _normalize_scope_value(value: str | None) -> str | None:
    """Normalizes one scope value for policy matching."""
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_scope_values(values: frozenset[str]) -> frozenset[str]:
    """Normalizes one frozenset of scope values for policy matching."""
    return frozenset(
        normalized
        for normalized in (_normalize_scope_value(item) for item in values)
        if normalized is not None
    )


@dataclass(frozen=True, slots=True)
class DependencyPolicyContext:
    """Execution scope used to determine whether a dependency policy applies."""

    backend_id: str | None = None
    phase_name: str | None = None
    op_tag: str | None = None

    def __post_init__(self) -> None:
        """Normalizes context values for robust matching."""
        object.__setattr__(self, "backend_id", _normalize_scope_value(self.backend_id))
        object.__setattr__(self, "phase_name", _normalize_scope_value(self.phase_name))
        object.__setattr__(self, "op_tag", _normalize_scope_value(self.op_tag))


@dataclass(frozen=True, slots=True)
class WarningPolicy:
    """Scoped warning policy applied during dependency execution contexts."""

    policy_id: str
    message_regex: str
    module_regex: str
    category: type[Warning]
    action: Literal["default", "error", "ignore"] = "ignore"
    backend_ids: frozenset[str] = frozenset()
    phase_names: frozenset[str] = frozenset()
    op_tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Validates and normalizes warning policy values."""
        normalized_policy_id = self.policy_id.strip()
        if not normalized_policy_id:
            raise ValueError("WarningPolicy policy_id must be non-empty.")
        if self.action not in {"default", "error", "ignore"}:
            raise ValueError(f"Unsupported warning action {self.action!r}.")
        if not self.message_regex.strip():
            raise ValueError("WarningPolicy message_regex must be non-empty.")
        if not self.module_regex.strip():
            raise ValueError("WarningPolicy module_regex must be non-empty.")
        re.compile(self.message_regex)
        re.compile(self.module_regex)
        if not issubclass(self.category, Warning):
            raise TypeError("WarningPolicy category must be a Warning subclass.")
        object.__setattr__(self, "policy_id", normalized_policy_id)
        object.__setattr__(
            self, "backend_ids", _normalize_scope_values(self.backend_ids)
        )
        object.__setattr__(
            self, "phase_names", _normalize_scope_values(self.phase_names)
        )
        object.__setattr__(self, "op_tags", _normalize_scope_values(self.op_tags))


@dataclass(frozen=True, slots=True)
class DependencyLogPolicy:
    """Scoped policy that targets noisy dependency log records and warnings."""

    logger_prefixes: frozenset[str]
    root_path_markers: frozenset[str] = frozenset()
    demote_from_level: int = logging.INFO
    demote_to_level: int = logging.DEBUG
    message_regex: str | None = None
    backend_ids: frozenset[str] = frozenset()
    phase_names: frozenset[str] = frozenset()
    op_tags: frozenset[str] = frozenset()
    warning_policies: tuple[WarningPolicy, ...] = ()

    def __post_init__(self) -> None:
        """Normalizes dependency logger prefixes, scope selectors, and warning rules."""
        normalized_prefixes = frozenset(
            prefix.strip().lower() for prefix in self.logger_prefixes if prefix.strip()
        )
        normalized_markers = frozenset(
            marker.replace("\\", "/").strip().lower()
            for marker in self.root_path_markers
            if marker.strip()
        )
        normalized_message_regex = self.message_regex
        if normalized_message_regex is not None:
            stripped_regex = normalized_message_regex.strip()
            normalized_message_regex = stripped_regex or None
            if normalized_message_regex is not None:
                re.compile(normalized_message_regex)
        object.__setattr__(self, "logger_prefixes", normalized_prefixes)
        object.__setattr__(self, "root_path_markers", normalized_markers)
        object.__setattr__(self, "message_regex", normalized_message_regex)
        object.__setattr__(
            self, "backend_ids", _normalize_scope_values(self.backend_ids)
        )
        object.__setattr__(
            self, "phase_names", _normalize_scope_values(self.phase_names)
        )
        object.__setattr__(self, "op_tags", _normalize_scope_values(self.op_tags))
        object.__setattr__(self, "warning_policies", tuple(self.warning_policies))


def _policy_matches_context(
    *,
    backend_ids: frozenset[str],
    phase_names: frozenset[str],
    op_tags: frozenset[str],
    context: DependencyPolicyContext | None,
) -> bool:
    """Returns whether scoped policy selectors match the active context."""
    if not backend_ids and not phase_names and not op_tags:
        return True
    if context is None:
        return False
    if backend_ids and context.backend_id not in backend_ids:
        return False
    if phase_names and context.phase_name not in phase_names:
        return False
    if op_tags and context.op_tag not in op_tags:
        return False
    return True


def _record_matches_policy(
    record: logging.LogRecord, policy: DependencyLogPolicy
) -> bool:
    """Checks whether a log record matches one dependency policy."""
    if policy.message_regex is not None and not re.search(
        policy.message_regex,
        record.getMessage(),
    ):
        return False
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
        minimum_visible_level: int = logging.NOTSET,
    ) -> None:
        """Initializes filter behavior for one scoped dependency policy."""
        super().__init__()
        self._policy = policy
        self._keep_demoted = keep_demoted
        self._minimum_visible_level = minimum_visible_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Demotes matching records and optionally suppresses demoted output."""
        if (
            _record_matches_policy(record, self._policy)
            and record.levelno == self._policy.demote_from_level
        ):
            record.levelno = self._policy.demote_to_level
            record.levelname = logging.getLevelName(self._policy.demote_to_level)
            if not self._keep_demoted:
                return False
            return record.levelno >= self._minimum_visible_level
        return True


def _active_warning_policies(
    *,
    policy: DependencyLogPolicy,
    context: DependencyPolicyContext | None,
) -> tuple[WarningPolicy, ...]:
    """Returns warning policies that apply to the active execution context."""
    return tuple(
        warning_policy
        for warning_policy in policy.warning_policies
        if _policy_matches_context(
            backend_ids=warning_policy.backend_ids,
            phase_names=warning_policy.phase_names,
            op_tags=warning_policy.op_tags,
            context=context,
        )
    )


@contextmanager
def scoped_dependency_log_policy(
    *,
    policy: DependencyLogPolicy,
    context: DependencyPolicyContext | None = None,
    keep_demoted: bool = True,
    handlers: Sequence[logging.Handler] | None = None,
    capture_std_streams: bool = False,
    suppressed_output_logger: logging.Logger | None = None,
    suppressed_output_level: int = logging.DEBUG,
) -> Iterator[None]:
    """Applies one dependency policy for the current code path scope."""
    if not _policy_matches_context(
        backend_ids=policy.backend_ids,
        phase_names=policy.phase_names,
        op_tags=policy.op_tags,
        context=context,
    ):
        yield
        return

    active_handlers = (
        tuple(handlers) if handlers is not None else tuple(logging.getLogger().handlers)
    )
    dependency_filter = DependencyLogFilter(
        policy=policy,
        keep_demoted=keep_demoted,
        minimum_visible_level=logging.getLogger().getEffectiveLevel(),
    )
    for handler in active_handlers:
        handler.addFilter(dependency_filter)

    stdout_buffer: io.StringIO | None = None
    stderr_buffer: io.StringIO | None = None
    warning_policies = _active_warning_policies(policy=policy, context=context)
    warning_context: AbstractContextManager[object]
    warning_context = warnings.catch_warnings() if warning_policies else nullcontext()
    try:
        with warning_context:
            for warning_policy in warning_policies:
                warnings.filterwarnings(
                    warning_policy.action,
                    message=warning_policy.message_regex,
                    category=warning_policy.category,
                    module=warning_policy.module_regex,
                )
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
        formatter = logging.Formatter(_LOG_FORMAT)
        for handler in root_logger.handlers:
            handler.setLevel(resolved_level)
            handler.setFormatter(formatter)
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
