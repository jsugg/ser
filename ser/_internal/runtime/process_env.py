"""Scoped process-environment application helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager

from ser._internal.runtime.environment_plan import ProcessEnvDelta


@contextmanager
def temporary_process_env(
    delta: ProcessEnvDelta,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Iterator[None]:
    """Temporarily applies one explicit env delta and restores prior values."""
    target_environ = os.environ if environ is None else environ
    previous_values: dict[str, str | None] = {key: target_environ.get(key) for key in delta.values}
    for key, value in delta.values.items():
        target_environ[key] = value
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                target_environ.pop(key, None)
            else:
                target_environ[key] = previous_value


__all__ = ["temporary_process_env"]
