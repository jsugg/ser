"""Retry-policy orchestration helpers for medium runtime inference."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

_ResultT = TypeVar("_ResultT")


def run_medium_inference_with_retry_policy(
    *,
    operation: Callable[[], _ResultT],
    runtime_config: object,
    allow_retries: bool,
    profile_label: str,
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    transient_exhausted_error: Callable[[Exception], Exception],
    retry_delay_seconds: Callable[..., float],
    logger: logging.Logger,
    on_transient_failure: Callable[[Exception, int, int], None],
    run_with_retry_policy: Callable[..., _ResultT],
    passthrough_error_types: tuple[type[BaseException], ...],
    runtime_error_factory: Callable[[RuntimeError], Exception],
) -> _ResultT:
    """Runs medium retry-operation execution under retry policy handling."""

    try:
        return run_with_retry_policy(
            operation=operation,
            runtime_config=runtime_config,
            allow_retries=allow_retries,
            profile_label=profile_label,
            timeout_error_type=timeout_error_type,
            transient_error_type=transient_error_type,
            transient_exhausted_error=transient_exhausted_error,
            retry_delay_seconds=retry_delay_seconds,
            logger=logger,
            on_transient_failure=on_transient_failure,
        )
    except passthrough_error_types:
        raise
    except RuntimeError as err:
        raise runtime_error_factory(err) from err


__all__ = ["run_medium_inference_with_retry_policy"]
