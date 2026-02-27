"""Shared runtime retry policy for profile inference execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from ser.config import ProfileRuntimeConfig

type RetryDelaySeconds = Callable[..., float]
type TransientErrorFactory = Callable[[Exception], Exception]
type TransientFailureHook = Callable[[Exception, int, int], None]


def run_with_retry_policy[ResultT](
    *,
    operation: Callable[[], ResultT],
    runtime_config: ProfileRuntimeConfig,
    allow_retries: bool,
    profile_label: str,
    timeout_error_type: type[Exception],
    transient_error_type: type[Exception],
    transient_exhausted_error: TransientErrorFactory,
    retry_delay_seconds: RetryDelaySeconds,
    logger: logging.Logger,
    on_transient_failure: TransientFailureHook | None = None,
) -> ResultT:
    """Runs one profile operation with split retry budgets by error type."""
    timeout_failures = 0
    transient_failures = 0
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except timeout_error_type as err:
            timeout_failures += 1
            logger.warning(
                "%s inference timed out on attempt %s (timeout retry %s/%s): %s",
                profile_label,
                attempt,
                timeout_failures,
                runtime_config.max_timeout_retries + 1,
                err,
            )
            if (
                not allow_retries
                or timeout_failures > runtime_config.max_timeout_retries
            ):
                raise err
        except transient_error_type as err:
            transient_failures += 1
            logger.warning(
                "%s inference transient backend failure on attempt %s "
                "(transient retry %s/%s): %s",
                profile_label,
                attempt,
                transient_failures,
                runtime_config.max_transient_retries + 1,
                err,
            )
            should_retry = (
                allow_retries
                and transient_failures <= runtime_config.max_transient_retries
            )
            if should_retry and on_transient_failure is not None:
                on_transient_failure(err, attempt, transient_failures)
            if not should_retry:
                raise transient_exhausted_error(err) from err

        delay_seconds = retry_delay_seconds(
            base_delay=runtime_config.retry_backoff_seconds,
            attempt=attempt,
        )
        if delay_seconds > 0.0:
            time.sleep(delay_seconds)
