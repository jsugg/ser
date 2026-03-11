"""Tests for medium retry-policy orchestration helper."""

from __future__ import annotations

import logging

import pytest

from ser.runtime.medium_retry_operation import run_medium_inference_with_retry_policy


def test_run_medium_inference_with_retry_policy_delegates() -> None:
    """Helper should delegate to the shared retry-policy runner."""
    captured: dict[str, object] = {}

    def _run_with_retry_policy(**kwargs: object) -> str:
        captured.update(kwargs)
        return "ok"

    result = run_medium_inference_with_retry_policy(
        operation=lambda: "never",
        runtime_config=object(),
        allow_retries=True,
        profile_label="Medium",
        timeout_error_type=TimeoutError,
        transient_error_type=RuntimeError,
        transient_exhausted_error=lambda err: RuntimeError(str(err)),
        retry_delay_seconds=lambda **_kwargs: 0.0,
        logger=logging.getLogger(__name__),
        on_transient_failure=lambda _err, _attempt, _failures: None,
        run_with_retry_policy=_run_with_retry_policy,
        passthrough_error_types=(ValueError,),
        runtime_error_factory=lambda err: RuntimeError(str(err)),
    )

    assert result == "ok"
    assert captured["allow_retries"] is True
    assert captured["profile_label"] == "Medium"
    assert callable(captured["operation"])
    assert callable(captured["on_transient_failure"])


def test_run_medium_inference_with_retry_policy_preserves_passthrough_errors() -> None:
    """Passthrough domain errors should not be remapped."""

    class _DomainError(RuntimeError):
        pass

    with pytest.raises(_DomainError, match="domain"):
        run_medium_inference_with_retry_policy(
            operation=lambda: "never",
            runtime_config=object(),
            allow_retries=True,
            profile_label="Medium",
            timeout_error_type=TimeoutError,
            transient_error_type=RuntimeError,
            transient_exhausted_error=lambda err: RuntimeError(str(err)),
            retry_delay_seconds=lambda **_kwargs: 0.0,
            logger=logging.getLogger(__name__),
            on_transient_failure=lambda _err, _attempt, _failures: None,
            run_with_retry_policy=lambda **_kwargs: (_ for _ in ()).throw(_DomainError("domain")),
            passthrough_error_types=(_DomainError,),
            runtime_error_factory=lambda err: RuntimeError(str(err)),
        )


def test_run_medium_inference_with_retry_policy_maps_runtime_errors() -> None:
    """Unexpected runtime errors should map through the provided factory."""
    with pytest.raises(RuntimeError, match="wrapped boom"):
        run_medium_inference_with_retry_policy(
            operation=lambda: "never",
            runtime_config=object(),
            allow_retries=True,
            profile_label="Medium",
            timeout_error_type=TimeoutError,
            transient_error_type=RuntimeError,
            transient_exhausted_error=lambda err: RuntimeError(str(err)),
            retry_delay_seconds=lambda **_kwargs: 0.0,
            logger=logging.getLogger(__name__),
            on_transient_failure=lambda _err, _attempt, _failures: None,
            run_with_retry_policy=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
            passthrough_error_types=(ValueError,),
            runtime_error_factory=lambda err: RuntimeError(f"wrapped {err}"),
        )
