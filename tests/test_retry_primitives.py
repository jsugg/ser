"""Tests for shared runtime retry/error primitives."""

from __future__ import annotations

import pytest

from ser.runtime import retry_primitives


def test_jittered_retry_delay_seconds_returns_zero_for_non_positive_base_delay() -> None:
    """Retry delay helper should short-circuit when backoff is disabled."""
    assert retry_primitives.jittered_retry_delay_seconds(base_delay=0.0, attempt=1) == 0.0
    assert retry_primitives.jittered_retry_delay_seconds(base_delay=-1.0, attempt=3) == 0.0


def test_jittered_retry_delay_seconds_applies_attempt_scaling_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry delay helper should apply deterministic jitter over linear backoff."""

    def _fixed_uniform(_start: float, _end: float) -> float:
        return 0.05

    monkeypatch.setattr(
        "ser.runtime.retry_primitives.random.uniform",
        _fixed_uniform,
    )

    assert retry_primitives.jittered_retry_delay_seconds(
        base_delay=0.5,
        attempt=2,
    ) == pytest.approx(1.05)


def test_is_optional_dependency_runtime_error_matches_expected_signatures() -> None:
    """Dependency classifier should detect common optional dependency messages."""
    assert retry_primitives.is_optional_dependency_runtime_error(
        RuntimeError("Backend requires optional dependencies to be installed.")
    )
    assert retry_primitives.is_optional_dependency_runtime_error(
        RuntimeError("transformers not installed in current environment")
    )
    assert not retry_primitives.is_optional_dependency_runtime_error(
        RuntimeError("backend compute failed unexpectedly")
    )
