"""Contract tests for medium retry/fallback helper policies."""

from __future__ import annotations

import pytest

from ser.runtime import medium_retry_policy


def test_retry_delay_seconds_returns_zero_for_non_positive_base_delay() -> None:
    """Retry delay helper should short-circuit when base delay is disabled."""
    assert medium_retry_policy.retry_delay_seconds(base_delay=0.0, attempt=1) == 0.0
    assert medium_retry_policy.retry_delay_seconds(base_delay=-1.0, attempt=3) == 0.0


def test_retry_delay_seconds_applies_attempt_scaling_with_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry delay helper should include deterministic jitter over attempt scaling."""

    def _fixed_uniform(_start: float, _end: float) -> float:
        return 0.05

    monkeypatch.setattr("ser.runtime.retry_primitives.random.uniform", _fixed_uniform)

    assert medium_retry_policy.retry_delay_seconds(base_delay=0.5, attempt=2) == pytest.approx(1.05)


def test_should_retry_on_cpu_after_transient_failure_matches_signatures() -> None:
    """CPU fallback classifier should detect known transient MPS failure signatures."""
    assert medium_retry_policy.should_retry_on_cpu_after_transient_failure(
        RuntimeError("MPS backend out of memory")
    )
    assert medium_retry_policy.should_retry_on_cpu_after_transient_failure(
        RuntimeError("Input type (c10::Half) and bias type (float) should be the same")
    )
    assert medium_retry_policy.should_retry_on_cpu_after_transient_failure(
        RuntimeError("non-finite embeddings produced by backend")
    )
    assert not medium_retry_policy.should_retry_on_cpu_after_transient_failure(
        RuntimeError("temporary network hiccup")
    )


def test_summarize_transient_failure_handles_blank_and_truncates() -> None:
    """Transient failure summary should normalize empty and long messages."""
    assert (
        medium_retry_policy.summarize_transient_failure(RuntimeError("   "))
        == "transient backend failure"
    )

    long_message = "x" * 181
    summary = medium_retry_policy.summarize_transient_failure(RuntimeError(long_message))

    assert summary.endswith("...")
    assert len(summary) == 180


def test_is_dependency_error_matches_optional_dependency_patterns() -> None:
    """Dependency classifier should detect optional dependency diagnostics."""
    assert medium_retry_policy.is_dependency_error(
        RuntimeError("Backend requires optional dependencies to be installed.")
    )
    assert medium_retry_policy.is_dependency_error(
        RuntimeError("transformers not installed in current environment")
    )
    assert not medium_retry_policy.is_dependency_error(
        RuntimeError("backend compute failed unexpectedly")
    )
