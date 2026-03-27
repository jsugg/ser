"""Unit tests for runtime environment synchronization helpers."""

from __future__ import annotations

import pytest

from ser._internal.config.runtime_environment import sync_torch_runtime_environment

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("enable_mps_fallback", "expected"),
    [(True, "1"), (False, "0")],
)
def test_sync_torch_runtime_environment_sets_explicit_flag_value(
    enable_mps_fallback: bool,
    expected: str,
) -> None:
    """Torch runtime environment helper should encode the boolean explicitly."""
    environ: dict[str, str] = {"UNCHANGED": "value"}

    sync_torch_runtime_environment(
        enable_mps_fallback=enable_mps_fallback,
        environ=environ,
        pytorch_enable_mps_fallback_env="PYTORCH_ENABLE_MPS_FALLBACK",
    )

    assert environ == {
        "UNCHANGED": "value",
        "PYTORCH_ENABLE_MPS_FALLBACK": expected,
    }
