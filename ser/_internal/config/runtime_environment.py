"""Runtime environment synchronization for explicit configuration transitions."""

from __future__ import annotations

from collections.abc import MutableMapping


def sync_torch_runtime_environment(
    *,
    enable_mps_fallback: bool,
    environ: MutableMapping[str, str],
    pytorch_enable_mps_fallback_env: str,
) -> None:
    """Synchronizes torch runtime compatibility env vars from explicit settings."""
    environ[pytorch_enable_mps_fallback_env] = "1" if enable_mps_fallback else "0"
