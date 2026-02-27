"""Unit tests for transcription backend adapter factory behavior."""

from __future__ import annotations

import importlib
import sys


def test_factory_resolves_faster_adapter_without_importing_stable_backend() -> None:
    """Resolving faster adapter should not import stable backend or torch stack."""
    module_names = (
        "ser.transcript.backends.factory",
        "ser.transcript.backends.stable_whisper",
        "ser.transcript.backends.stable_whisper_mps_compat",
    )
    removed_modules = {
        module_name: sys.modules.pop(module_name, None) for module_name in module_names
    }
    try:
        factory = importlib.import_module("ser.transcript.backends.factory")
        reloaded_factory = importlib.reload(factory)

        assert "ser.transcript.backends.stable_whisper" not in sys.modules
        assert "ser.transcript.backends.stable_whisper_mps_compat" not in sys.modules

        adapter = reloaded_factory.resolve_transcription_backend_adapter(
            "faster_whisper"
        )

        assert adapter.backend_id == "faster_whisper"
        assert "ser.transcript.backends.stable_whisper" not in sys.modules
        assert "ser.transcript.backends.stable_whisper_mps_compat" not in sys.modules
    finally:
        for module_name, module in removed_modules.items():
            if module is not None:
                sys.modules[module_name] = module
