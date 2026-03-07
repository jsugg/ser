"""Regression contracts for transcription compatibility boundaries."""

from __future__ import annotations

import importlib

_COMPATIBILITY_BOUNDARY_SYMBOLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "ser.transcript.backends.stable_whisper",
        (
            "StableWhisperAdapter.load_model",
            "StableWhisperAdapter.transcribe",
            "StableWhisperAdapter._resolve_mps_admission_decision",
            "StableWhisperAdapter._should_enforce_transcribe_admission",
            "StableWhisperAdapter._mps_admission_control_enabled",
            "StableWhisperAdapter._mps_hard_oom_shortcut_enabled",
        ),
    ),
    (
        "ser.transcript.backends.stable_whisper_torio_probe",
        (
            "is_module_available",
            "detect_torio_ffmpeg_operational_issue",
            "probe_torio_ffmpeg_loader_error",
            "extract_missing_dynamic_library",
            "detect_default_torio_ffmpeg_operational_issue",
        ),
    ),
    (
        "ser.transcript.backends.stable_whisper_mps_compat",
        (
            "enable_stable_whisper_mps_compatibility",
            "move_model_to_mps_with_alignment_placeholder",
            "stable_whisper_mps_timing_compatibility_context",
            "set_stable_whisper_mps_compatibility_enabled",
            "is_stable_whisper_mps_compatibility_enabled",
            "set_stable_whisper_runtime_device",
            "get_stable_whisper_runtime_device",
        ),
    ),
    (
        "ser.transcript.mps_admission",
        (
            "decide_mps_admission_for_transcription",
            "capture_mps_pressure_snapshot",
        ),
    ),
    (
        "ser.transcript.mps_admission_overrides",
        ("apply_calibrated_mps_admission_override",),
    ),
)


def test_transcription_compatibility_boundary_symbols_are_unique_and_non_empty() -> (
    None
):
    """Compatibility boundary symbol inventory should stay non-overlapping."""
    assert _COMPATIBILITY_BOUNDARY_SYMBOLS

    owned_symbols = [
        (target_module, symbol)
        for target_module, symbols in _COMPATIBILITY_BOUNDARY_SYMBOLS
        for symbol in symbols
    ]
    assert len(owned_symbols) == len(set(owned_symbols))


def test_transcription_compatibility_boundary_symbols_exist() -> None:
    """Compatibility boundary symbols should resolve in their declared modules."""
    for target_module_name, symbols in _COMPATIBILITY_BOUNDARY_SYMBOLS:
        target_module = importlib.import_module(target_module_name)
        for symbol in symbols:
            resolved = target_module
            for part in symbol.split("."):
                assert hasattr(
                    resolved, part
                ), f"Missing boundary symbol {target_module_name}.{symbol}"
                resolved = getattr(resolved, part)
