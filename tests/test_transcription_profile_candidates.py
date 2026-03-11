"""Tests for internal transcription profile-candidate helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ser._internal.transcription import profile_candidates as helpers


@dataclass(frozen=True)
class _Candidate:
    name: str
    source_profile: str
    backend_id: str
    model_name: str
    use_demucs: bool
    use_vad: bool


def test_candidate_name_includes_runtime_flags() -> None:
    """Candidate name helper should include backend/model and toggle labels."""

    assert (
        helpers.candidate_name(
            source_profile="accurate",
            backend_id="stable_whisper",
            model_name="large-v3",
            use_demucs=True,
            use_vad=False,
        )
        == "accurate_stable_whisper_large-v3_demucs_no_vad"
    )


def test_build_profile_candidates_uses_resolved_defaults() -> None:
    """Candidate builder should resolve and materialize one candidate per profile."""

    captured: list[str] = []

    def _resolve(profile_name: str) -> tuple[str, str, bool, bool]:
        captured.append(profile_name)
        return ("stable_whisper", f"model-for-{profile_name}", True, False)

    candidates = helpers.build_profile_candidates(
        profiles=("medium", "fast"),
        resolve_profile_config=_resolve,
        candidate_factory=lambda name, source_profile, backend_id, model_name, use_demucs, use_vad: _Candidate(
            name=name,
            source_profile=source_profile,
            backend_id=backend_id,
            model_name=model_name,
            use_demucs=use_demucs,
            use_vad=use_vad,
        ),
    )

    assert captured == ["medium", "fast"]
    assert candidates == (
        _Candidate(
            name="medium_stable_whisper_model-for-medium_demucs_no_vad",
            source_profile="medium",
            backend_id="stable_whisper",
            model_name="model-for-medium",
            use_demucs=True,
            use_vad=False,
        ),
        _Candidate(
            name="fast_stable_whisper_model-for-fast_demucs_no_vad",
            source_profile="fast",
            backend_id="stable_whisper",
            model_name="model-for-fast",
            use_demucs=True,
            use_vad=False,
        ),
    )
