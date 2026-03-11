"""Profile-candidate assembly helpers for transcription profiling."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

CandidateT = TypeVar("CandidateT")
ProfileNameT = TypeVar("ProfileNameT", bound=str)
BackendIdT = TypeVar("BackendIdT", bound=str)


def candidate_name(
    *,
    source_profile: str,
    backend_id: str,
    model_name: str,
    use_demucs: bool,
    use_vad: bool,
) -> str:
    """Build one deterministic benchmark candidate identifier."""

    demucs_label = "demucs" if use_demucs else "no_demucs"
    vad_label = "vad" if use_vad else "no_vad"
    return f"{source_profile}_{backend_id}_{model_name}_{demucs_label}_{vad_label}"


def build_profile_candidates(
    *,
    profiles: tuple[ProfileNameT, ...],
    resolve_profile_config: Callable[[ProfileNameT], tuple[BackendIdT, str, bool, bool]],
    candidate_factory: Callable[
        [str, ProfileNameT, BackendIdT, str, bool, bool],
        CandidateT,
    ],
) -> tuple[CandidateT, ...]:
    """Build candidate descriptors from resolved transcription defaults."""

    return tuple(
        candidate_factory(
            candidate_name(
                source_profile=profile_name,
                backend_id=backend_id,
                model_name=model_name,
                use_demucs=use_demucs,
                use_vad=use_vad,
            ),
            profile_name,
            backend_id,
            model_name,
            use_demucs,
            use_vad,
        )
        for profile_name in profiles
        for backend_id, model_name, use_demucs, use_vad in (resolve_profile_config(profile_name),)
    )


__all__ = [
    "build_profile_candidates",
    "candidate_name",
]
