"""Model files: one read, one verified identity, one licence record (ADR 0001).

The safe, pickle-free format and its packager belong to brief P2-4, which registers
it in ``artifact_formats``. Legacy pickles are trusted plumbing only: they load only
on explicit opt-in, as baselines, and are refused before anything is unpickled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from ser.profiles import ProfileName
from ser.runtime.contracts import ModelFormat, ModelManifest, ModelProvenance, TrainingAudioForm


@dataclass(frozen=True)
class ArtifactMetadata:
    """What a format reads from a model file before the payload is deserialised.

    Safe artifacts must carry ``model_id`` and ``label_set_id``. ``model_revision``
    is the legacy envelope's revision; safe formats leave it ``None``.
    """

    profile: ProfileName
    backend_id: str
    model_id: str | None
    label_set_id: str | None
    labels: tuple[str, ...]
    training_audio: TrainingAudioForm | None
    provenance: ModelProvenance | None
    model_revision: str | None = None


@dataclass(frozen=True)
class OpenedArtifact:
    """Metadata plus the payload bytes that ``provenance.payload_sha256`` must match."""

    metadata: ArtifactMetadata
    payload: bytes


class ArtifactFormat(Protocol):
    """One on-disk model format."""

    @property
    def name(self) -> ModelFormat:
        """Format name recorded in the manifest."""
        ...

    def claims(self, data: bytes) -> bool:
        """Whether ``data`` is in this format; inspects magic bytes only."""
        ...

    def open(self, data: bytes) -> OpenedArtifact:
        """Splits ``data`` into metadata and payload."""
        ...


class LegacyPickleFormat:
    """Model envelopes written by ``ser --train`` before brief P2-4."""

    @property
    def name(self) -> ModelFormat:
        return "legacy-pickle"

    def claims(self, data: bytes) -> bool:
        raise NotImplementedError("Legacy pickle reading is not implemented yet (brief A1-2).")

    def open(self, data: bytes) -> OpenedArtifact:
        """Unpickles; call only after the caller opted in to a legacy baseline."""
        raise NotImplementedError("Legacy pickle reading is not implemented yet (brief A1-2).")


def artifact_formats() -> tuple[ArtifactFormat, ...]:
    """Formats analysis can read; P2-4 adds the safe format ahead of the legacy one."""
    raise NotImplementedError("Artifact formats are registered by briefs P2-4 and A1-2.")


def read_model_file(path: str | os.PathLike[str]) -> bytes:
    """The only read of a model file; its bytes are hashed and kept for restarts."""
    raise NotImplementedError("Model file reading is not implemented yet (brief A1-2).")


def open_model(
    data: bytes,
    *,
    profile: ProfileName,
    expected_sha256: str | None,
    allow_legacy_baseline: bool,
) -> tuple[ModelManifest, OpenedArtifact]:
    """Identifies and verifies model bytes, in this order.

    1. ``model_sha256`` is the SHA-256 of ``data``; it must equal ``expected_sha256``.
    2. A legacy pickle without ``allow_legacy_baseline`` is refused unopened.
    3. A safe artifact must carry provenance whose ``payload_sha256`` matches the
       payload, even with the opt-in.
    4. That provenance must be complete: ``encoder`` is a licence record or
       ``"none"``, at least one dataset is listed, and every record has a
       non-empty ``name``, ``licence`` and ``source`` and a known ``licence_class``.
       Anything less raises ``AnalysisModelLoadError``, even with the opt-in.
    5. A safe artifact must carry ``model_id`` and ``label_set_id``.
    6. The profile must match; training audio that is not canonical, or unknown,
       needs the opt-in and makes the model a baseline.
    7. Baseline ids are ``baseline:<name>@<model_sha256>``, where ``<name>`` is the
       artifact's ``model_id``, else its ``model_revision``, else
       ``<profile>-<backend_id>``. A legacy file without ``label_set_id`` gets
       ``labels:`` plus the SHA-256 of the sorted unique labels as compact JSON.

    ``model_sha256`` is the model's identity. A safe artifact's ``model_id`` is
    copied verbatim and is not promised unique: two different files may share it.
    """
    raise NotImplementedError("Model verification is not implemented yet (brief A1-2).")


__all__ = [
    "ArtifactFormat",
    "ArtifactMetadata",
    "LegacyPickleFormat",
    "OpenedArtifact",
    "artifact_formats",
    "open_model",
    "read_model_file",
]
