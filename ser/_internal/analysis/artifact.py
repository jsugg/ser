"""Model files: one read, one verified identity, one licence record (ADR 0001).

The safe, pickle-free format and its packager belong to brief P2-4, which registers
it in ``artifact_formats``. Legacy pickles are trusted plumbing only: they load only
on explicit opt-in, as baselines, and are refused before anything is unpickled.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from ser._internal.analysis.errors import (
    AnalysisModelLoadError,
    AnalysisModelNotConformingError,
    AnalysisModelUnavailableError,
)
from ser.profiles import ProfileName
from ser.runtime.contracts import (
    LicenceRecord,
    ModelFormat,
    ModelIdentity,
    ModelManifest,
    ModelProvenance,
    TrainingAudioForm,
)


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
        # SER's historical writer uses pickle's binary protocol.
        return data.startswith(b"\x80")

    def open(self, data: bytes) -> OpenedArtifact:
        """Unpickles; call only after the caller opted in to a legacy baseline."""
        from ser._internal.models.artifact_envelope import deserialize_model_artifact

        model = deserialize_model_artifact(pickle.loads(data))
        metadata = model.artifact_metadata
        if metadata is None:
            raise ValueError("Missing legacy metadata.")
        return OpenedArtifact(
            ArtifactMetadata(
                profile=cast(ProfileName, metadata["profile"]),
                backend_id=cast(str, metadata["backend_id"]),
                model_id=None,
                label_set_id=None,
                labels=tuple(cast(list[str], metadata["labels"])),
                training_audio=None,
                provenance=None,
                model_revision=cast(str | None, metadata.get("model_revision")),
            ),
            data,
        )


def artifact_formats() -> tuple[ArtifactFormat, ...]:
    """Formats analysis can read; P2-4 adds the safe format ahead of the legacy one."""
    return (LegacyPickleFormat(),)


def read_model_file(path: str | os.PathLike[str]) -> bytes:
    """The only read of a model file; its bytes are hashed and kept for restarts."""
    try:
        return Path(path).read_bytes()
    except FileNotFoundError:
        raise AnalysisModelUnavailableError("No model file is available.") from None
    except (OSError, ValueError):
        raise AnalysisModelLoadError("The model file could not be read.") from None


_LICENCE_CLASSES = {"open", "non-commercial", "research-only", "academic-only"}


def _complete_licence(record: object) -> bool:
    return (
        isinstance(record, LicenceRecord)
        and all(
            isinstance(value, str) and value.strip()
            for value in (record.name, record.licence, record.source)
        )
        and record.licence_class in _LICENCE_CLASSES
    )


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
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise AnalysisModelLoadError("The model digest does not match.") from None

    artifact_format = next((fmt for fmt in artifact_formats() if fmt.claims(data)), None)
    if artifact_format is None:
        raise AnalysisModelLoadError("Unrecognised model format.") from None
    legacy = artifact_format.name == "legacy-pickle"
    if legacy and not allow_legacy_baseline:
        raise AnalysisModelNotConformingError(
            "The legacy model requires baseline opt-in."
        ) from None
    try:
        opened = artifact_format.open(data)
    except MemoryError:
        raise
    except Exception:
        raise AnalysisModelLoadError("The model artifact could not be opened.") from None

    metadata = opened.metadata
    provenance = metadata.provenance
    model_id = metadata.model_id
    label_set_id = metadata.label_set_id
    if not legacy:
        if (
            provenance is None
            or provenance.payload_sha256 != hashlib.sha256(opened.payload).hexdigest()
        ):
            raise AnalysisModelLoadError("The model payload lacks verified provenance.") from None
        if (
            (provenance.encoder != "none" and not _complete_licence(provenance.encoder))
            or not provenance.datasets
            or not all(map(_complete_licence, provenance.datasets))
        ):
            raise AnalysisModelLoadError("The model licence record is incomplete.") from None
        if not model_id or not label_set_id:
            raise AnalysisModelLoadError("The model or label set identifier is missing.") from None
    if metadata.profile != profile:
        raise AnalysisModelLoadError("The model does not match the requested profile.") from None
    baseline = legacy or metadata.training_audio != TrainingAudioForm(16_000, 1, True)
    if baseline:
        if not allow_legacy_baseline:
            raise AnalysisModelNotConformingError("The model requires baseline opt-in.") from None
        name = model_id or metadata.model_revision or f"{profile}-{metadata.backend_id}"
        model_id = f"baseline:{name}@{digest}"
    if legacy:
        encoded_labels = json.dumps(sorted(set(metadata.labels)), separators=(",", ":")).encode()
        label_set_id = f"labels:{hashlib.sha256(encoded_labels).hexdigest()}"
    if (
        not model_id
        or not label_set_id
        or not all(not any(map(str.isspace, identifier)) for identifier in (model_id, label_set_id))
    ):
        raise AnalysisModelLoadError("The model identifiers are invalid.") from None
    return (
        ModelManifest(
            identity=ModelIdentity(profile, model_id, digest, label_set_id, metadata.labels),
            artifact_format=artifact_format.name,
            training_audio=metadata.training_audio,
            provenance=provenance,
            baseline=baseline,
        ),
        opened,
    )


__all__ = [
    "ArtifactFormat",
    "ArtifactMetadata",
    "LegacyPickleFormat",
    "OpenedArtifact",
    "artifact_formats",
    "open_model",
    "read_model_file",
]
