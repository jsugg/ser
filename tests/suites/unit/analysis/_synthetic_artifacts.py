"""A synthetic "safe" model format for the contract tests, plus a booby-trapped pickle.

A file is ``MAGIC``, one JSON header line, then the payload. The header carries what
brief P2-4's format will carry; the payload is the scripted backend's behaviour.
The format only parses: verification is the analyser's job.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from ser._internal.analysis.artifact import ArtifactMetadata, OpenedArtifact
from ser.profiles import ProfileName
from ser.runtime.contracts import (
    LicenceClass,
    LicenceRecord,
    ModelFormat,
    ModelProvenance,
    TrainingAudioForm,
)

MAGIC = b"SER-TEST-ARTIFACT/1\n"
SCRIPTED_LABELS: tuple[str, ...] = ("a", "b", "c")
CANONICAL_FORM: Mapping[str, object] = {
    "sample_rate_hz": 16_000,
    "channels": 1,
    "canonical_conversion": True,
}
OPEN_ENCODER: Mapping[str, str] = {
    "name": "test-encoder",
    "licence": "MIT",
    "licence_class": "open",
    "source": "generated",
}
RESEARCH_DATASET: Mapping[str, str] = {
    "name": "test-dataset",
    "licence": "custom",
    "licence_class": "research-only",
    "source": "generated",
}

type ProvenanceMode = Literal["valid", "missing", "tampered"]


def write_artifact(
    path: Path,
    *,
    script: Mapping[str, object] | None = None,
    labels: Sequence[str] = SCRIPTED_LABELS,
    profile: str = "fast",
    backend_id: str = "scripted",
    model_id: str | None = "test-model",
    label_set_id: str | None = "test-labels/v1",
    training_audio: Mapping[str, object] | None = CANONICAL_FORM,
    provenance: ProvenanceMode = "valid",
    datasets: Sequence[Mapping[str, str]] = (RESEARCH_DATASET,),
) -> Path:
    """Writes one synthetic artifact whose payload is ``script`` as JSON."""
    payload = json.dumps(dict(script or {}), sort_keys=True).encode()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    header: dict[str, object] = {
        "profile": profile,
        "backend_id": backend_id,
        "model_id": model_id,
        "label_set_id": label_set_id,
        "labels": list(labels),
        "training_audio": None if training_audio is None else dict(training_audio),
        "provenance": (
            None
            if provenance == "missing"
            else {
                "payload_sha256": payload_sha256,
                "encoder": dict(OPEN_ENCODER),
                "datasets": [dict(dataset) for dataset in datasets],
            }
        ),
    }
    if provenance == "tampered":
        payload = payload + b" "
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(MAGIC + json.dumps(header).encode() + b"\n" + payload)
    return path


def payload_of(path: Path) -> bytes:
    """The payload bytes of a synthetic artifact."""
    return path.read_bytes()[len(MAGIC) :].split(b"\n", 1)[1]


class SyntheticSafeFormat:
    """Registered in place of P2-4's safe format during the contract tests."""

    @property
    def name(self) -> ModelFormat:
        return "safe"

    def claims(self, data: bytes) -> bool:
        return data.startswith(MAGIC)

    def open(self, data: bytes) -> OpenedArtifact:
        header_line, payload = data[len(MAGIC) :].split(b"\n", 1)
        header = cast(dict[str, object], json.loads(header_line))
        training = cast(dict[str, object] | None, header["training_audio"])
        provenance = cast(dict[str, object] | None, header["provenance"])
        return OpenedArtifact(
            metadata=ArtifactMetadata(
                profile=cast(ProfileName, header["profile"]),
                backend_id=cast(str, header["backend_id"]),
                model_id=cast(str | None, header["model_id"]),
                label_set_id=cast(str | None, header["label_set_id"]),
                labels=tuple(cast(list[str], header["labels"])),
                training_audio=(
                    None
                    if training is None
                    else TrainingAudioForm(
                        sample_rate_hz=cast(int, training["sample_rate_hz"]),
                        channels=cast(int, training["channels"]),
                        canonical_conversion=cast(bool, training["canonical_conversion"]),
                    )
                ),
                provenance=None if provenance is None else _provenance(provenance),
            ),
            payload=payload,
        )


def _licence(raw: object) -> LicenceRecord:
    fields = cast(dict[str, str], raw)
    return LicenceRecord(
        name=fields["name"],
        licence=fields["licence"],
        licence_class=cast(LicenceClass, fields["licence_class"]),
        source=fields["source"],
    )


def _provenance(raw: dict[str, object]) -> ModelProvenance:
    encoder = raw["encoder"]
    return ModelProvenance(
        payload_sha256=cast(str, raw["payload_sha256"]),
        encoder=None if encoder is None else _licence(encoder),
        datasets=tuple(_licence(dataset) for dataset in cast(list[object], raw["datasets"])),
    )


TRAP: dict[str, bool] = {"fired": False}


def _spring() -> None:
    TRAP["fired"] = True


class _Trap:
    def __reduce__(self) -> tuple[object, tuple[()]]:
        return (_spring, ())


def write_booby_trapped_pickle(path: Path) -> Path:
    """A pickle that records being unpickled; refusing it must leave ``TRAP`` unset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(_Trap(), protocol=pickle.HIGHEST_PROTOCOL))
    return path
