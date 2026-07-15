"""Versioned manifest schema for cross-domain SER training."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ser._internal.data.ontology import LabelOntology, ensure_label_allowed, normalize_label

MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset({1, MANIFEST_SCHEMA_VERSION})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

type SplitName = Literal["train", "dev", "test"]
type AnnotationTarget = Literal[
    "emotion", "vad", "social_attitude", "binary_affect", "language", "text"
]


def _read_text_field(record: Mapping[str, object], field: str) -> str | None:
    raw = record.get(field)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _read_float_field(record: Mapping[str, object], field: str) -> float | None:
    raw = record.get(field)
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return None


def _read_optional_float_field(record: Mapping[str, object], field: str) -> float | None:
    if field not in record or record.get(field) is None:
        return None
    value = _read_float_field(record, field)
    if value is None:
        raise ValueError(f"Manifest {field!r} must be numeric when provided.")
    return value


def _resolve_audio_path(path_text: str, base_dir: Path) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).expanduser()


def _maybe_relative(path: Path, base_dir: Path) -> Path:
    try:
        return path.relative_to(base_dir)
    except ValueError:
        return path


@dataclass(frozen=True)
class VadTarget:
    """Normalized valence, arousal, and dominance target in ``[-1, 1]``."""

    valence: float
    arousal: float
    dominance: float

    def validate(self) -> None:
        """Validates finite normalized VAD coordinates."""
        for name, value in (
            ("valence", self.valence),
            ("arousal", self.arousal),
            ("dominance", self.dominance),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"VAD {name} must be finite and within [-1, 1].")

    @staticmethod
    def from_record(raw: object) -> VadTarget | None:
        """Parses an optional VAD object from a manifest record."""
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("Manifest 'vad' target must be an object.")
        values: list[float] = []
        for field in ("valence", "arousal", "dominance"):
            value = raw.get(field)
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"Manifest 'vad.{field}' must be numeric.")
            values.append(float(value))
        target = VadTarget(*values)
        target.validate()
        return target

    def to_record(self) -> dict[str, float]:
        """Serializes VAD coordinates."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


@dataclass(frozen=True)
class TargetAnnotation:
    """Source and confidence metadata for one available target."""

    target: AnnotationTarget
    source: str
    confidence: float | None = None

    def validate(self) -> None:
        """Validates annotation source and confidence."""
        if self.target not in {
            "emotion",
            "vad",
            "social_attitude",
            "binary_affect",
            "language",
            "text",
        }:
            raise ValueError(f"Unsupported annotation target {self.target!r}.")
        if not self.source.strip():
            raise ValueError("Annotation source must be non-empty.")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("Annotation confidence must be finite and within [0, 1].")

    @staticmethod
    def from_record(raw: object) -> TargetAnnotation:
        """Parses annotation metadata from a JSON object."""
        if not isinstance(raw, dict):
            raise ValueError("Manifest annotations must contain objects.")
        target = _read_text_field(raw, "target")
        source = _read_text_field(raw, "source")
        confidence = _read_float_field(raw, "confidence")
        if "confidence" in raw and raw.get("confidence") is not None and confidence is None:
            raise ValueError("Manifest annotation confidence must be numeric when provided.")
        if target is None or source is None:
            raise ValueError("Manifest annotations require target and source fields.")
        annotation = TargetAnnotation(cast(AnnotationTarget, target), source, confidence)
        annotation.validate()
        return annotation

    def to_record(self) -> dict[str, object]:
        """Serializes annotation metadata."""
        record: dict[str, object] = {"target": self.target, "source": self.source}
        if self.confidence is not None:
            record["confidence"] = self.confidence
        return record


@dataclass(frozen=True)
class Utterance:
    """One audio segment and any targets available for training."""

    schema_version: int
    sample_id: str
    corpus: str
    audio_path: Path
    label: str | None
    raw_label: str | None = None
    vad: VadTarget | None = None
    social_attitude: str | None = None
    binary_affect: str | None = None
    transcript: str | None = None
    annotations: tuple[TargetAnnotation, ...] = ()
    speaker_id: str | None = None
    session_id: str | None = None
    language: str | None = None
    split: SplitName | None = None
    native_split: SplitName | None = None
    start_seconds: float | None = None
    duration_seconds: float | None = None
    normalized_audio_sha256: str | None = None
    dataset_revision: str | None = None
    dataset_policy_id: str | None = None
    dataset_license_id: str | None = None
    source_url: str | None = None

    def require_label(self) -> str:
        """Returns the primary label or raises at a supervised-only boundary."""
        if self.label is None:
            raise ValueError(f"Utterance {self.sample_id!r} has no primary emotion target.")
        return self.label

    def validate(self, *, ontology: LabelOntology) -> None:
        """Validates record fields and target boundaries."""
        if self.schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported manifest schema version {self.schema_version!r}; "
                f"supported versions are {sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}."
            )
        if not self.sample_id.strip():
            raise ValueError("Utterance.sample_id must be non-empty.")
        if not self.corpus.strip():
            raise ValueError("Utterance.corpus must be non-empty.")
        if not isinstance(self.audio_path, Path) or not str(self.audio_path).strip():
            raise ValueError("Utterance.audio_path must be a non-empty Path.")
        if self.label is not None:
            ensure_label_allowed(label=self.label, ontology=ontology)
        if self.schema_version == 1 and self.label is None:
            raise ValueError("Manifest schema v1 requires a categorical label.")
        if self.schema_version == 2 and not any(
            (
                self.label,
                self.vad,
                self.social_attitude,
                self.binary_affect,
                self.language,
                self.transcript,
            )
        ):
            raise ValueError("Manifest schema v2 requires at least one training target.")
        expected_prefix = f"{self.corpus}:"
        for field_name, identity in (
            ("speaker_id", self.speaker_id),
            ("session_id", self.session_id),
        ):
            if identity is not None and not identity.startswith(expected_prefix):
                raise ValueError(
                    f"{field_name} must be corpus-scoped to avoid collisions: "
                    f"expected prefix {expected_prefix!r} in {identity!r}."
                )
        if self.start_seconds is not None and (
            not math.isfinite(self.start_seconds) or self.start_seconds < 0.0
        ):
            raise ValueError("start_seconds must be finite and non-negative.")
        if self.duration_seconds is not None and (
            not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0
        ):
            raise ValueError("duration_seconds must be finite and positive when provided.")
        if self.normalized_audio_sha256 is not None and not _SHA256_PATTERN.fullmatch(
            self.normalized_audio_sha256
        ):
            raise ValueError("normalized_audio_sha256 must be 64 lowercase hexadecimal characters.")
        if self.dataset_revision is not None and not self.dataset_revision.strip():
            raise ValueError("dataset_revision must be non-empty when provided.")
        if self.vad is not None:
            self.vad.validate()
        seen_targets: set[AnnotationTarget] = set()
        for annotation in self.annotations:
            annotation.validate()
            if annotation.target in seen_targets:
                raise ValueError(f"Duplicate annotation metadata for {annotation.target!r}.")
            seen_targets.add(annotation.target)

    @staticmethod
    def from_record(
        record: Mapping[str, object],
        *,
        base_dir: Path,
        ontology: LabelOntology,
    ) -> Utterance:
        """Builds an utterance from a v1 or v2 parsed manifest record."""
        schema_version_raw = record.get("schema_version", 1)
        if not isinstance(schema_version_raw, int) or isinstance(schema_version_raw, bool):
            raise ValueError("Manifest schema_version must be an integer.")
        if schema_version_raw not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported manifest schema version {schema_version_raw!r}; "
                f"supported versions are {sorted(SUPPORTED_MANIFEST_SCHEMA_VERSIONS)}."
            )
        sample_id = _read_text_field(record, "sample_id")
        corpus = _read_text_field(record, "corpus")
        audio_path_text = _read_text_field(record, "audio_path") or _read_text_field(record, "path")
        if sample_id is None or corpus is None or audio_path_text is None:
            raise ValueError(
                "Manifest record must include sample_id, corpus, and audio_path fields."
            )

        label_text = _read_text_field(record, "label")
        if schema_version_raw == 1 and label_text is None:
            raise ValueError("Manifest schema v1 requires a categorical label.")
        label = normalize_label(label_text) if label_text is not None else None
        split_raw = _read_text_field(record, "split")
        native_split_raw = _read_text_field(record, "native_split")
        annotations_raw = record.get("annotations", [])
        if not isinstance(annotations_raw, list):
            raise ValueError("Manifest 'annotations' must be a list.")
        annotations = tuple(TargetAnnotation.from_record(raw) for raw in annotations_raw)

        utterance = Utterance(
            schema_version=MANIFEST_SCHEMA_VERSION,
            sample_id=sample_id,
            corpus=corpus,
            audio_path=_resolve_audio_path(audio_path_text, base_dir),
            label=label,
            raw_label=_read_text_field(record, "raw_label"),
            vad=VadTarget.from_record(record.get("vad")),
            social_attitude=_read_text_field(record, "social_attitude"),
            binary_affect=_read_text_field(record, "binary_affect"),
            transcript=_read_text_field(record, "transcript"),
            annotations=annotations,
            speaker_id=_read_text_field(record, "speaker_id"),
            session_id=_read_text_field(record, "session_id"),
            language=_read_text_field(record, "language"),
            split=cast(SplitName, split_raw) if split_raw in {"train", "dev", "test"} else None,
            native_split=(
                cast(SplitName, native_split_raw)
                if native_split_raw in {"train", "dev", "test"}
                else None
            ),
            start_seconds=_read_optional_float_field(record, "start_seconds"),
            duration_seconds=_read_optional_float_field(record, "duration_seconds"),
            normalized_audio_sha256=_read_text_field(record, "normalized_audio_sha256"),
            dataset_revision=_read_text_field(record, "dataset_revision"),
            dataset_policy_id=_read_text_field(record, "dataset_policy_id"),
            dataset_license_id=_read_text_field(record, "dataset_license_id"),
            source_url=_read_text_field(record, "source_url"),
        )
        utterance.validate(ontology=ontology)
        return utterance

    def to_record(self, *, base_dir: Path | None = None) -> dict[str, object]:
        """Serializes a v1 or v2 record for JSONL persistence."""
        path = str(_maybe_relative(self.audio_path, base_dir)) if base_dir else str(self.audio_path)
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "corpus": self.corpus,
            "audio_path": path,
        }
        optional_fields: dict[str, object | None] = {
            "label": self.label,
            "raw_label": self.raw_label,
            "vad": self.vad.to_record() if self.vad is not None else None,
            "social_attitude": self.social_attitude,
            "binary_affect": self.binary_affect,
            "transcript": self.transcript,
            "annotations": (
                [annotation.to_record() for annotation in self.annotations]
                if self.annotations
                else None
            ),
            "speaker_id": self.speaker_id,
            "session_id": self.session_id,
            "language": self.language,
            "split": self.split,
            "native_split": self.native_split,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "normalized_audio_sha256": self.normalized_audio_sha256,
            "dataset_revision": self.dataset_revision,
            "dataset_policy_id": self.dataset_policy_id,
            "dataset_license_id": self.dataset_license_id,
            "source_url": self.source_url,
        }
        record.update((key, value) for key, value in optional_fields.items() if value is not None)
        return record
