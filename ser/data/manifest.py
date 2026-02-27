"""Manifest schema for multi-corpus SER training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ser.data.ontology import LabelOntology, ensure_label_allowed, normalize_label

MANIFEST_SCHEMA_VERSION = 1

type SplitName = Literal["train", "dev", "test"]


def _read_text_field(record: Mapping[str, object], field: str) -> str | None:
    raw = record.get(field)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _read_float_field(record: Mapping[str, object], field: str) -> float | None:
    raw = record.get(field)
    if isinstance(raw, int | float):
        return float(raw)
    return None


def _resolve_audio_path(path_text: str, base_dir: Path) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).expanduser()


def _maybe_relative(path: Path, base_dir: Path) -> Path:
    try:
        return path.relative_to(base_dir)
    except Exception:
        return path


@dataclass(frozen=True)
class Utterance:
    """One supervised SER training record."""

    schema_version: int
    sample_id: str
    corpus: str
    audio_path: Path
    label: str
    raw_label: str | None = None
    speaker_id: str | None = None
    language: str | None = None
    split: SplitName | None = None
    start_seconds: float | None = None
    duration_seconds: float | None = None
    dataset_policy_id: str | None = None
    dataset_license_id: str | None = None
    source_url: str | None = None

    def validate(self, *, ontology: LabelOntology) -> None:
        """Validates record fields for stability across adapters."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported manifest schema version {self.schema_version!r}; "
                f"expected {MANIFEST_SCHEMA_VERSION}."
            )
        if not self.sample_id.strip():
            raise ValueError("Utterance.sample_id must be non-empty.")
        if not self.corpus.strip():
            raise ValueError("Utterance.corpus must be non-empty.")
        if not isinstance(self.audio_path, Path):
            raise ValueError("Utterance.audio_path must be a Path.")
        if not str(self.audio_path).strip():
            raise ValueError("Utterance.audio_path must be non-empty.")
        ensure_label_allowed(label=self.label, ontology=ontology)
        if self.speaker_id is not None:
            expected_prefix = f"{self.corpus}:"
            if not self.speaker_id.startswith(expected_prefix):
                raise ValueError(
                    "speaker_id must be corpus-scoped to avoid collisions: "
                    f"expected prefix {expected_prefix!r} in {self.speaker_id!r}."
                )
        if self.start_seconds is not None and self.start_seconds < 0.0:
            raise ValueError("start_seconds must be non-negative.")
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be positive when provided.")

    @staticmethod
    def from_record(
        record: Mapping[str, object],
        *,
        base_dir: Path,
        ontology: LabelOntology,
    ) -> Utterance:
        """Builds an Utterance from a parsed manifest record."""
        schema_version_raw = record.get("schema_version", MANIFEST_SCHEMA_VERSION)
        schema_version = (
            int(schema_version_raw)
            if isinstance(schema_version_raw, int)
            else MANIFEST_SCHEMA_VERSION
        )
        sample_id = _read_text_field(record, "sample_id")
        corpus = _read_text_field(record, "corpus")
        audio_path_text = _read_text_field(record, "audio_path") or _read_text_field(
            record, "path"
        )
        label_text = _read_text_field(record, "label")
        if (
            sample_id is None
            or corpus is None
            or audio_path_text is None
            or label_text is None
        ):
            raise ValueError(
                "Manifest record must include sample_id, corpus, audio_path, and label fields."
            )

        label = normalize_label(label_text)
        speaker_id = _read_text_field(record, "speaker_id")
        language = _read_text_field(record, "language")
        split_raw = _read_text_field(record, "split")
        split: SplitName | None = (
            cast(SplitName, split_raw)
            if split_raw in {"train", "dev", "test"}
            else None
        )
        start_seconds = _read_float_field(record, "start_seconds")
        duration_seconds = _read_float_field(record, "duration_seconds")

        utterance = Utterance(
            schema_version=schema_version,
            sample_id=sample_id,
            corpus=corpus,
            audio_path=_resolve_audio_path(audio_path_text, base_dir),
            label=label,
            raw_label=_read_text_field(record, "raw_label"),
            speaker_id=speaker_id,
            language=language,
            split=split,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            dataset_policy_id=_read_text_field(record, "dataset_policy_id"),
            dataset_license_id=_read_text_field(record, "dataset_license_id"),
            source_url=_read_text_field(record, "source_url"),
        )
        utterance.validate(ontology=ontology)
        return utterance

    def to_record(self, *, base_dir: Path | None = None) -> dict[str, object]:
        """Serializes record for JSONL persistence."""
        path = (
            str(_maybe_relative(self.audio_path, base_dir))
            if base_dir is not None
            else str(self.audio_path)
        )
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "corpus": self.corpus,
            "audio_path": path,
            "label": self.label,
        }
        optional_fields: dict[str, object | None] = {
            "raw_label": self.raw_label,
            "speaker_id": self.speaker_id,
            "language": self.language,
            "split": self.split,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "dataset_policy_id": self.dataset_policy_id,
            "dataset_license_id": self.dataset_license_id,
            "source_url": self.source_url,
        }
        for key, value in optional_fields.items():
            if value is not None:
                record[key] = value
        return record
