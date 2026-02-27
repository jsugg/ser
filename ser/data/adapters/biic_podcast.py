"""BIIC-Podcast adapter for building manifest-compatible utterances."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ser.data.manifest import SplitName, Utterance
from ser.data.manifest_jsonl import write_manifest_jsonl
from ser.data.ontology import LabelOntology, remap_label
from ser.utils.logger import get_logger

logger = get_logger(__name__)

BIIC_PODCAST_CORPUS_ID = "biic-podcast"
BIIC_PODCAST_DATASET_POLICY_ID = "academic_only"
BIIC_PODCAST_DATASET_LICENSE_ID = "biic-academic-license"
BIIC_PODCAST_SOURCE_URL = "https://biic.ee.nthu.edu.tw/"


_EMO_CLASS_MAP: dict[str, str] = {
    "0": "angry",
    "1": "sad",
    "2": "happy",
    "3": "surprised",
    "4": "fearful",
    "5": "disgust",
    "6": "contempt",
    "7": "neutral",
}


def _normalize_split(value: str | None) -> SplitName | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in {"train", "training"}:
        return "train"
    if normalized in {"dev", "valid", "validation", "development"}:
        return "dev"
    if normalized in {"test", "evaluation", "eval"}:
        return "test"
    return None


def _read_optional_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = row.get(key)
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            try:
                return float(text)
            except ValueError:
                continue
    return None


def _build_sample_id(
    *,
    file_name: str,
    start_seconds: float | None,
    duration_seconds: float | None,
) -> str:
    normalized = file_name.replace("\\", "/")
    base = f"{BIIC_PODCAST_CORPUS_ID}:{normalized}"
    if start_seconds is None or duration_seconds is None:
        return base
    return f"{base}@{start_seconds:.3f}+{duration_seconds:.3f}"


def build_biic_podcast_utterances(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
) -> list[Utterance] | None:
    """Builds utterances from BIIC-Podcast label CSV.

    The corpus is often distributed under access-controlled terms. This adapter
    expects a CSV/TSV with at least FileName and EmoClass.
    """

    if not labels_csv_path.is_file():
        logger.warning("BIIC-Podcast labels CSV not found: %s", labels_csv_path)
        return None
    base_dir = audio_base_dir if audio_base_dir is not None else dataset_root
    utterances: list[Utterance] = []
    parse_errors: list[str] = []
    with labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            file_name = (row.get("FileName") or row.get("filename") or "").strip()
            if not file_name:
                parse_errors.append(f"Row {index}: missing FileName")
                continue
            raw_class = (row.get("EmoClass") or row.get("emotion") or "").strip()
            if not raw_class:
                parse_errors.append(f"Row {index}: missing EmoClass for {file_name}")
                continue
            mapped_raw = _EMO_CLASS_MAP.get(raw_class, raw_class)
            mapped_label = remap_label(
                raw_label=mapped_raw,
                mapping={
                    "anger": "angry",
                    "angry": "angry",
                    "sad": "sad",
                    "happy": "happy",
                    "surprise": "surprised",
                    "surprised": "surprised",
                    "fear": "fearful",
                    "fearful": "fearful",
                    "disgust": "disgust",
                    "neutral": "neutral",
                    "contempt": "contempt",
                },
                ontology=ontology,
            )
            if mapped_label is None:
                continue

            split = _normalize_split(row.get("Split_Set") or row.get("split"))
            start_seconds = _read_optional_float(
                row,
                "start_seconds",
                "Start",
                "start",
                "start_time",
                "StartTime",
            )
            duration_seconds = _read_optional_float(
                row,
                "duration_seconds",
                "Duration",
                "duration",
            )
            end_seconds = _read_optional_float(
                row,
                "end_seconds",
                "End",
                "end",
                "end_time",
                "EndTime",
            )
            if (
                duration_seconds is None
                and start_seconds is not None
                and end_seconds is not None
            ):
                duration = end_seconds - start_seconds
                duration_seconds = duration if duration > 0.0 else None

            language = (
                row.get("Language") or row.get("language") or ""
            ).strip() or default_language
            speaker_raw = (
                row.get("Speaker") or row.get("Speaker_ID") or row.get("speaker") or ""
            ).strip()
            speaker_id = (
                f"{BIIC_PODCAST_CORPUS_ID}:{speaker_raw}" if speaker_raw else None
            )
            audio_path = (base_dir / file_name).expanduser()
            sample_id = _build_sample_id(
                file_name=file_name,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
            utterances.append(
                Utterance(
                    schema_version=1,
                    sample_id=sample_id,
                    corpus=BIIC_PODCAST_CORPUS_ID,
                    audio_path=audio_path,
                    label=mapped_label,
                    raw_label=mapped_raw,
                    speaker_id=speaker_id,
                    language=language,
                    split=split,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    dataset_policy_id=BIIC_PODCAST_DATASET_POLICY_ID,
                    dataset_license_id=BIIC_PODCAST_DATASET_LICENSE_ID,
                    source_url=BIIC_PODCAST_SOURCE_URL,
                )
            )

    if parse_errors:
        logger.warning(
            "Skipped %s rows while parsing BIIC-Podcast labels.",
            len(parse_errors),
        )
        for error in parse_errors[:5]:
            logger.warning("%s", error)

    total_rows = len(utterances) + len(parse_errors)
    if total_rows > 0:
        failure_ratio = float(len(parse_errors)) / float(total_rows)
        if failure_ratio > max_failed_file_ratio:
            raise RuntimeError(
                "Aborting data load: "
                f"{failure_ratio * 100.0:.1f}% label-row failures exceeded configured "
                "limit "
                f"{max_failed_file_ratio * 100.0:.1f}%."
            )

    if not utterances:
        logger.warning("No labeled audio samples matched configured emotions.")
        return None
    labels = [utterance.label for utterance in utterances]
    if len(set(labels)) < 2:
        logger.warning("At least two emotion classes are required to train the model.")
        return None
    return utterances


def build_biic_podcast_manifest_jsonl(
    *,
    dataset_root: Path,
    labels_csv_path: Path,
    audio_base_dir: Path | None,
    ontology: LabelOntology,
    default_language: str,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a BIIC-Podcast JSONL manifest."""

    utterances = build_biic_podcast_utterances(
        dataset_root=dataset_root,
        labels_csv_path=labels_csv_path,
        audio_base_dir=audio_base_dir,
        ontology=ontology,
        default_language=default_language,
        max_failed_file_ratio=max_failed_file_ratio,
    )
    if utterances is None:
        return None
    write_manifest_jsonl(output_path, utterances, base_dir=dataset_root)
    return utterances
