"""RAVDESS adapter that emits manifest-compatible utterances."""

from __future__ import annotations

import glob
import os
from collections.abc import Mapping
from pathlib import Path

from ser.data.manifest import MANIFEST_SCHEMA_VERSION, Utterance
from ser.data.ontology import LabelOntology, remap_label
from ser.utils.logger import get_logger

logger = get_logger(__name__)

RAVDESS_CORPUS_ID = "ravdess"


def _extract_emotion_code(file_name: str) -> str | None:
    parts = file_name.split("-")
    return None if len(parts) < 3 else parts[2]


def _extract_speaker_id(file_name: str) -> str | None:
    parts = file_name.split("-")
    if len(parts) < 7:
        return None
    speaker_id = parts[6].split(".")[0].strip()
    return speaker_id or None


def build_ravdess_utterances(
    *,
    dataset_root: Path,
    dataset_glob_pattern: str,
    emotion_code_map: Mapping[str, str],
    default_language: str,
    ontology: LabelOntology,
    max_failed_file_ratio: float,
) -> list[Utterance] | None:
    """Builds utterances from a RAVDESS-style folder layout."""
    files = sorted(glob.glob(dataset_glob_pattern))
    if not files:
        logger.warning("No dataset files found for pattern: %s", dataset_glob_pattern)
        return None

    utterances: list[Utterance] = []
    parse_errors: list[str] = []
    root = dataset_root.expanduser()
    for file_path in files:
        file_name = os.path.basename(file_path)
        emotion_code = _extract_emotion_code(file_name)
        if emotion_code is None:
            parse_errors.append(
                "Skipping file with unexpected name format "
                f"(missing emotion code): {file_name}"
            )
            continue
        mapped_label = remap_label(
            raw_label=emotion_code,
            mapping=emotion_code_map,
            ontology=ontology,
        )
        if mapped_label is None:
            continue

        audio_path = Path(file_path).expanduser()
        speaker_raw = _extract_speaker_id(file_name)
        speaker_id = (
            f"{RAVDESS_CORPUS_ID}:{speaker_raw}" if speaker_raw is not None else None
        )
        try:
            rel_path = audio_path.relative_to(root).as_posix()
            sample_id = f"{RAVDESS_CORPUS_ID}:{rel_path}"
        except ValueError:
            sample_id = f"{RAVDESS_CORPUS_ID}:{audio_path.name}"
        utterances.append(
            Utterance(
                schema_version=MANIFEST_SCHEMA_VERSION,
                sample_id=sample_id,
                corpus=RAVDESS_CORPUS_ID,
                audio_path=audio_path,
                label=mapped_label,
                raw_label=emotion_code,
                speaker_id=speaker_id,
                language=default_language,
            )
        )

    if parse_errors:
        logger.warning(
            "Skipped %s/%s files while resolving labels.",
            len(parse_errors),
            len(files),
        )
        for error in parse_errors[:5]:
            logger.warning("%s", error)

    failure_ratio = float(len(parse_errors)) / float(len(files))
    if failure_ratio > max_failed_file_ratio:
        raise RuntimeError(
            "Aborting data load: "
            f"{failure_ratio * 100.0:.1f}% file failures exceeded configured "
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
