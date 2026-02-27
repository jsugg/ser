"""CREMA-D adapter for building manifest-compatible utterances."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from ser.data.manifest import Utterance
from ser.data.manifest_jsonl import write_manifest_jsonl
from ser.data.ontology import LabelOntology, remap_label
from ser.utils.logger import get_logger

logger = get_logger(__name__)

CREMA_D_CORPUS_ID = "crema-d"
CREMA_D_DATASET_POLICY_ID = "share_alike"
CREMA_D_DATASET_LICENSE_ID = "odbl-1.0"
CREMA_D_SOURCE_URL = "https://github.com/CheyneyComputerScience/CREMA-D"


def _extract_actor_id(file_name: str) -> str | None:
    parts = file_name.split("_")
    if len(parts) < 3:
        return None
    actor = parts[0].strip()
    return actor or None


def _extract_emotion_code(file_name: str) -> str | None:
    parts = file_name.split("_")
    if len(parts) < 3:
        return None
    code = parts[2].split(".")[0].strip()
    return code or None


def build_crema_d_utterances(
    *,
    dataset_root: Path,
    dataset_glob_pattern: str,
    emotion_code_map: Mapping[str, str],
    default_language: str,
    ontology: LabelOntology,
    max_failed_file_ratio: float,
) -> list[Utterance] | None:
    """Builds utterances from a CREMA-D folder layout.

    Args:
        dataset_root: Root folder used to create stable sample_id values.
        dataset_glob_pattern: Glob used to discover audio files.
        emotion_code_map: Mapping from CREMA-D emotion code to canonical label.
        default_language: Language tag for created utterances.
        ontology: Canonical label ontology.
        max_failed_file_ratio: Maximum fraction of filename parse failures.

    Returns:
        A list of utterances or ``None`` when no usable samples exist.
    """

    files = sorted(dataset_root.glob(dataset_glob_pattern))
    if not files:
        logger.warning("No dataset files found for pattern: %s", dataset_glob_pattern)
        return None

    utterances: list[Utterance] = []
    parse_errors: list[str] = []
    root = dataset_root.expanduser()
    for audio_path in files:
        if audio_path.is_dir():
            continue
        file_name = os.path.basename(str(audio_path))
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

        actor_raw = _extract_actor_id(file_name)
        speaker_id = (
            f"{CREMA_D_CORPUS_ID}:{actor_raw}" if actor_raw is not None else None
        )
        try:
            rel = audio_path.expanduser().relative_to(root)
            sample_id = f"{CREMA_D_CORPUS_ID}:{rel.as_posix()}"
        except Exception:
            sample_id = f"{CREMA_D_CORPUS_ID}:{audio_path.name}"

        utterances.append(
            Utterance(
                schema_version=1,
                sample_id=sample_id,
                corpus=CREMA_D_CORPUS_ID,
                audio_path=audio_path.expanduser(),
                label=mapped_label,
                raw_label=emotion_code,
                speaker_id=speaker_id,
                language=default_language,
                split=None,
                dataset_policy_id=CREMA_D_DATASET_POLICY_ID,
                dataset_license_id=CREMA_D_DATASET_LICENSE_ID,
                source_url=CREMA_D_SOURCE_URL,
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


def build_crema_d_manifest_jsonl(
    *,
    dataset_root: Path,
    dataset_glob_pattern: str,
    emotion_code_map: Mapping[str, str],
    default_language: str,
    ontology: LabelOntology,
    max_failed_file_ratio: float,
    output_path: Path,
) -> list[Utterance] | None:
    """Builds utterances and persists a CREMA-D JSONL manifest."""

    utterances = build_crema_d_utterances(
        dataset_root=dataset_root,
        dataset_glob_pattern=dataset_glob_pattern,
        emotion_code_map=emotion_code_map,
        default_language=default_language,
        ontology=ontology,
        max_failed_file_ratio=max_failed_file_ratio,
    )
    if utterances is None:
        return None
    write_manifest_jsonl(output_path, utterances, base_dir=dataset_root)
    return utterances
