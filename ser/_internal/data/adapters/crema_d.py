"""CREMA-D adapter for building manifest-compatible utterances."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path

import soundfile as sf

from ser._internal.data.manifest import Utterance
from ser._internal.data.manifest_jsonl import write_manifest_jsonl
from ser._internal.data.ontology import LabelOntology, remap_label
from ser._internal.utils.logger import get_logger

logger = get_logger(__name__)

CREMA_D_CORPUS_ID = "crema-d"
CREMA_D_DATASET_POLICY_ID = "share_alike"
CREMA_D_DATASET_LICENSE_ID = "odbl-1.0"
CREMA_D_SOURCE_URL = "https://github.com/CheyneyComputerScience/CREMA-D"
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_MIN_WAV_FILE_BYTES = 44


class CremaDDatasetIntegrityError(RuntimeError):
    """Raised when a CREMA-D tree contains missing or invalid audio."""


def _audio_integrity_error(audio_path: Path) -> str | None:
    """Returns an integrity diagnostic for one CREMA-D audio file."""
    try:
        if not audio_path.is_file():
            return "not a regular file"
        file_size = audio_path.stat().st_size
        with audio_path.open("rb") as audio_file:
            prefix = audio_file.read(len(_GIT_LFS_POINTER_PREFIX))
        if prefix == _GIT_LFS_POINTER_PREFIX:
            return "unmaterialized Git LFS pointer"
        if file_size == 0:
            return "empty file"
        if file_size < _MIN_WAV_FILE_BYTES:
            return f"implausibly small file ({file_size} bytes)"
        metadata = sf.info(str(audio_path))
        if metadata.samplerate <= 0:
            return "invalid sample rate"
        if metadata.channels <= 0:
            return "invalid channel count"
        if metadata.frames <= 0:
            return "audio contains no frames"
    except Exception as err:
        detail = str(err).strip() or type(err).__name__
        return f"audio metadata decode failed ({detail})"
    return None


def validate_crema_d_audio_files(
    *,
    dataset_root: Path,
    dataset_glob_pattern: str,
) -> tuple[Path, ...]:
    """Validates every candidate CREMA-D WAV before manifest registration.

    Args:
        dataset_root: Root of the CREMA-D checkout.
        dataset_glob_pattern: Glob used to discover candidate WAV files.

    Returns:
        Validated audio paths in deterministic order.

    Raises:
        CremaDDatasetIntegrityError: If audio is missing, unmaterialized, or invalid.
    """
    started_at = time.perf_counter()
    files = tuple(sorted(dataset_root.glob(dataset_glob_pattern)))
    if not files:
        raise CremaDDatasetIntegrityError(
            "CREMA-D audio integrity check failed: no WAV files were found. "
            "The dataset checkout is incomplete."
        )
    logger.info(
        "DATASET_INTEGRITY_START dataset_id=%s files=%d root=%s",
        CREMA_D_CORPUS_ID,
        len(files),
        dataset_root,
    )

    invalid_count = 0
    samples: list[str] = []
    interval = max(1, len(files) // 10)
    last_progress_at = started_at
    for processed, audio_path in enumerate(files, start=1):
        diagnostic = _audio_integrity_error(audio_path)
        if diagnostic is None:
            pass
        else:
            invalid_count += 1
            if len(samples) < 5:
                try:
                    display_path = audio_path.relative_to(dataset_root).as_posix()
                except ValueError:
                    display_path = str(audio_path)
                samples.append(f"{display_path}: {diagnostic}")
        now = time.perf_counter()
        if (
            processed == 1
            or processed == len(files)
            or processed % interval == 0
            or (now - last_progress_at) >= 30.0
        ):
            logger.info(
                "DATASET_INTEGRITY_PROGRESS dataset_id=%s checked=%d total=%d invalid=%d elapsed=%.1fs",
                CREMA_D_CORPUS_ID,
                processed,
                len(files),
                invalid_count,
                now - started_at,
            )
            last_progress_at = now

    if invalid_count:
        sample_text = "; ".join(samples)
        logger.error(
            "DATASET_INTEGRITY_DONE dataset_id=%s files=%d invalid=%d elapsed=%.1fs",
            CREMA_D_CORPUS_ID,
            len(files),
            invalid_count,
            time.perf_counter() - started_at,
        )
        raise CremaDDatasetIntegrityError(
            "CREMA-D audio integrity check failed: "
            f"{invalid_count}/{len(files)} invalid file(s). Examples: {sample_text}. "
            "CREMA-D audio must be materialized with Git LFS; run `git lfs pull` and "
            "`git lfs checkout`, or move the incomplete dataset aside and download it again."
        )
    logger.info(
        "DATASET_INTEGRITY_DONE dataset_id=%s files=%d invalid=0 elapsed=%.1fs",
        CREMA_D_CORPUS_ID,
        len(files),
        time.perf_counter() - started_at,
    )
    return files


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

    files = validate_crema_d_audio_files(
        dataset_root=dataset_root,
        dataset_glob_pattern=dataset_glob_pattern,
    )

    utterances: list[Utterance] = []
    parse_errors: list[str] = []
    root = dataset_root.expanduser()
    for audio_path in files:
        file_name = os.path.basename(str(audio_path))
        emotion_code = _extract_emotion_code(file_name)
        if emotion_code is None:
            parse_errors.append(
                f"Skipping file with unexpected name format (missing emotion code): {file_name}"
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
        speaker_id = f"{CREMA_D_CORPUS_ID}:{actor_raw}" if actor_raw is not None else None
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
