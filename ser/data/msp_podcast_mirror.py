"""MSP-Podcast mirror acquisition and label generation utilities.

This module downloads the public `AbstractTTS/PODCAST` mirror from Hugging Face,
extracts embedded WAV bytes from parquet shards, writes a metadata index, and
generates an adapter-compatible labels CSV.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ser.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MSP_MIRROR_REPO_ID = "AbstractTTS/PODCAST"
DEFAULT_MSP_MIRROR_REVISION = "main"
DEFAULT_MSP_MIRROR_MAX_WORKERS = 8
DEFAULT_MSP_MIRROR_BATCH_SIZE = 64
DEFAULT_MSP_MIRROR_AUDIO_SUBDIR = "audio"
DEFAULT_MSP_MIRROR_REPO_SUBDIR = "repo"
DEFAULT_MSP_MIRROR_METADATA_FILE = "metadata.jsonl"
DEFAULT_MSP_MIRROR_LABELS_FILE = "labels.csv"
DEFAULT_MSP_MIRROR_MANIFEST_FILE = "manifest.json"

_CANONICAL_SCORE_COLUMNS: tuple[str, ...] = (
    "angry",
    "sad",
    "happy",
    "surprise",
    "fear",
    "disgust",
    "contempt",
    "neutral",
)
_CANONICAL_SCORE_MAP: dict[str, str] = {
    "angry": "angry",
    "sad": "sad",
    "happy": "happy",
    "surprise": "surprised",
    "fear": "fearful",
    "disgust": "disgust",
    "contempt": "contempt",
    "neutral": "neutral",
}
_MAJOR_EMOTION_MAP: dict[str, str] = {
    "angry": "angry",
    "anger": "angry",
    "sad": "sad",
    "happy": "happy",
    "surprise": "surprised",
    "surprised": "surprised",
    "fear": "fearful",
    "fearful": "fearful",
    "disgust": "disgust",
    "disgusted": "disgust",
    "contempt": "contempt",
    "neutral": "neutral",
}


@dataclass(frozen=True, slots=True)
class MspPodcastMirrorArtifacts:
    """Artifacts produced from one MSP mirror acquisition run.

    Attributes:
        dataset_root: Dataset root directory.
        repo_dir: Local snapshot mirror directory.
        audio_dir: Extracted WAV root.
        metadata_jsonl_path: Extracted metadata JSONL path.
        labels_csv_path: Generated labels CSV path.
        manifest_json_path: Provenance manifest path.
        rows_seen: Number of parquet rows processed.
        files_written: Number of WAV files written this run.
        files_reused: Number of WAV files reused from previous run.
        labels_written: Number of label rows written.
        dropped_rows: Number of metadata rows dropped from labels CSV.
        commit_sha: Resolved dataset commit SHA, when available.
    """

    dataset_root: Path
    repo_dir: Path
    audio_dir: Path
    metadata_jsonl_path: Path
    labels_csv_path: Path
    manifest_json_path: Path
    rows_seen: int
    files_written: int
    files_reused: int
    labels_written: int
    dropped_rows: int
    commit_sha: str | None


@dataclass(frozen=True, slots=True)
class _RemoteFile:
    path: str
    size: int | None


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def _safe_relative_path(value: str) -> Path:
    """Converts one POSIX relative path into a safe local path.

    Args:
        value: Relative POSIX path value from mirror metadata.

    Returns:
        Safe local relative path.

    Raises:
        ValueError: Path is absolute or traversal-like.
    """

    posix_path = PurePosixPath(value)
    if posix_path.is_absolute():
        raise ValueError(f"Expected relative path but got absolute path: {value!r}")
    if any(part in {"", ".."} for part in posix_path.parts):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return Path(*posix_path.parts)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _parse_finite_float(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _resolve_canonical_label(metadata_record: dict[str, Any]) -> str | None:
    """Resolves one canonical label from mirror metadata.

    Resolution policy:
      1. Prefer argmax over canonical score columns (deterministic tie-breaking
         by fixed column order).
      2. Fallback to normalized `major_emotion` mapping.

    Args:
        metadata_record: One metadata row.

    Returns:
        Canonical label for adapters (`angry|sad|happy|surprised|fearful|...`)
        or ``None`` when no supported label can be determined.
    """

    best_column: str | None = None
    best_score: float | None = None
    for column in _CANONICAL_SCORE_COLUMNS:
        value = _parse_finite_float(metadata_record.get(column))
        if value is None:
            continue
        if best_score is None or value > best_score:
            best_score = value
            best_column = column
    if best_column is not None:
        return _CANONICAL_SCORE_MAP[best_column]

    major_raw = metadata_record.get("major_emotion")
    if isinstance(major_raw, str):
        normalized = major_raw.strip().lower()
        if normalized:
            return _MAJOR_EMOTION_MAP.get(normalized)
    return None


def generate_msp_labels_csv_from_metadata_jsonl(
    *,
    metadata_jsonl_path: Path,
    labels_csv_path: Path,
) -> tuple[int, int]:
    """Generates adapter-compatible labels CSV from mirror metadata JSONL.

    Args:
        metadata_jsonl_path: Input metadata JSONL path.
        labels_csv_path: Output labels CSV path.

    Returns:
        Tuple `(labels_written, dropped_rows)`.
    """

    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = labels_csv_path.with_suffix(labels_csv_path.suffix + ".tmp")
    labels_by_file: dict[str, str] = {}
    duplicate_conflict_rows = 0
    duplicate_same_label_rows = 0
    dropped_rows = 0
    with metadata_jsonl_path.open("r", encoding="utf-8") as source_fp:
        for line in source_fp:
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                dropped_rows += 1
                continue
            metadata_record = dict(parsed)
            label = _resolve_canonical_label(metadata_record)
            if label is None:
                dropped_rows += 1
                continue
            audio_relpath_raw = metadata_record.get("audio_relpath")
            if not isinstance(audio_relpath_raw, str) or not audio_relpath_raw.strip():
                dropped_rows += 1
                continue
            audio_relpath = _safe_relative_path(audio_relpath_raw.strip()).as_posix()
            existing_label = labels_by_file.get(audio_relpath)
            if existing_label is not None:
                dropped_rows += 1
                if existing_label != label:
                    duplicate_conflict_rows += 1
                else:
                    duplicate_same_label_rows += 1
                continue
            labels_by_file[audio_relpath] = label
    labels_written = len(labels_by_file)
    with tmp_path.open("w", encoding="utf-8", newline="") as target_fp:
        writer = csv.DictWriter(target_fp, fieldnames=["FileName", "emotion"])
        writer.writeheader()
        for audio_relpath in sorted(labels_by_file):
            writer.writerow(
                {"FileName": audio_relpath, "emotion": labels_by_file[audio_relpath]}
            )
    if duplicate_conflict_rows > 0:
        logger.warning(
            "MSP mirror labels dropped %s conflicting duplicate row(s) by audio_relpath.",
            duplicate_conflict_rows,
        )
    if duplicate_same_label_rows > 0:
        logger.info(
            "MSP mirror labels ignored %s duplicate row(s) with identical labels.",
            duplicate_same_label_rows,
        )
    os.replace(tmp_path, labels_csv_path)
    return labels_written, dropped_rows


def _load_hf_clients() -> tuple[type[Any], Any, Any]:
    """Lazily imports optional Hugging Face/pyarrow dependencies."""

    try:
        hf_module = importlib.import_module("huggingface_hub")
        HfApi = hf_module.HfApi
        snapshot_download = hf_module.snapshot_download
    except Exception as err:
        raise RuntimeError(
            "MSP mirror download requires dependency `huggingface_hub`. "
            "Install it with `uv pip install huggingface_hub`."
        ) from err
    try:
        pq = importlib.import_module("pyarrow.parquet")
    except Exception as err:
        raise RuntimeError(
            "MSP mirror extraction requires dependency `pyarrow`. "
            "Install it with `uv pip install pyarrow`."
        ) from err
    return HfApi, snapshot_download, pq


def _download_snapshot(
    *,
    repo_id: str,
    revision: str,
    repo_dir: Path,
    max_workers: int,
    token: str | None,
) -> tuple[list[_RemoteFile], str | None]:
    """Downloads one dataset snapshot and validates advertised files."""

    HfApi, snapshot_download, _pq = _load_hf_clients()
    api = HfApi(token=token)
    info = api.dataset_info(repo_id=repo_id, revision=revision, files_metadata=True)
    remote_files: list[_RemoteFile] = []
    for sibling in getattr(info, "siblings", None) or []:
        path = getattr(sibling, "rfilename", None)
        if not isinstance(path, str) or not path:
            continue
        size_raw = getattr(sibling, "size", None)
        size = size_raw if isinstance(size_raw, int) and size_raw >= 0 else None
        remote_files.append(_RemoteFile(path=path, size=size))
    remote_files.sort(key=lambda item: item.path)
    required_snapshot_bytes = sum(
        remote.size for remote in remote_files if remote.size is not None
    )
    # Conservative estimate: snapshot parquet + extracted audio in dataset_root.
    required_free_bytes = required_snapshot_bytes * 2
    free_bytes = shutil.disk_usage(repo_dir).free
    if required_free_bytes > 0 and free_bytes < required_free_bytes:
        raise RuntimeError(
            "MSP mirror download aborted due insufficient disk space. "
            f"Required approximately {required_free_bytes} bytes, free {free_bytes} bytes at {repo_dir}. "
            "Use `--dataset-root` on a volume with enough space or uninstall other datasets first."
        )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=repo_dir,
        token=token,
        max_workers=max_workers,
    )

    missing_files: list[str] = []
    size_mismatches: list[str] = []
    for remote in remote_files:
        local_path = repo_dir / remote.path
        if not local_path.is_file():
            missing_files.append(remote.path)
            continue
        if remote.size is None:
            continue
        local_size = local_path.stat().st_size
        if local_size != remote.size:
            size_mismatches.append(
                f"{remote.path}: local={local_size} remote={remote.size}"
            )
    if missing_files or size_mismatches:
        raise RuntimeError(
            "MSP mirror validation failed: "
            f"missing={len(missing_files)} mismatched={len(size_mismatches)}."
        )
    commit_sha_raw = getattr(info, "sha", None)
    commit_sha = commit_sha_raw if isinstance(commit_sha_raw, str) else None
    return remote_files, commit_sha


def _extract_metadata_and_audio(
    *,
    repo_dir: Path,
    audio_dir: Path,
    metadata_jsonl_path: Path,
    batch_size: int,
) -> tuple[int, int, int]:
    """Extracts embedded WAV bytes and metadata rows from parquet shards."""

    _HfApi, _snapshot_download, pq = _load_hf_clients()
    data_dir = repo_dir / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"MSP mirror data shard directory missing: {data_dir}")
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError(f"MSP mirror data shard glob is empty: {data_dir}")

    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_tmp_path = metadata_jsonl_path.with_suffix(
        metadata_jsonl_path.suffix + ".tmp"
    )

    rows_seen = 0
    files_written = 0
    files_reused = 0
    with metadata_tmp_path.open("w", encoding="utf-8") as metadata_fp:
        for shard_path in shards:
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                column_names = list(batch.schema.names)
                if "audio" not in column_names:
                    raise RuntimeError(
                        f"MSP mirror shard missing `audio` column: {shard_path.name}"
                    )
                if "file" not in column_names:
                    raise RuntimeError(
                        f"MSP mirror shard missing `file` column: {shard_path.name}"
                    )
                columns = {
                    name: batch.column(index) for index, name in enumerate(column_names)
                }
                non_audio_columns = [name for name in column_names if name != "audio"]
                for row_index in range(batch.num_rows):
                    audio_cell = columns["audio"][row_index].as_py()
                    if not isinstance(audio_cell, dict):
                        raise RuntimeError(
                            "MSP mirror audio payload is not a dict in "
                            f"{shard_path.name}:{row_index}."
                        )
                    file_value = columns["file"][row_index].as_py()
                    audio_path_value = audio_cell.get("path")
                    selected_name = file_value or audio_path_value
                    if not isinstance(selected_name, str) or not selected_name:
                        raise RuntimeError(
                            "MSP mirror row missing filename in "
                            f"{shard_path.name}:{row_index}."
                        )
                    relpath = _safe_relative_path(selected_name)
                    output_audio_path = audio_dir / relpath
                    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

                    embedded_bytes = audio_cell.get("bytes")
                    if not isinstance(embedded_bytes, (bytes, bytearray, memoryview)):
                        raise RuntimeError(
                            "MSP mirror row missing embedded bytes in "
                            f"{shard_path.name}:{row_index}."
                        )
                    audio_bytes = bytes(embedded_bytes)
                    if (
                        output_audio_path.is_file()
                        and output_audio_path.stat().st_size == len(audio_bytes)
                    ):
                        files_reused += 1
                    else:
                        partial_path = output_audio_path.with_suffix(
                            output_audio_path.suffix + ".partial"
                        )
                        with partial_path.open("wb") as audio_fp:
                            audio_fp.write(audio_bytes)
                        os.replace(partial_path, output_audio_path)
                        files_written += 1

                    metadata_record: dict[str, Any] = {
                        name: _json_safe(columns[name][row_index].as_py())
                        for name in non_audio_columns
                    }
                    metadata_record["audio_relpath"] = relpath.as_posix()
                    metadata_record["audio_num_bytes"] = len(audio_bytes)
                    metadata_record["source_parquet_shard"] = shard_path.name
                    metadata_fp.write(
                        json.dumps(metadata_record, ensure_ascii=False) + "\n"
                    )
                    rows_seen += 1
    os.replace(metadata_tmp_path, metadata_jsonl_path)
    return rows_seen, files_written, files_reused


def prepare_msp_podcast_from_hf_mirror(
    *,
    dataset_root: Path,
    repo_id: str = DEFAULT_MSP_MIRROR_REPO_ID,
    revision: str = DEFAULT_MSP_MIRROR_REVISION,
    max_workers: int = DEFAULT_MSP_MIRROR_MAX_WORKERS,
    batch_size: int = DEFAULT_MSP_MIRROR_BATCH_SIZE,
    token: str | None = None,
) -> MspPodcastMirrorArtifacts:
    """Downloads and prepares one MSP mirror dataset root.

    Args:
        dataset_root: Destination dataset root.
        repo_id: Hugging Face dataset repository id.
        revision: Repository revision/tag/commit.
        max_workers: Snapshot download workers.
        batch_size: Parquet extraction batch size.
        token: Optional Hugging Face token (defaults to ``HF_TOKEN`` env).

    Returns:
        MspPodcastMirrorArtifacts with resolved paths and extraction stats.
    """

    resolved_token = token if token is not None else (os.getenv("HF_TOKEN") or None)
    root = dataset_root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    repo_dir = root / DEFAULT_MSP_MIRROR_REPO_SUBDIR
    audio_dir = root / DEFAULT_MSP_MIRROR_AUDIO_SUBDIR
    metadata_jsonl_path = root / DEFAULT_MSP_MIRROR_METADATA_FILE
    labels_csv_path = root / DEFAULT_MSP_MIRROR_LABELS_FILE
    manifest_json_path = root / DEFAULT_MSP_MIRROR_MANIFEST_FILE
    repo_dir.mkdir(parents=True, exist_ok=True)

    remote_files, commit_sha = _download_snapshot(
        repo_id=repo_id,
        revision=revision,
        repo_dir=repo_dir,
        max_workers=max_workers,
        token=resolved_token,
    )
    rows_seen, files_written, files_reused = _extract_metadata_and_audio(
        repo_dir=repo_dir,
        audio_dir=audio_dir,
        metadata_jsonl_path=metadata_jsonl_path,
        batch_size=batch_size,
    )
    labels_written, dropped_rows = generate_msp_labels_csv_from_metadata_jsonl(
        metadata_jsonl_path=metadata_jsonl_path,
        labels_csv_path=labels_csv_path,
    )
    if labels_written == 0:
        raise RuntimeError(
            "MSP mirror label generation produced zero rows; cannot build manifest."
        )

    manifest_payload: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "source": {
            "repo_id": repo_id,
            "revision": revision,
            "commit_sha": commit_sha,
            "remote_file_count": len(remote_files),
        },
        "artifacts": {
            "dataset_root": str(root),
            "repo_dir": str(repo_dir),
            "audio_dir": str(audio_dir),
            "metadata_jsonl": str(metadata_jsonl_path),
            "labels_csv": str(labels_csv_path),
        },
        "stats": {
            "rows_seen": rows_seen,
            "files_written": files_written,
            "files_reused": files_reused,
            "labels_written": labels_written,
            "dropped_rows": dropped_rows,
        },
    }
    manifest_tmp_path = manifest_json_path.with_suffix(
        manifest_json_path.suffix + ".tmp"
    )
    with manifest_tmp_path.open("w", encoding="utf-8") as manifest_fp:
        json.dump(manifest_payload, manifest_fp, indent=2, ensure_ascii=False)
        manifest_fp.write("\n")
    os.replace(manifest_tmp_path, manifest_json_path)

    return MspPodcastMirrorArtifacts(
        dataset_root=root,
        repo_dir=repo_dir,
        audio_dir=audio_dir,
        metadata_jsonl_path=metadata_jsonl_path,
        labels_csv_path=labels_csv_path,
        manifest_json_path=manifest_json_path,
        rows_seen=rows_seen,
        files_written=files_written,
        files_reused=files_reused,
        labels_written=labels_written,
        dropped_rows=dropped_rows,
        commit_sha=commit_sha,
    )
