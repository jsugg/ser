"""Tests for CREMA-D dataset adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from ser._internal.data.adapters.crema_d import (
    CREMA_D_CORPUS_ID,
    CREMA_D_DATASET_LICENSE_ID,
    CREMA_D_DATASET_POLICY_ID,
    CremaDDatasetIntegrityError,
    build_crema_d_manifest_jsonl,
    build_crema_d_utterances,
    validate_crema_d_audio_files,
)
from ser._internal.data.ontology import LabelOntology


def _ontology() -> LabelOntology:
    return LabelOntology(
        ontology_id="default_v1",
        allowed_labels=frozenset({"happy", "sad", "angry"}),
    )


def _write_wav(path: Path, *, frames: int = 4) -> None:
    """Writes a minimal valid mono WAV fixture."""
    sf.write(path, np.linspace(-0.5, 0.5, frames, dtype=np.float32), 16_000)


def test_build_crema_d_utterances_maps_codes_and_metadata(tmp_path: Path) -> None:
    """CREMA-D adapter should parse filename code and attach metadata."""
    audio_root = tmp_path / "AudioWAV"
    audio_root.mkdir(parents=True, exist_ok=True)
    _write_wav(audio_root / "1001_IEO_HAP_LO.wav")
    _write_wav(audio_root / "1002_IEO_SAD_LO.wav")

    utterances = build_crema_d_utterances(
        dataset_root=tmp_path,
        dataset_glob_pattern="AudioWAV/**/*.wav",
        emotion_code_map={"HAP": "happy", "SAD": "sad"},
        default_language="en",
        ontology=_ontology(),
        max_failed_file_ratio=0.5,
    )

    assert utterances is not None
    assert len(utterances) == 2
    assert {item.corpus for item in utterances} == {CREMA_D_CORPUS_ID}
    assert {item.dataset_policy_id for item in utterances} == {CREMA_D_DATASET_POLICY_ID}
    assert {item.dataset_license_id for item in utterances} == {CREMA_D_DATASET_LICENSE_ID}


def test_validate_crema_d_audio_accepts_a_one_frame_wav(tmp_path: Path) -> None:
    """Metadata validation should accept a genuinely decodable tiny WAV."""
    audio_path = tmp_path / "AudioWAV" / "1001_IEO_HAP_LO.wav"
    audio_path.parent.mkdir(parents=True)
    _write_wav(audio_path, frames=1)

    files = validate_crema_d_audio_files(
        dataset_root=tmp_path,
        dataset_glob_pattern="AudioWAV/**/*.wav",
    )

    assert files == (audio_path,)


def test_validate_crema_d_audio_logs_integrity_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CREMA-D validation should report progress around metadata scans."""
    audio_root = tmp_path / "AudioWAV"
    audio_root.mkdir(parents=True)
    _write_wav(audio_root / "1001_IEO_HAP_LO.wav", frames=1)
    _write_wav(audio_root / "1002_IEO_HAP_LO.wav", frames=1)
    caplog.set_level(logging.INFO, logger="ser._internal.data.adapters.crema_d")

    files = validate_crema_d_audio_files(
        dataset_root=tmp_path,
        dataset_glob_pattern="AudioWAV/**/*.wav",
    )

    messages = [record.getMessage() for record in caplog.records]
    assert len(files) == 2
    assert any(message.startswith("DATASET_INTEGRITY_START") for message in messages)
    assert any(message.startswith("DATASET_INTEGRITY_PROGRESS") for message in messages)
    assert any(message.startswith("DATASET_INTEGRITY_DONE") for message in messages)


def test_crema_d_lfs_pointer_aborts_before_manifest_write(tmp_path: Path) -> None:
    """An unmaterialized LFS object must never be registered as ready audio."""
    audio_path = tmp_path / "AudioWAV" / "1077_WSI_ANG_XX.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:be5c849653d28aaed49fbca687812f5352f295b1e1a66c269e4a3f2b7ae46489\n"
        b"size 85462\n"
    )
    manifest_path = tmp_path / "manifests" / "crema-d.jsonl"

    with pytest.raises(CremaDDatasetIntegrityError, match="Git LFS pointer"):
        build_crema_d_manifest_jsonl(
            dataset_root=tmp_path,
            dataset_glob_pattern="AudioWAV/**/*.wav",
            emotion_code_map={"ANG": "angry"},
            default_language="en",
            ontology=_ontology(),
            max_failed_file_ratio=0.5,
            output_path=manifest_path,
        )

    assert not manifest_path.exists()
