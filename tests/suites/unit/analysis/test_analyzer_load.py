"""Loading: one verified model file, an inspectable licence record, baselines on opt-in."""

from __future__ import annotations

import hashlib
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from tests.suites.unit.analysis import _scripted_backend
from tests.suites.unit.analysis._contract import (
    IoGuard,
    assert_valid_record,
    load_artifact,
    load_baseline,
)
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, LABELS, tone
from tests.suites.unit.analysis._synthetic_artifacts import (
    RESEARCH_DATASET,
    SCRIPTED_LABELS,
    TRAP,
    payload_of,
    write_artifact,
    write_booby_trapped_pickle,
)

import ser.config as config
from ser.api import (
    AnalysisError,
    AnalysisModelLoadError,
    AnalysisModelNotConformingError,
    AnalysisModelUnavailableError,
    AnalysisUsageError,
    LicenceRecord,
    ModelProvenance,
    ProfileName,
    TrainingAudioForm,
    UtteranceAnalyzer,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.usefixtures("scripted")
def test_safe_artifact_manifest_is_its_verified_identity_and_licence(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser")

    analyzer = load_artifact(path)
    record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    manifest = analyzer.model_manifest
    assert manifest.identity.profile == "fast"
    assert manifest.identity.model_id == "test-model"
    assert manifest.identity.model_sha256 == _sha256(path.read_bytes())
    assert manifest.identity.label_set_id == "test-labels/v1"
    assert manifest.identity.labels == SCRIPTED_LABELS
    assert manifest.artifact_format == "safe"
    assert manifest.baseline is False
    assert manifest.training_audio == TrainingAudioForm(16_000, 1, True)
    assert manifest.provenance == ModelProvenance(
        payload_sha256=_sha256(payload_of(path)),
        encoder=LicenceRecord("test-encoder", "MIT", "open", "generated"),
        datasets=(LicenceRecord("test-dataset", "custom", "research-only", "generated"),),
    )
    assert record["status"] == "ok"
    assert_valid_record(record, manifest)


@pytest.mark.usefixtures("scripted")
def test_inspection_reads_the_licence_without_loading_the_model(
    tmp_path: Path, spies: Counter[str]
) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"name": "inspected"})

    manifest = UtteranceAnalyzer.inspect_model(profile="fast", model_path=path)

    assert manifest.provenance is not None
    assert [d.licence_class for d in manifest.provenance.datasets] == ["research-only"]
    assert spies["start_worker"] == 0
    assert _scripted_backend.LOADS["inspected"] == 0


@pytest.mark.usefixtures("scripted")
def test_allow_list_hand_off_pins_the_inspected_bytes(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"labels": {"a": 1.0, "b": 0.0, "c": 0.0}})
    allowed = {"open", "research-only"}

    manifest = UtteranceAnalyzer.inspect_model(profile="fast", model_path=path)
    assert manifest.provenance is not None
    approved = {d.licence_class for d in manifest.provenance.datasets} <= allowed
    assert approved
    pinned = manifest.identity.model_sha256
    write_artifact(path, script={"labels": {"a": 0.0, "b": 1.0, "c": 0.0}})

    with pytest.raises(AnalysisModelLoadError):
        UtteranceAnalyzer.load(profile="fast", model_path=path, expected_sha256=pinned)


_UNVERIFIABLE: dict[str, dict[str, Any]] = {
    "tampered": {"provenance": "tampered"},
    "missing": {"provenance": "missing"},
    "encoder-unrecorded": {"provenance": "encoder-unrecorded"},
    "no-datasets": {"datasets": ()},
    "dataset-without-id": {"datasets": ({**RESEARCH_DATASET, "name": ""},)},
    "dataset-without-licence": {"datasets": ({**RESEARCH_DATASET, "licence": ""},)},
    "dataset-without-class": {"datasets": ({**RESEARCH_DATASET, "licence_class": ""},)},
}


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("allow_legacy_baseline", [False, True])
@pytest.mark.parametrize("artifact", list(_UNVERIFIABLE.values()), ids=list(_UNVERIFIABLE))
def test_unverifiable_safe_artifacts_never_load(
    tmp_path: Path, artifact: dict[str, Any], allow_legacy_baseline: bool
) -> None:
    path = write_artifact(tmp_path / "model.ser", **artifact)

    for attempt in (UtteranceAnalyzer.inspect_model, UtteranceAnalyzer.load):
        with pytest.raises(AnalysisModelLoadError) as raised:
            attempt(profile="fast", model_path=path, allow_legacy_baseline=allow_legacy_baseline)
        assert not isinstance(raised.value, AnalysisModelNotConformingError)


@pytest.mark.usefixtures("scripted")
def test_a_model_without_a_pretrained_encoder_says_so(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", provenance="no-pretrained-encoder")

    manifest = UtteranceAnalyzer.inspect_model(profile="fast", model_path=path)

    assert manifest.provenance is not None
    assert manifest.provenance.encoder == "none"


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize("missing", ["model_id", "label_set_id"])
def test_safe_artifacts_must_name_model_and_label_set(tmp_path: Path, missing: str) -> None:
    if missing == "model_id":
        path = write_artifact(tmp_path / "model.ser", model_id=None)
    else:
        path = write_artifact(tmp_path / "model.ser", label_set_id=None)

    with pytest.raises(AnalysisModelLoadError):
        load_artifact(path)


@pytest.mark.usefixtures("scripted")
def test_an_artifact_for_another_profile_is_a_load_error(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", profile="medium")

    with pytest.raises(AnalysisModelLoadError):
        load_artifact(path)


@pytest.mark.usefixtures("scripted")
@pytest.mark.parametrize(
    "training_audio",
    [
        None,
        {"sample_rate_hz": 44_100, "channels": 1, "canonical_conversion": True},
        {"sample_rate_hz": 16_000, "channels": 2, "canonical_conversion": True},
        {"sample_rate_hz": 16_000, "channels": 1, "canonical_conversion": False},
    ],
    ids=["unknown", "44.1k", "stereo", "other-conversion"],
)
def test_non_canonical_training_audio_is_a_baseline_on_opt_in_only(
    tmp_path: Path, training_audio: dict[str, object] | None
) -> None:
    path = write_artifact(tmp_path / "model.ser", training_audio=training_audio)
    sha256 = _sha256(path.read_bytes())

    with pytest.raises(AnalysisModelNotConformingError):
        load_artifact(path)
    with UtteranceAnalyzer.load(
        profile="fast", model_path=path, allow_legacy_baseline=True
    ) as analyzer:
        assert analyzer.model_manifest.baseline is True
        assert analyzer.model_manifest.identity.model_id == f"baseline:test-model@{sha256}"
        record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)
    assert record["model_id"] == f"baseline:test-model@{sha256}"


@pytest.mark.usefixtures("scripted")
def test_a_pickle_is_refused_before_it_is_unpickled(tmp_path: Path) -> None:
    path = write_booby_trapped_pickle(tmp_path / "model.pkl")
    TRAP["fired"] = False

    with pytest.raises(AnalysisModelNotConformingError):
        load_artifact(path)
    with pytest.raises(AnalysisModelNotConformingError):
        UtteranceAnalyzer.inspect_model(profile="fast", model_path=path)

    assert TRAP["fired"] is False


def test_legacy_pickle_loads_only_as_a_baseline_with_exact_ids(fast_model_path: Path) -> None:
    sha256 = _sha256(fast_model_path.read_bytes())
    canonical_labels = json.dumps(sorted(set(LABELS)), separators=(",", ":")).encode()

    with pytest.raises(AnalysisModelNotConformingError):
        UtteranceAnalyzer.load(profile="fast", model_path=fast_model_path)
    manifest = load_baseline(fast_model_path).model_manifest

    assert manifest.artifact_format == "legacy-pickle"
    assert manifest.baseline is True
    assert manifest.training_audio is None
    assert manifest.provenance is None
    assert manifest.identity.model_sha256 == sha256
    assert manifest.identity.model_id == f"baseline:fast-handcrafted@{sha256}"
    assert manifest.identity.label_set_id == "labels:" + _sha256(canonical_labels)
    assert manifest.identity.labels == LABELS


def test_legacy_model_revision_names_the_baseline(revision_model_path: Path) -> None:
    sha256 = _sha256(revision_model_path.read_bytes())

    manifest = load_baseline(revision_model_path).model_manifest

    assert manifest.identity.model_id == f"baseline:rev-7@{sha256}"


def test_baseline_ids_carry_the_digest_and_label_set_ids_follow_the_labels(
    fast_model_path: Path, other_fast_model_path: Path, seven_label_model_path: Path
) -> None:
    first = load_baseline(fast_model_path).model_manifest.identity
    other = load_baseline(other_fast_model_path).model_manifest.identity
    seven = load_baseline(seven_label_model_path).model_manifest.identity

    assert other.model_sha256 != first.model_sha256
    assert other.model_id != first.model_id
    assert other.label_set_id == first.label_set_id
    assert seven.label_set_id != first.label_set_id


def test_missing_model_file_fails_at_load(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model-token-5d1e" / "ser_model.pkl"

    with pytest.raises(AnalysisModelUnavailableError) as raised:
        load_baseline(missing)

    assert isinstance(raised.value, AnalysisError)
    assert "token-5d1e" not in "".join(traceback.format_exception(raised.value))


@pytest.mark.usefixtures("reset_ambient_settings")
def test_no_configured_model_fails_at_load_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: IoGuard
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    settings = config.reload_settings()

    with guard.watching(), pytest.raises(AnalysisModelUnavailableError):
        UtteranceAnalyzer.load(profile="fast", settings=settings)

    assert guard.network == []


@pytest.mark.parametrize("kind", ["other-profile", "feature-size", "corrupt"])
def test_incompatible_legacy_files_are_load_errors_with_clean_tracebacks(
    incompatible_model_paths: dict[str, Path], kind: str
) -> None:
    with pytest.raises(AnalysisModelLoadError) as raised:
        load_baseline(incompatible_model_paths[kind])

    error = raised.value
    assert error.__cause__ is None and error.__suppress_context__ is True
    assert "token-91c2" not in "".join(traceback.format_exception(error))


@pytest.mark.usefixtures("scripted")
def test_a_backend_that_cannot_start_is_a_load_error(tmp_path: Path) -> None:
    path = write_artifact(tmp_path / "model.ser", script={"fail_load": True})

    with pytest.raises(AnalysisModelLoadError) as raised:
        load_artifact(path)

    assert _scripted_backend.SECRET not in "".join(traceback.format_exception(raised.value))


@pytest.mark.parametrize("profile", ["accurate", "accurate-research"])
def test_profile_without_an_analysis_backend_is_a_usage_error(
    fast_model_path: Path, profile: ProfileName
) -> None:
    with pytest.raises(AnalysisUsageError):
        UtteranceAnalyzer.load(
            profile=profile, model_path=fast_model_path, allow_legacy_baseline=True
        )


@pytest.mark.usefixtures("scripted")
def test_a_second_backend_serves_its_own_profile(tmp_path: Path) -> None:
    path = write_artifact(
        tmp_path / "medium.ser", profile="medium", backend_id="scripted-medium", model_id="m-1"
    )

    with UtteranceAnalyzer.load(profile="medium", model_path=path) as analyzer:
        record = analyzer.analyze(tone(1.0), sample_rate_hz=CANONICAL_RATE_HZ)

    assert analyzer.model_manifest.identity.profile == "medium"
    assert record["status"] == "ok" and record["model_id"] == "m-1"
    assert_valid_record(record, analyzer.model_manifest)
