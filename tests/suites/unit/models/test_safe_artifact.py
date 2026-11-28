"""Verify the safe fast-model format at its writer and reader interfaces."""

from __future__ import annotations

import builtins
import hashlib
import pickle
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ser._internal.analysis.artifact import open_model
from ser._internal.analysis.errors import AnalysisModelLoadError, AnalysisModelNotConformingError
from ser._internal.models import artifact_safe
from ser._internal.models.artifact_envelope import build_model_artifact
from ser.api import UtteranceAnalyzer
from ser.runtime.contracts import LicenceRecord, TrainingAudioForm


def _fit() -> Pipeline:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(48, 5))
    y = ["calm", "happy", "sad"] * 16
    classifier = MLPClassifier(hidden_layer_sizes=(4, 3), solver="lbfgs", random_state=42)
    model = make_pipeline(StandardScaler(), classifier)
    model.fit(train, y)
    return model


@pytest.fixture
def safe_bytes() -> bytes:
    return artifact_safe.encode_model(
        _fit(),
        model_id="fixture-fast",
        training_audio=TrainingAudioForm(16000, 1, True),
        datasets=(
            LicenceRecord("fixture", "cc-by-nc-4.0", "non-commercial", "https://example.org"),
        ),
        split_ledger_digest="a" * 64,
        ser_version="1.0.0",
    )


@pytest.mark.parametrize(
    ("scaled", "class_count", "dtype"), [(True, 3, "float64"), (False, 2, "float32")]
)
def test_converted_predictions_match_exactly_without_unsafe_loads(
    scaled: bool, class_count: int, dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(42, 4)).astype(dtype)
    y = [("a", "b", "c")[index % class_count] for index in range(len(x))]
    clf = MLPClassifier(
        hidden_layer_sizes=(4,),
        solver="lbfgs",
        random_state=17,
        activation="logistic",
        max_iter=1000,
    )
    model = make_pipeline(StandardScaler(), clf) if scaled else clf
    model.fit(x, y)
    legacy = pickle.loads(pickle.dumps(model))
    probe = rng.normal(size=(29, 4)).astype(dtype)
    expected_labels, expected_proba = legacy.predict(probe), legacy.predict_proba(probe)
    data = artifact_safe.encode_model(legacy, model_id="fixture")

    def forbid(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unsafe deserializer used")

    original_import = builtins.__import__

    def restricted_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"pickle", "joblib", "sklearn"}:
            forbid()
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", restricted_import)
        patch.setattr(pickle, "load", forbid)
        patch.setattr(pickle, "loads", forbid)
        patch.setattr(joblib, "load", forbid)
        patch.setattr(np, "load", forbid)
        restored = artifact_safe.verify_artifact(data).classifier
        np.testing.assert_array_equal(restored.predict(probe), expected_labels)
        np.testing.assert_array_equal(restored.predict_proba(probe), expected_proba)


def test_every_single_byte_is_covered_by_verification(safe_bytes: bytes) -> None:
    for i in range(len(safe_bytes)):
        tampered = bytearray(safe_bytes)
        tampered[i] ^= 1
        with pytest.raises(AnalysisModelLoadError, match="digest|format"):
            open_model(
                bytes(tampered), profile="fast", expected_sha256=None, allow_legacy_baseline=False
            )


def test_newer_version_refused_explicitly(safe_bytes: bytes) -> None:
    changed = safe_bytes[:10] + (2).to_bytes(4, "big") + safe_bytes[14:-32]
    changed += hashlib.sha256(changed).digest()
    with pytest.raises(AnalysisModelLoadError, match="newer.*version 2"):
        open_model(changed, profile="fast", expected_sha256=None, allow_legacy_baseline=False)


def test_consumer_can_refuse_a_disallowed_licence_before_loading(
    safe_bytes: bytes, tmp_path: Path
) -> None:
    model_path = tmp_path / "model.ser"
    model_path.write_bytes(safe_bytes)
    manifest = UtteranceAnalyzer.inspect_model(profile="fast", model_path=model_path)
    assert manifest.provenance is not None
    assert not all(record.licence_class == "open" for record in manifest.provenance.datasets)
    assert manifest.identity.model_sha256 == hashlib.sha256(safe_bytes).hexdigest()


def test_payload_digest_is_checked_even_if_trailer_is_recomputed(safe_bytes: bytes) -> None:
    changed = safe_bytes[:-33] + b" "
    changed += hashlib.sha256(changed).digest()
    with pytest.raises(AnalysisModelLoadError, match="payload digest"):
        open_model(changed, profile="fast", expected_sha256=None, allow_legacy_baseline=False)


def test_incomplete_conversion_cannot_bypass_analysis_licence_gate() -> None:
    data = artifact_safe.encode_model(_fit(), model_id="historical")
    with pytest.raises(AnalysisModelLoadError, match="licence"):
        open_model(data, profile="fast", expected_sha256=None, allow_legacy_baseline=True)


def test_legacy_opt_in_warns_without_a_path(tmp_path: Path) -> None:
    path = tmp_path / "private-model.pkl"
    model = _fit()
    path.write_bytes(pickle.dumps(build_model_artifact(model, 5, 48, ["calm", "happy", "sad"])))
    raw = path.read_bytes()
    with pytest.raises(AnalysisModelNotConformingError):
        open_model(raw, profile="fast", expected_sha256=None, allow_legacy_baseline=False)
    with pytest.warns(DeprecationWarning, match="legacy.*deprecated") as records:
        manifest, _ = open_model(
            raw, profile="fast", expected_sha256=None, allow_legacy_baseline=True
        )
    assert path.name not in str(records[0].message)
    assert manifest.baseline
