"""Verify the safe fast-model format at its writer and reader interfaces."""

from __future__ import annotations

import base64
import builtins
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import joblib
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ser._internal.analysis.artifact import open_model, read_model_file
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
    changed = safe_bytes[:10] + (2).to_bytes(4, "big") + safe_bytes[14:]
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
    with pytest.warns(FutureWarning, match="legacy.*deprecated") as records:
        manifest = UtteranceAnalyzer.inspect_model(
            profile="fast", model_path=path, allow_legacy_baseline=True
        )
    assert records[0].filename == __file__
    assert path.name not in str(records[0].message)
    assert manifest.baseline


def test_api_import_leaves_scipy_unloaded() -> None:
    subprocess.run(
        [sys.executable, "-c", "import sys, ser.api; assert 'scipy' not in sys.modules"], check=True
    )


@pytest.mark.parametrize(
    "part",
    [
        "required",
        "missing-required",
        "undeclared-calibration",
        "label_set_id",
        "utc",
        "duplicate",
        "older",
        "nonfinite",
        "zero-scale",
    ],
)
def test_invalid_manifest_or_weights(safe_bytes: bytes, part: str) -> None:
    size = int.from_bytes(safe_bytes[14:18], "big")
    manifest = json.loads(safe_bytes[18 : 18 + size])
    payload = json.loads(safe_bytes[18 + size : -32])
    version = 1
    if part == "required":
        manifest["required"] = ["calibration"]
    elif part == "undeclared-calibration":
        manifest["calibration"] = {"temperature": 2}
    elif part == "missing-required":
        manifest.pop("required", None)
    elif part == "label_set_id":
        manifest["label_set_id"] = "unrelated"
    elif part == "utc":
        manifest["provenance"]["created_at"] = "2020-01-01T01:00:00+01:00"
    elif part == "older":
        version = manifest["format_version"] = 0
    elif part in {"nonfinite", "zero-scale"}:
        array = payload["scale"]
        values = np.ones(array["shape"], dtype=array["dtype"])
        values[0] = np.nan if part == "nonfinite" else 0
        array["data"] = base64.b64encode(values.tobytes()).decode()
    raw = json.dumps(payload).encode()
    manifest["payload_sha256"] = hashlib.sha256(raw).hexdigest()
    header = json.dumps(manifest).encode()
    if part == "duplicate":
        header = b'{"profile":"fast",' + header[1:]
    body = (
        artifact_safe.MAGIC
        + version.to_bytes(4, "big")
        + len(header).to_bytes(4, "big")
        + header
        + raw
    )
    with pytest.raises(ValueError):
        artifact_safe.verify_artifact(body + hashlib.sha256(body).digest())


def test_bounded_model_reads(tmp_path: Path) -> None:
    path = tmp_path / "oversize"
    raw = bytes(8 * 1024 * 1024 + 1)
    path.write_bytes(raw)
    with pytest.raises(AnalysisModelLoadError, match="size"):
        read_model_file(path)
    with pytest.raises(ValueError, match="size"):
        artifact_safe.verify_artifact(raw)
    with pytest.raises(AnalysisModelLoadError):
        read_model_file("/dev/null")


@pytest.mark.parametrize("activation", ["identity", "tanh"])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_scaled_parity(activation: str, dtype: str) -> None:
    model = _fit()
    clf = model.steps[-1][1]
    clf.activation = activation
    clf.coefs_[0] = np.asfortranarray(clf.coefs_[0])
    probe = np.random.default_rng(404).normal(size=(32, 5)).astype(dtype)
    classifier = artifact_safe.verify_artifact(
        artifact_safe.encode_model(model, model_id="parity")
    ).classifier
    np.testing.assert_array_equal(classifier.predict_proba(probe), model.predict_proba(probe))


def test_binary_tie_and_extra_pipeline_step() -> None:
    clf = MLPClassifier(hidden_layer_sizes=(2,), solver="lbfgs", random_state=1).fit(
        [[0], [1]], ["a", "b"]
    )
    clf.intercepts_[-1].fill(0)
    clf.coefs_[-1].fill(0)
    classifier = artifact_safe.verify_artifact(
        artifact_safe.encode_model(clf, model_id="tie")
    ).classifier
    assert classifier.predict([[1]]).tolist() == ["a"]
    pipeline = _fit()
    pipeline.steps.append(("extra", "passthrough"))
    with pytest.raises(ValueError):
        artifact_safe.encode_model(pipeline, model_id="extra")


def test_fast_loader_needs_no_legacy_modules(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    model = MLPClassifier(hidden_layer_sizes=(3,), solver="lbfgs", random_state=42)
    model.fit(rng.normal(size=(12, 12)), ["calm", "happy"] * 6)
    path = tmp_path / "subset.ser"
    path.write_bytes(
        artifact_safe.encode_model(
            model,
            model_id="subset",
            enabled_features=("chroma",),
            training_audio=TrainingAudioForm(16000, 1, True),
            datasets=(LicenceRecord("fixture", "MIT", "open", "fixture"),),
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from pathlib import Path
from ser._internal.analysis.artifact import open_model
from ser._internal.analysis.backend import BackendLoadRequest

class RefuseLegacyImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"ser._internal.analysis.legacy", "ser._internal.models.artifact_envelope", "joblib", "sklearn"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RefuseLegacyImports())
from ser._internal.analysis.fast import load_fast
manifest, opened = open_model(Path(sys.argv[1]).read_bytes(), profile="fast",
                              expected_sha256=None, allow_legacy_baseline=False)
load_fast(BackendLoadRequest("safe", opened.payload, manifest, None)).close()
""",
            str(path),
        ],
        check=True,
    )


def test_conversion_preserves_destination_created_during_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ser._internal.models import artifact_cli

    legacy, destination = tmp_path / "legacy.pkl", tmp_path / "safe.ser"
    model = _fit()
    legacy.write_bytes(pickle.dumps(build_model_artifact(model, 5, 48, ["calm", "happy", "sad"])))

    def concurrent_writer(*args: object, **kwargs: object) -> bytes:
        destination.write_bytes(b"another conversion")
        return b"this conversion"

    monkeypatch.setattr(artifact_cli, "encode_model", concurrent_writer)
    assert (
        artifact_cli.run_models_command(
            ["convert", "--trust-pickle", str(legacy), str(destination)]
        )
        == 1
    )
    assert destination.read_bytes() == b"another conversion"
