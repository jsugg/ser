"""Check real subprocess conversion and manifest inspection."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

from ser._internal.models.artifact_envelope import build_model_artifact
from ser._internal.models.artifact_safe import encode_model, verify_artifact
from ser.runtime.contracts import LicenceRecord, TrainingAudioForm


def _models(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ser", "models", *argv],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )


def test_convert_and_show_a_legacy_model(tmp_path: Path) -> None:
    legacy, safe = tmp_path / "legacy.pkl", tmp_path / "model.ser"
    x = np.random.default_rng(20).normal(size=(24, 3))
    labels = ["calm", "happy"] * 12
    model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(3,), random_state=20).fit(x, labels)
    legacy.write_bytes(pickle.dumps(build_model_artifact(model, 3, len(x), labels)))

    no_opt_in = _models("convert", str(legacy), str(safe))
    assert no_opt_in.returncode != 0
    assert "unpickles" in no_opt_in.stderr and "trust" in no_opt_in.stderr
    assert not safe.exists()
    converted = _models("convert", "--trust-pickle", str(legacy), str(safe))
    assert converted.returncode == 0, converted.stderr
    data = safe.read_bytes()
    probe = np.random.default_rng(42).normal(size=(18, 3))
    restored = verify_artifact(data).classifier
    expected_labels = model.predict(probe)
    assert isinstance(expected_labels, np.ndarray)
    np.testing.assert_array_equal(restored.predict(probe), expected_labels)
    np.testing.assert_array_equal(restored.predict_proba(probe), model.predict_proba(probe))

    shown = _models("show", str(safe))
    assert shown.returncode == 0, shown.stderr
    result = json.loads(shown.stdout)
    assert result["verified"] is True
    assert result["manifest"]["labels"] == ["calm", "happy"]
    assert result["manifest"]["provenance"]["datasets"] == []
    assert "coefs" not in shown.stdout

    safe.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    failed = _models("show", str(safe))
    assert failed.returncode != 0 and "digest" in failed.stderr


def test_show_prints_the_verified_licence_record(tmp_path: Path) -> None:
    path = tmp_path / "model.ser"
    x = np.random.default_rng(1).normal(size=(24, 3))
    model = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(3,), random_state=1)
    model.fit(x, ["calm", "happy"] * 12)
    path.write_bytes(
        encode_model(
            model,
            model_id="fixture",
            training_audio=TrainingAudioForm(16000, 1, True),
            datasets=(
                LicenceRecord("fixture", "cc-by-nc-4.0", "non-commercial", "https://example.org"),
            ),
            split_ledger_digest="a" * 64,
            ser_version="1.0.0",
        )
    )
    shown = _models("show", str(path))
    assert shown.returncode == 0, shown.stderr
    result = json.loads(shown.stdout)
    assert result["verified"] is True
    assert result["manifest"]["provenance"]["datasets"] == [
        {
            "name": "fixture",
            "licence": "cc-by-nc-4.0",
            "licence_class": "non-commercial",
            "source": "https://example.org",
        }
    ]
