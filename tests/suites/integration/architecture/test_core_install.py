"""Exercise the plain-install contract in the full development environment."""

import os
import subprocess
import sys
from pathlib import Path

# Spawned analysis workers also load sitecustomize from the temporary PYTHONPATH.
_CORE_SITECUSTOMIZE = """
import sys
from importlib.machinery import PathFinder

blocked = {
    "torch", "torchaudio", "transformers", "whisper", "stable_whisper",
    "faster_whisper", "ctranslate2", "onnxruntime", "funasr", "demucs",
    "huggingface_hub",
}

class CoreFinder(PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            return None
        return super().find_spec(fullname, path, target)

sys.meta_path[sys.meta_path.index(PathFinder)] = CoreFinder
"""

_ANALYZE = """
import importlib.util
import sys
from pathlib import Path

from sitecustomize import blocked
assert all(importlib.util.find_spec(name) is None for name in blocked)
from ser.api import AnalysisConfig, UtteranceAnalyzer
from ser._internal.features.feature_extractor import extract_feature_from_signal
from tests.suites.unit.analysis._stand_ins import CANONICAL_RATE_HZ, tone, write_model

clip = tone(2)
width = extract_feature_from_signal(clip, CANONICAL_RATE_HZ).size
artifact = write_model(Path("stand-in.pkl"), feature_size=width)
for config in (AnalysisConfig(isolation="inline", timeout_floor_s=120),
               AnalysisConfig(timeout_floor_s=120, worker_start_timeout_s=120)):
    with UtteranceAnalyzer.load(profile="fast", model_path=artifact,
                                allow_legacy_baseline=True, config=config) as analyzer:
        record = analyzer.analyze(clip, sample_rate_hz=CANONICAL_RATE_HZ)
        assert record["status"] == "ok", record
    assert not blocked.intersection(sys.modules), blocked.intersection(sys.modules)
"""


def test_core_analysis_and_doctor(repo_root: Path, tmp_path: Path) -> None:
    (tmp_path / "sitecustomize.py").write_text(_CORE_SITECUSTOMIZE, encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, (tmp_path, repo_root))),
        "SER_DATA_DIR": str(tmp_path / "data"),
        "SER_CACHE_DIR": str(tmp_path / "cache"),
        "DATASET_FOLDER": str(tmp_path / "dataset"),
        "SER_DATASET_MANIFESTS": "",
        "SER_DATASET_RECIPE": "",
    }
    analysis = subprocess.run(
        [sys.executable, "-c", _ANALYZE], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert analysis.returncode == 0, analysis.stderr

    for backend in ("stable_whisper", "faster_whisper"):
        doctor = subprocess.run(
            [sys.executable, "-m", "ser", "doctor", "--profile", "fast"],
            cwd=tmp_path,
            env={**env, "WHISPER_BACKEND": backend},
            capture_output=True,
            text=True,
        )
        output = doctor.stdout + doctor.stderr
        assert doctor.returncode == 2, output
        assert f"Transcription backend '{backend}' requires `ser[transcript]`" in output, output
        assert "No module named" not in output, output
