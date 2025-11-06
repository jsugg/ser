import contextlib
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_dummy_colored = types.ModuleType("colored")
_dummy_colored.attr = lambda name: ""
_dummy_colored.bg = lambda color: ""
_dummy_colored.fg = lambda color: ""
sys.modules.setdefault("colored", _dummy_colored)

class _AutoHalo:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False

_dummy_halo = types.ModuleType("halo")
_dummy_halo.Halo = _AutoHalo
sys.modules.setdefault("halo", _dummy_halo)

class _Array(list):
    def tolist(self):
        return list(self)

    def __itruediv__(self, other):
        for index, value in enumerate(self):
            self[index] = value / other
        return self

_dummy_numpy = types.ModuleType("numpy")
_dummy_numpy.ndarray = _Array
_dummy_numpy.float32 = float
_dummy_numpy.bool_ = bool

def _as_array(data):
    return _Array(data) if not isinstance(data, _Array) else data

def _array(data, dtype=None):
    return _Array(list(data))

_dummy_numpy.array = _array
def _zeros(n, dtype=None):
    return _Array(0.0 for _ in range(int(n)))

_dummy_numpy.zeros = _zeros
def _zeros_like(data):
    return _Array(0.0 for _ in data)

_dummy_numpy.zeros_like = _zeros_like
def _abs(data):
    return _Array(abs(x) for x in data)

_dummy_numpy.abs = _abs
_dummy_numpy.max = lambda data: max(data)
_dummy_numpy.isscalar = lambda obj: isinstance(obj, (int, float, str, bool))

sys.modules.setdefault("numpy", _dummy_numpy)

_dummy_librosa = types.ModuleType("librosa")
_dummy_librosa.load = lambda *args, **kwargs: (_Array([]), 16000)
_dummy_librosa.get_duration = lambda y: float(len(y))
sys.modules.setdefault("librosa", _dummy_librosa)

class _SoundFile:
    def __init__(self, *args, **kwargs):
        self._data = _Array([0.0])
        self.samplerate = 16000
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def read(self, dtype=None):
        return _Array(self._data)

_dummy_soundfile = types.ModuleType("soundfile")
_dummy_soundfile.SoundFile = _SoundFile
sys.modules.setdefault("soundfile", _dummy_soundfile)

class _DummyMLP:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, X, y):
        return self
    def predict(self, X):
        return [0 for _ in range(len(X))]

_dummy_sklearn_metrics = types.ModuleType("sklearn.metrics")
_dummy_sklearn_metrics.accuracy_score = lambda y_true, y_pred: 1.0
sys.modules.setdefault("sklearn.metrics", _dummy_sklearn_metrics)

_dummy_sklearn_nn = types.ModuleType("sklearn.neural_network")
_dummy_sklearn_nn.MLPClassifier = _DummyMLP
sys.modules.setdefault("sklearn.neural_network", _dummy_sklearn_nn)

_dummy_sklearn_model_selection = types.ModuleType("sklearn.model_selection")
_dummy_sklearn_model_selection.train_test_split = lambda X, y, test_size=0.2, random_state=None: (X, X, y, y)
sys.modules.setdefault("sklearn.model_selection", _dummy_sklearn_model_selection)

_dummy_sklearn = types.ModuleType("sklearn")
_dummy_sklearn.metrics = _dummy_sklearn_metrics
_dummy_sklearn.neural_network = _dummy_sklearn_nn
_dummy_sklearn.model_selection = _dummy_sklearn_model_selection
sys.modules.setdefault("sklearn", _dummy_sklearn)

_dummy_dotenv = types.ModuleType("dotenv")
_dummy_dotenv.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", _dummy_dotenv)

_dummy_stable_whisper = types.ModuleType("stable_whisper")
_dummy_stable_whisper.load_model = lambda *args, **kwargs: object()
_dummy_stable_whisper_result = types.ModuleType("stable_whisper.result")
class _DummyWhisperResult:
    segments = []

_dummy_stable_whisper_result.WhisperResult = _DummyWhisperResult
sys.modules.setdefault("stable_whisper", _dummy_stable_whisper)
sys.modules.setdefault("stable_whisper.result", _dummy_stable_whisper_result)

_dummy_whisper_model = types.ModuleType("whisper.model")
class _DummyWhisper:
    def transcribe(self, *args, **kwargs):
        return None

_dummy_whisper_model.Whisper = _DummyWhisper
sys.modules.setdefault("whisper.model", _dummy_whisper_model)

import io
import logging
from typing import Sequence

import pytest

import ser.__main__ as ser_main


@pytest.fixture
def run_cli(monkeypatch):
    """Run the SER CLI with a custom argv list."""

    def _run_cli(args: Sequence[str], *, expect_exit: bool = True) -> tuple[int, str]:
        argv = ["ser", *args]
        monkeypatch.setattr(sys, "argv", argv)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            try:
                ser_main.main()
            except SystemExit as exc:  # pragma: no cover - exercised in tests
                return exc.code, stdout.getvalue()
        if expect_exit:
            raise AssertionError("CLI did not exit as expected")
        return 0, stdout.getvalue()

    return _run_cli


@pytest.fixture(autouse=True)
def _silence_halo(monkeypatch):
    """Replace Halo spinners with a no-op context manager for tests."""

    class _DummyHalo:
        def __init__(self, *args, **kwargs):
            self.text = kwargs.get("text")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ser.utils.timeline_utils.Halo", _DummyHalo, raising=False)
    monkeypatch.setattr("ser.models.emotion_model.Halo", _DummyHalo, raising=False)


@pytest.fixture
def caplog_info(caplog):
    caplog.set_level(logging.INFO)
    return caplog
