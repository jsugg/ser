"""Tests for torch runtime inference helpers."""

from __future__ import annotations

from contextlib import nullcontext

import pytest

import ser.utils.torch_inference as torch_inference


class _FakeCuda:
    """Minimal CUDA capability stub."""

    def __init__(self, *, available: bool, bf16_supported: bool) -> None:
        self._available = available
        self._bf16_supported = bf16_supported

    def is_available(self) -> bool:
        """Returns configured CUDA availability."""
        return self._available

    def is_bf16_supported(self) -> bool:
        """Returns configured CUDA bf16 support."""
        return self._bf16_supported


class _FakeMps:
    """Minimal MPS capability stub."""

    def __init__(self, *, available: bool, built: bool) -> None:
        self._available = available
        self._built = built

    def is_available(self) -> bool:
        """Returns configured MPS availability."""
        return self._available

    def is_built(self) -> bool:
        """Returns configured MPS build flag."""
        return self._built


class _FakeBackends:
    """Container for torch.backends namespace stubs."""

    def __init__(self, *, mps: _FakeMps) -> None:
        self.mps = mps


class _FakeTorchModule:
    """Minimal torch-like module stub for runtime resolution tests."""

    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

    def __init__(
        self,
        *,
        cuda_available: bool,
        cuda_bf16_supported: bool,
        mps_available: bool,
        mps_built: bool,
    ) -> None:
        self.cuda = _FakeCuda(
            available=cuda_available,
            bf16_supported=cuda_bf16_supported,
        )
        self.backends = _FakeBackends(
            mps=_FakeMps(available=mps_available, built=mps_built)
        )
        self.context_calls: list[str] = []

    def device(self, spec: str) -> str:
        """Builds a deterministic device token."""
        return f"device:{spec}"

    def inference_mode(self) -> object:
        """Tracks inference_mode usage and returns a null context."""
        self.context_calls.append("inference_mode")
        return nullcontext()

    def no_grad(self) -> object:
        """Tracks no_grad usage and returns a null context."""
        self.context_calls.append("no_grad")
        return nullcontext()


class _FakeTensor:
    """Tensor-like stub that records `.to(...)` argument dictionaries."""

    def __init__(self) -> None:
        self.to_calls: list[dict[str, object]] = []

    def to(self, **kwargs: object) -> _FakeTensor:
        """Records move/cast call and returns self for chaining."""
        self.to_calls.append(dict(kwargs))
        return self


def test_maybe_resolve_torch_runtime_returns_none_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver should return None when torch import is unavailable."""
    monkeypatch.setattr(
        torch_inference.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("torch missing")),
    )
    runtime = torch_inference.maybe_resolve_torch_runtime(device="auto", dtype="auto")
    assert runtime is None


def test_maybe_resolve_torch_runtime_prefers_cuda_and_bf16_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto selectors should pick CUDA + bf16 when available."""
    fake_torch = _FakeTorchModule(
        cuda_available=True,
        cuda_bf16_supported=True,
        mps_available=True,
        mps_built=True,
    )
    monkeypatch.setattr(
        torch_inference.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else None,
    )

    runtime = torch_inference.maybe_resolve_torch_runtime(device="auto", dtype="auto")

    assert runtime is not None
    assert runtime.device_spec == "cuda"
    assert runtime.device_type == "cuda"
    assert runtime.dtype_name == "bfloat16"
    assert runtime.device == "device:cuda"
    assert runtime.dtype == "bfloat16"


def test_maybe_resolve_torch_runtime_rejects_non_float32_cpu_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU selection should reject dtype choices other than float32/auto."""
    fake_torch = _FakeTorchModule(
        cuda_available=False,
        cuda_bf16_supported=False,
        mps_available=False,
        mps_built=False,
    )
    monkeypatch.setattr(
        torch_inference.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else None,
    )
    with pytest.raises(RuntimeError, match="CPU runtime only supports"):
        torch_inference.maybe_resolve_torch_runtime(device="cpu", dtype="float16")


def test_move_inputs_to_runtime_casts_only_selected_dtype_keys() -> None:
    """Input move helper should apply dtype cast only for configured keys."""
    runtime = torch_inference.TorchRuntime(
        device="device:cuda",
        dtype="float16",
        device_spec="cuda:0",
        device_type="cuda",
        dtype_name="float16",
    )
    input_values = _FakeTensor()
    attention_mask = _FakeTensor()

    moved = torch_inference.move_inputs_to_runtime(
        {
            "input_values": input_values,
            "attention_mask": attention_mask,
        },
        runtime,
        dtype_keys=frozenset({"input_values"}),
    )

    assert moved["input_values"] is input_values
    assert moved["attention_mask"] is attention_mask
    assert input_values.to_calls == [
        {"device": "device:cuda", "dtype": "float16"},
    ]
    assert attention_mask.to_calls == [
        {"device": "device:cuda"},
    ]


def test_inference_context_prefers_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context helper should use inference_mode when available."""
    fake_torch = _FakeTorchModule(
        cuda_available=False,
        cuda_bf16_supported=False,
        mps_available=False,
        mps_built=False,
    )
    monkeypatch.setattr(
        torch_inference.importlib,
        "import_module",
        lambda name: fake_torch if name == "torch" else None,
    )

    with torch_inference.inference_context():
        pass

    assert fake_torch.context_calls == ["inference_mode"]
