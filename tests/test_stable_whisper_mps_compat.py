"""Unit tests for stable-whisper MPS compatibility helpers."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from ser.transcript.backends import stable_whisper_mps_compat as mps_compat


class _FakeModel:
    """Minimal model-like object for MPS compatibility move tests."""

    def __init__(self) -> None:
        indices = torch.tensor([[0], [0]], dtype=torch.int64)
        values = torch.tensor([True], dtype=torch.bool)
        self.alignment_heads = torch.sparse_coo_tensor(
            indices,
            values,
            size=(1, 1),
        )
        self.buffer_updates: list[tuple[bool, str]] = []
        self.to_calls: list[str] = []

    def register_buffer(
        self,
        name: str,
        tensor: torch.Tensor,
        persistent: bool = False,
    ) -> None:
        assert name == "alignment_heads"
        assert persistent is False
        self.alignment_heads = tensor
        self.buffer_updates.append((bool(tensor.is_sparse), tensor.device.type))

    def to(self, *, device: str) -> _FakeModel:
        self.to_calls.append(device)
        return self


class _FailingMoveModel(_FakeModel):
    """Model-like object that fails during MPS move after mutating device state."""

    def __init__(self) -> None:
        super().__init__()
        self.device_type = "cpu"

    def to(self, *, device: str) -> _FailingMoveModel:
        self.to_calls.append(device)
        self.device_type = device
        if device == "mps":
            raise RuntimeError("MPS backend out of memory")
        return self


class _FakeHookHandle:
    """Minimal torch-style hook handle used in timing compatibility tests."""

    def __init__(self, hooks: list[Any], hook: Any) -> None:
        self._hooks = hooks
        self._hook = hook

    def remove(self) -> None:
        if self._hook in self._hooks:
            self._hooks.remove(self._hook)


class _FakeCrossAttention:
    """Cross-attention stub that invokes registered forward hooks."""

    def __init__(self) -> None:
        self.hooks: list[Any] = []

    def register_forward_hook(self, hook: Any) -> _FakeHookHandle:
        self.hooks.append(hook)
        return _FakeHookHandle(self.hooks, hook)

    def emit(self) -> None:
        qk = torch.randn((1, 2, 3, 4), dtype=torch.float32, requires_grad=True)
        outputs = (torch.zeros((1,), dtype=torch.float32), qk)
        for hook in list(self.hooks):
            hook(self, (), outputs)


class _FakeDecoder:
    """Decoder stub exposing blocks and producing one logits tensor."""

    def __init__(self, cross_attn: _FakeCrossAttention) -> None:
        self.blocks = [SimpleNamespace(cross_attn=cross_attn)]
        self._cross_attn = cross_attn

    def __call__(
        self, tokens: torch.Tensor, _audio_features: torch.Tensor
    ) -> torch.Tensor:
        self._cross_attn.emit()
        sequence_len = int(tokens.shape[1])
        return torch.randn((1, sequence_len, 8), dtype=torch.float32)


class _FakeStableWhisperModel:
    """Stable-whisper model stub for _compute_qks offload tests."""

    def __init__(self, cross_attn: _FakeCrossAttention) -> None:
        self.dims = SimpleNamespace(n_text_layer=1)
        self.decoder = _FakeDecoder(cross_attn)

    def encoder(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.mean(dim=-1, keepdim=True)


def test_move_model_to_mps_with_alignment_placeholder_restores_sparse_cpu_buffer() -> (
    None
):
    """Sparse alignment buffer should be restored on CPU after MPS move."""
    model = _FakeModel()

    moved_model = mps_compat.move_model_to_mps_with_alignment_placeholder(model)

    assert moved_model is model
    assert model.to_calls == ["mps"]
    assert model.buffer_updates[0] == (False, "cpu")
    assert model.buffer_updates[1] == (True, "cpu")
    assert bool(model.alignment_heads.is_sparse) is True
    assert model.alignment_heads.device.type == "cpu"


def test_move_model_to_mps_with_alignment_placeholder_rolls_back_to_cpu_on_failure() -> (
    None
):
    """Failed MPS move should restore sparse buffer and force model rollback to CPU."""
    model = _FailingMoveModel()

    with pytest.raises(RuntimeError, match="MPS backend out of memory"):
        mps_compat.move_model_to_mps_with_alignment_placeholder(model)

    assert model.to_calls == ["mps", "cpu"]
    assert model.device_type == "cpu"
    assert bool(model.alignment_heads.is_sparse) is True
    assert model.alignment_heads.device.type == "cpu"


def test_mps_timing_compatibility_context_patches_and_restores_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context should patch DTW/std_mean/_compute_qks aliases and restore on exit."""

    def _original_compute_qks(*_args: object, **_kwargs: object) -> None:
        return None

    @contextmanager
    def _disable_sdpa() -> Any:
        yield

    fake_timing = SimpleNamespace(
        dtw=lambda _x: "timing_original",
        _compute_qks=_original_compute_qks,
    )
    fake_compat = SimpleNamespace(
        dtw=lambda _x: "compat_original",
        disable_sdpa=_disable_sdpa,
    )
    fake_whisper_timing = SimpleNamespace(dtw_cpu=lambda x: ("cpu", x))

    def _fake_import_module(name: str) -> object:
        if name == "stable_whisper.timing":
            return fake_timing
        if name == "stable_whisper.whisper_compatibility":
            return fake_compat
        if name == "whisper.timing":
            return fake_whisper_timing
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(mps_compat.importlib, "import_module", _fake_import_module)
    original_std_mean = torch.std_mean
    original_compute_qks = fake_timing._compute_qks

    with mps_compat.stable_whisper_mps_timing_compatibility_context():
        assert fake_timing.dtw is not None
        assert fake_timing.dtw is fake_compat.dtw
        assert torch.std_mean is not original_std_mean
        assert fake_timing._compute_qks is not original_compute_qks
        dtw_result = cast(object, fake_timing.dtw(torch.tensor([1.0])))
        assert isinstance(dtw_result, tuple)
        assert dtw_result[0] == "cpu"

    assert fake_timing.dtw(torch.tensor([1.0])) == "timing_original"
    assert fake_compat.dtw(torch.tensor([1.0])) == "compat_original"
    assert torch.std_mean is original_std_mean
    assert fake_timing._compute_qks is original_compute_qks


def test_mps_timing_compatibility_context_offloads_compute_qks_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patched _compute_qks should cache detached QKs on CPU and remove hooks."""
    original_calls: list[str] = []

    def _original_compute_qks(
        *,
        model: object,
        tokenizer: object,
        text_tokens: object,
        mel: object,
        tokens: object,
        cache: object,
    ) -> None:
        del model, tokenizer, text_tokens, mel, tokens, cache
        original_calls.append("called")

    @contextmanager
    def _disable_sdpa() -> Any:
        yield

    fake_cross_attn = _FakeCrossAttention()
    fake_model = _FakeStableWhisperModel(fake_cross_attn)
    fake_tokenizer = SimpleNamespace(sot_sequence=[0, 1], eot=7)
    fake_timing = SimpleNamespace(
        dtw=lambda _x: "timing_original",
        _compute_qks=_original_compute_qks,
    )
    fake_compat = SimpleNamespace(
        dtw=lambda _x: "compat_original",
        disable_sdpa=_disable_sdpa,
    )
    fake_whisper_timing = SimpleNamespace(dtw_cpu=lambda x: ("cpu", x))

    def _fake_import_module(name: str) -> object:
        if name == "stable_whisper.timing":
            return fake_timing
        if name == "stable_whisper.whisper_compatibility":
            return fake_compat
        if name == "whisper.timing":
            return fake_whisper_timing
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(mps_compat.importlib, "import_module", _fake_import_module)
    cache: dict[str, object] = {"audio_features": None}
    mel = torch.ones((80, 6), dtype=torch.float32)
    tokens = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    text_tokens = [2, 3]

    with mps_compat.stable_whisper_mps_timing_compatibility_context():
        fake_timing._compute_qks(
            model=fake_model,
            tokenizer=fake_tokenizer,
            text_tokens=text_tokens,
            mel=mel,
            tokens=tokens,
            cache=cache,
        )

    assert original_calls == []
    assert isinstance(cache["qks"], list)
    cached_qk = cast(torch.Tensor, cast(list[object], cache["qks"])[0])
    assert cached_qk.device.type == "cpu"
    assert cached_qk.requires_grad is False
    assert len(cast(list[float], cache["text_token_probs"])) == len(text_tokens)
    assert fake_cross_attn.hooks == []
