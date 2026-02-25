"""Torch runtime helpers for optional device and dtype selection."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class TorchRuntime:
    """Resolved runtime selectors for one torch-backed inference backend."""

    device: object
    dtype: object
    device_spec: str
    device_type: str
    dtype_name: str


def _cuda_is_available(torch_module: object) -> bool:
    """Returns whether CUDA runtime is available in current environment."""
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def _cuda_bf16_is_supported(torch_module: object) -> bool:
    """Returns whether CUDA backend supports bfloat16 inference kernels."""
    cuda = getattr(torch_module, "cuda", None)
    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    return bool(is_bf16_supported()) if callable(is_bf16_supported) else False


def _mps_is_available(torch_module: object) -> bool:
    """Returns whether Apple MPS runtime is available and built."""
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    is_built = getattr(mps, "is_built", None)
    available = bool(is_available()) if callable(is_available) else False
    built = bool(is_built()) if callable(is_built) else False
    return available and built


def _resolve_device_spec(torch_module: object, requested_device: str) -> str:
    """Resolves runtime device selector from one requested device string."""
    normalized = requested_device.strip().lower()
    if normalized == "auto":
        if _cuda_is_available(torch_module):
            return "cuda"
        if _mps_is_available(torch_module):
            return "mps"
        return "cpu"
    if normalized.startswith("cuda"):
        if not _cuda_is_available(torch_module):
            raise RuntimeError(
                "SER_TORCH_DEVICE requested CUDA, but CUDA is unavailable."
            )
        return normalized
    if normalized == "mps":
        if not _mps_is_available(torch_module):
            raise RuntimeError("SER_TORCH_DEVICE requested MPS, but MPS is unavailable.")
        return "mps"
    return "cpu"


def _resolve_dtype_name(
    torch_module: object,
    *,
    requested_dtype: str,
    device_type: str,
) -> str:
    """Resolves runtime dtype selector from requested dtype and device type."""
    normalized = requested_dtype.strip().lower()
    if normalized not in {"auto", "float32", "float16", "bfloat16"}:
        raise RuntimeError(f"Unsupported torch dtype selector {requested_dtype!r}.")

    if device_type == "cpu":
        if normalized in {"auto", "float32"}:
            return "float32"
        raise RuntimeError("CPU runtime only supports SER_TORCH_DTYPE=float32.")

    if normalized == "auto":
        if device_type == "cuda" and _cuda_bf16_is_supported(torch_module):
            if getattr(torch_module, "bfloat16", None) is not None:
                return "bfloat16"
        return "float16"

    if normalized == "bfloat16":
        if device_type == "mps":
            raise RuntimeError("SER_TORCH_DTYPE=bfloat16 is unsupported for MPS.")
        if device_type == "cuda" and not _cuda_bf16_is_supported(torch_module):
            raise RuntimeError(
                "SER_TORCH_DTYPE=bfloat16 requested, but CUDA bf16 is unsupported."
            )
        if getattr(torch_module, "bfloat16", None) is None:
            raise RuntimeError("Installed torch build does not expose bfloat16 dtype.")
    return normalized


def maybe_resolve_torch_runtime(
    *,
    device: str = "auto",
    dtype: str = "auto",
) -> TorchRuntime | None:
    """Resolves torch runtime selectors, or returns None when torch is absent."""
    try:
        torch_module = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None

    resolved_device_spec = _resolve_device_spec(torch_module, device)
    resolved_device_type = (
        "cuda"
        if resolved_device_spec.startswith("cuda")
        else ("mps" if resolved_device_spec == "mps" else "cpu")
    )
    resolved_dtype_name = _resolve_dtype_name(
        torch_module,
        requested_dtype=dtype,
        device_type=resolved_device_type,
    )

    device_ctor = getattr(torch_module, "device", None)
    if not callable(device_ctor):
        raise RuntimeError("torch.device is unavailable in current torch installation.")
    dtype_obj = getattr(torch_module, resolved_dtype_name, None)
    if dtype_obj is None:
        raise RuntimeError(
            f"torch.{resolved_dtype_name} is unavailable in current torch installation."
        )

    return TorchRuntime(
        device=device_ctor(resolved_device_spec),
        dtype=dtype_obj,
        device_spec=resolved_device_spec,
        device_type=resolved_device_type,
        dtype_name=resolved_dtype_name,
    )


def runtime_with_dtype(runtime: TorchRuntime, *, dtype: str) -> TorchRuntime | None:
    """Builds a runtime clone with the same device selector and new dtype."""
    return maybe_resolve_torch_runtime(device=runtime.device_spec, dtype=dtype)


def move_model_to_runtime(model: object, runtime: TorchRuntime) -> None:
    """Moves one backend model instance to selected device and dtype."""
    to_method = getattr(model, "to", None)
    if not callable(to_method):
        return
    to_method(device=runtime.device, dtype=runtime.dtype)


def move_inputs_to_runtime(
    inputs: Mapping[str, object],
    runtime: TorchRuntime,
    *,
    dtype_keys: frozenset[str],
) -> dict[str, object]:
    """Moves tensor-like mapping values to selected runtime device and dtype."""
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        to_method = getattr(value, "to", None)
        if callable(to_method):
            if key in dtype_keys:
                moved[key] = to_method(device=runtime.device, dtype=runtime.dtype)
            else:
                moved[key] = to_method(device=runtime.device)
        else:
            moved[key] = value
    return moved


def inference_context() -> AbstractContextManager[object]:
    """Returns best-effort torch inference context when available."""
    try:
        torch_module = importlib.import_module("torch")
    except ModuleNotFoundError:
        return nullcontext()
    inference_mode = getattr(torch_module, "inference_mode", None)
    if callable(inference_mode):
        return cast(AbstractContextManager[object], inference_mode())
    no_grad = getattr(torch_module, "no_grad", None)
    if callable(no_grad):
        return cast(AbstractContextManager[object], no_grad())
    return nullcontext()

