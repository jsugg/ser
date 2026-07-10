"""Torch-based classifier head helpers for medium/accurate profile training."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

type HiddenDims = tuple[int, ...]
type LogitMatrix = NDArray[np.float64]
type FeatureMatrix = NDArray[np.float64]


def _load_torch_modules() -> tuple[Any, Any]:
    """Loads torch + torch.nn modules with explicit dependency errors."""
    try:
        torch_module = importlib.import_module("torch")
        nn_module = importlib.import_module("torch.nn")
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Torch head requires optional dependency 'torch'. "
            "Install medium-profile extras and retry."
        ) from err
    return torch_module, nn_module


def build_torch_mlp_head(
    *,
    input_dim: int,
    hidden_dims: HiddenDims = (256, 128),
    output_dim: int,
    dropout: float = 0.3,
) -> object:
    """Builds a simple MLP head for embedding classification.

    Args:
        input_dim: Input feature width.
        hidden_dims: Hidden-layer widths in stack order.
        output_dim: Number of target classes.
        dropout: Dropout probability for hidden layers.

    Returns:
        A ``torch.nn.Sequential`` model object.

    Raises:
        ValueError: If any dimensions or dropout are invalid.
        RuntimeError: If torch is unavailable.
    """
    if input_dim <= 0:
        raise ValueError("input_dim must be a positive integer.")
    if output_dim <= 0:
        raise ValueError("output_dim must be a positive integer.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0.0, 1.0).")
    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise ValueError("hidden_dims entries must be positive integers.")

    _, nn_module = _load_torch_modules()
    layers: list[object] = []
    previous_dim: int = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            (
                nn_module.Linear(previous_dim, hidden_dim),
                nn_module.ReLU(),
                nn_module.Dropout(dropout),
            )
        )
        previous_dim = hidden_dim
    layers.append(nn_module.Linear(previous_dim, output_dim))
    return nn_module.Sequential(*layers)


def forward_torch_head(head: object, features: FeatureMatrix) -> LogitMatrix:
    """Runs forward pass and returns logits as numpy float64 matrix.

    Args:
        head: Torch module returned by :func:`build_torch_mlp_head`.
        features: 2D feature matrix.

    Returns:
        Logits matrix with shape ``(rows, output_dim)``.

    Raises:
        ValueError: If input/output shapes are invalid.
        RuntimeError: If torch is unavailable or head is not callable.
    """
    if features.ndim != 2:
        raise ValueError("features must be a 2D matrix.")
    torch_module, _ = _load_torch_modules()
    if not callable(head):
        raise RuntimeError("Torch head must be a callable model.")

    tensor = torch_module.as_tensor(features, dtype=torch_module.float32)
    no_grad = getattr(torch_module, "no_grad", None)
    context: AbstractContextManager[object]
    if callable(no_grad):
        context = cast(AbstractContextManager[object], no_grad())
    else:
        context = nullcontext()
    with context:
        logits: object = head(tensor)

    current: object = logits
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    to_numpy = getattr(current, "numpy", None)
    if callable(to_numpy):
        current = to_numpy()
    logits_array: NDArray[np.float64] = np.asarray(current, dtype=np.float64)
    if logits_array.ndim != 2:
        raise ValueError("Torch head forward pass must return a 2D logits matrix.")
    if logits_array.shape[0] != features.shape[0]:
        raise ValueError(
            "Torch head forward pass row mismatch. "
            f"Expected {features.shape[0]}, got {logits_array.shape[0]}."
        )
    return logits_array
