"""Tests for torch head construction and deterministic forward contracts."""

from __future__ import annotations

import numpy as np
import pytest

from ser.heads.torch_head import build_torch_mlp_head, forward_torch_head


def test_build_torch_mlp_head_rejects_invalid_dimensions() -> None:
    """Torch head builder should validate dimension arguments strictly."""
    with pytest.raises(ValueError, match="input_dim"):
        build_torch_mlp_head(input_dim=0, output_dim=3)
    with pytest.raises(ValueError, match="output_dim"):
        build_torch_mlp_head(input_dim=4, output_dim=0)
    with pytest.raises(ValueError, match="hidden_dims"):
        build_torch_mlp_head(input_dim=4, output_dim=3, hidden_dims=(8, 0))


def test_forward_torch_head_emits_expected_logits_shape() -> None:
    """Forward pass should preserve row count and emit configured class width."""
    pytest.importorskip("torch")
    head = build_torch_mlp_head(
        input_dim=4,
        hidden_dims=(8, 4),
        output_dim=3,
        dropout=0.0,
    )
    features = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float64,
    )

    logits = forward_torch_head(head, features)

    assert logits.shape == (2, 3)
    assert np.isfinite(logits).all()
