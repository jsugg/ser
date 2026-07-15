"""Tests for masked uncertainty-weighted multitask losses."""

from __future__ import annotations

import pytest
import torch

from ser._internal.heads.multitask_loss import MaskedUncertaintyWeightedLoss


def test_missing_auxiliary_targets_contribute_no_loss_or_gradient() -> None:
    """Unavailable auxiliary labels remain isolated from the primary objective."""
    objective = MaskedUncertaintyWeightedLoss(("primary_emotion", "vad"))
    primary = torch.tensor([1.0, 3.0], requires_grad=True)
    auxiliary = torch.tensor([100.0, 200.0], requires_grad=True)

    value = objective(
        {"primary_emotion": primary, "vad": auxiliary},
        {
            "primary_emotion": torch.tensor([True, True]),
            "vad": torch.tensor([False, False]),
        },
    )
    value.backward()

    assert value.item() == pytest.approx(2.0)
    assert primary.grad is not None
    assert auxiliary.grad is None
    assert objective.log_variances["vad"].grad is None


def test_loss_rejects_batches_without_any_available_target() -> None:
    """Silent zero-loss batches fail closed."""
    objective = MaskedUncertaintyWeightedLoss(("primary_emotion",))

    with pytest.raises(ValueError, match="No available targets"):
        objective(
            {"primary_emotion": torch.tensor([1.0])},
            {"primary_emotion": torch.tensor([False])},
        )
