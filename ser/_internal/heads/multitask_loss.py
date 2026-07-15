"""Masked uncertainty-weighted objectives for auxiliary training heads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn


class MaskedUncertaintyWeightedLoss(nn.Module):
    """Combines available per-sample task losses with learned uncertainty weights."""

    def __init__(
        self,
        tasks: Sequence[str],
        *,
        primary_task: str = "primary_emotion",
        minimum_primary_weight: float = 0.25,
    ) -> None:
        """Initializes trainable log variances for a fixed task set."""
        super().__init__()
        normalized_tasks = tuple(dict.fromkeys(task.strip() for task in tasks if task.strip()))
        if not normalized_tasks:
            raise ValueError("At least one multitask objective is required.")
        if not 0.0 < minimum_primary_weight <= 1.0:
            raise ValueError("minimum_primary_weight must be within (0, 1].")
        if any("." in task for task in normalized_tasks):
            raise ValueError("Task names cannot contain '.'.")
        self.primary_task = primary_task
        self.minimum_primary_weight = minimum_primary_weight
        self.log_variances = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros((), dtype=torch.float32)) for task in normalized_tasks}
        )

    def forward(
        self,
        losses: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> Tensor:
        """Returns a scalar loss using only targets marked available by each mask."""
        total: Tensor | None = None
        active_tasks = 0
        for task, log_variance in self.log_variances.items():
            if task not in losses or task not in masks:
                continue
            task_losses = losses[task]
            mask = masks[task]
            if task_losses.ndim == 0:
                task_losses = task_losses.unsqueeze(0)
            if mask.shape != task_losses.shape:
                raise ValueError(f"Loss and mask shapes differ for task {task!r}.")
            active = mask.to(dtype=torch.bool)
            if not bool(torch.any(active)):
                continue
            mean_loss = task_losses[active].mean()
            weight = torch.exp(-log_variance)
            if task == self.primary_task:
                weight = torch.clamp_min(weight, self.minimum_primary_weight)
            weighted = weight * mean_loss + log_variance
            total = weighted if total is None else total + weighted
            active_tasks += 1
        if total is None or active_tasks == 0:
            raise ValueError("No available targets were supplied to the multitask loss.")
        return total
