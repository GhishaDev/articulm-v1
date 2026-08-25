"""Warmup + decay learning-rate schedules driven by optimizer steps."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ..config import SchedulerConfig


def build_scheduler(
    optimizer: Optimizer, cfg: SchedulerConfig, total_steps: int
) -> LambdaLR:
    """Build a per-optimizer-step LR schedule with linear warmup.

    ``total_steps`` counts optimizer steps (after gradient accumulation), not
    micro-batches.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    warmup_steps = round(total_steps * cfg.warmup_ratio)
    warmup_steps = min(max(warmup_steps, 0), max(total_steps - 1, 0))
    min_ratio = cfg.min_lr_ratio
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0,1], got {min_ratio}")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if cfg.type == "constant":
            return 1.0
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        if cfg.type == "linear":
            factor = 1.0 - progress
        elif cfg.type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"unsupported scheduler {cfg.type!r}")
        return min_ratio + (1.0 - min_ratio) * factor

    return LambdaLR(optimizer, lr_lambda)


def current_learning_rate(optimizer: Optimizer) -> float:
    """LR of the first parameter group, for logging."""
    return float(optimizer.param_groups[0]["lr"])
