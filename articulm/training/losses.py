"""Masked Viseme and Strength losses.

```text
Viseme   : CrossEntropy, label_smoothing from config (0.05 synthetic baseline)
Strength : SmoothL1 / Huber on strength/100 against the sigmoid output
Total    : viseme_weight * L_viseme + strength_weight * L_strength
```

Synthetic baseline weights are ``1.0 / 0.3``; Human Gold fine-tuning uses
``1.0 / 1.0``. Both come from YAML, never from a hard-coded branch.

Padding never contributes: both losses gather only the tokens selected by
``batch.loss_mask``. ``pseudo_strength_v1`` is a programmatic prior and is
reweighted through ``source_weights``; it is never treated as Human Gold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..config import LossConfig
from ..data.collator import Batch
from ..model.articulm_v1 import ArticuLMOutput


@dataclass
class LossBreakdown:
    """Scalar losses plus the token counts they were averaged over."""

    total: torch.Tensor
    viseme: torch.Tensor
    strength: torch.Tensor
    num_supervised_tokens: int
    strength_weight_sum: float

    def as_floats(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "viseme_loss": float(self.viseme.detach()),
            "strength_loss": float(self.strength.detach()),
        }


class ArticuLMLoss(nn.Module):
    """Combined masked Viseme + Strength objective."""

    def __init__(self, cfg: LossConfig, *, num_classes: int = 18) -> None:
        super().__init__()
        self.config = cfg
        self.num_classes = num_classes
        self.viseme_weight = cfg.viseme.weight
        self.strength_weight = cfg.strength.weight
        self.label_smoothing = cfg.viseme.label_smoothing
        self.strength_type = cfg.strength.type
        self.strength_beta = cfg.strength.beta

    # ---------------------------------------------------------------- parts

    def viseme_loss(
        self, viseme_logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy over unpadded tokens only."""
        selected_logits = viseme_logits[mask]  # [N, C]
        selected_targets = targets[mask]  # [N]
        if selected_targets.numel() == 0:
            return viseme_logits.new_zeros(())
        if int(selected_targets.min()) < 0 or int(selected_targets.max()) >= self.num_classes:
            raise ValueError(
                "viseme targets outside [0,num_classes) survived the loss mask; "
                "this indicates a padding/mask bug"
            )
        return F.cross_entropy(
            selected_logits,
            selected_targets,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

    def strength_loss(
        self,
        strength_norm: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, float]:
        """Weighted regression loss on normalised strength."""
        prediction = strength_norm[mask]
        target = targets[mask]
        if target.numel() == 0:
            return strength_norm.new_zeros(()), 0.0

        if self.strength_type in {"smooth_l1", "huber"}:
            per_token = F.smooth_l1_loss(
                prediction, target, reduction="none", beta=self.strength_beta
            )
        elif self.strength_type == "l1":
            per_token = F.l1_loss(prediction, target, reduction="none")
        elif self.strength_type == "mse":
            per_token = F.mse_loss(prediction, target, reduction="none")
        else:
            raise ValueError(f"unsupported strength loss {self.strength_type!r}")

        if weights is None:
            return per_token.mean(), float(target.numel())

        token_weights = weights[mask]
        weight_sum = token_weights.sum()
        if float(weight_sum) <= 0.0:
            return strength_norm.new_zeros(()), 0.0
        return (per_token * token_weights).sum() / weight_sum, float(weight_sum)

    # -------------------------------------------------------------- forward

    def forward(self, output: ArticuLMOutput, batch: Batch) -> LossBreakdown:
        if batch.viseme_targets is None or batch.strength_targets is None:
            raise ValueError("ArticuLMLoss requires a labelled batch")

        mask = batch.loss_mask
        if mask.dtype != torch.bool:
            mask = mask.bool()
        # Padding must never widen the supervised set.
        if (mask & ~batch.attention_mask.bool()).any():
            raise ValueError("loss_mask selects tokens outside attention_mask")

        viseme = self.viseme_loss(output.viseme_logits, batch.viseme_targets, mask)
        strength, weight_sum = self.strength_loss(
            output.strength_norm, batch.strength_targets, mask, batch.strength_weight
        )
        total = self.viseme_weight * viseme + self.strength_weight * strength

        return LossBreakdown(
            total=total,
            viseme=viseme,
            strength=strength,
            num_supervised_tokens=int(mask.sum()),
            strength_weight_sum=weight_sum,
        )
