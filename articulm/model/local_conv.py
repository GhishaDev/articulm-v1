"""Optional local coarticulation module.

Depthwise Conv1D over the time axis, inserted every ``every_n_layers``
transformer blocks. OFF in the baseline config; the ON/OFF ablation is
experiment E3 in ``docs/08_experiment_plan.md``.

Padded positions are zeroed before convolving so PAD never leaks into a real
neighbour through the kernel window.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from .fusion import build_activation


class LocalCoarticulationConv(nn.Module):
    """``LayerNorm -> Conv1D(k) -> activation -> Dropout -> residual``."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        *,
        depthwise: bool = True,
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for 'same' padding, got {kernel_size}")

        self.norm = nn.LayerNorm(hidden_size)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size if depthwise else 1,
        )
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.depthwise = depthwise

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B,T,H]``, ``attention_mask`` ``[B,T]`` bool (True = real)."""
        keep = attention_mask.unsqueeze(-1).to(hidden.dtype)

        residual = hidden
        normalised = self.norm(hidden) * keep
        # Conv1d wants [B, C, T].
        convolved = self.conv(normalised.transpose(1, 2)).transpose(1, 2)
        convolved = self.dropout(self.activation(convolved))
        return (residual + convolved) * keep


def build_local_conv_stack(cfg: ModelConfig) -> nn.ModuleList:
    """One conv module per insertion point, or an empty list when disabled.

    Insertion points are the block indices *after* which a conv runs:
    ``every_n_layers = 2`` over 10 layers gives blocks 2, 4, 6, 8, 10.
    """
    if not cfg.local_conv.enabled:
        return nn.ModuleList()
    num_insertions = cfg.transformer.num_layers // cfg.local_conv.every_n_layers
    return nn.ModuleList(
        [
            LocalCoarticulationConv(
                hidden_size=cfg.transformer.hidden_size,
                kernel_size=cfg.local_conv.kernel_size,
                depthwise=cfg.local_conv.depthwise,
                activation=cfg.transformer.activation,
                dropout=cfg.transformer.dropout,
            )
            for _ in range(num_insertions)
        ]
    )


def local_conv_insertion_points(cfg: ModelConfig) -> tuple[int, ...]:
    """Block indices (0-based) after which a local conv is applied."""
    if not cfg.local_conv.enabled:
        return ()
    step = cfg.local_conv.every_n_layers
    return tuple(
        index
        for index in range(cfg.transformer.num_layers)
        if (index + 1) % step == 0
    )
