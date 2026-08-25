"""Feature fusion projection.

```text
[B,T,384] -> Linear 384->640 -> LayerNorm -> GELU -> Dropout -> [B,T,640]
```
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


def build_activation(name: str) -> nn.Module:
    try:
        return ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}") from exc


class FeatureFusion(nn.Module):
    """Project concatenated feature embeddings into the transformer width."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        input_dim = cfg.input.fused_input_dim
        output_dim = cfg.fusion.output_dim

        self.projection = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if cfg.fusion.layer_norm else nn.Identity()
        self.activation = build_activation(cfg.fusion.activation)
        self.dropout = nn.Dropout(cfg.fusion.dropout)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"FeatureFusion expected last dim {self.input_dim}, got {features.shape[-1]}"
            )
        hidden = self.projection(features)
        hidden = self.norm(hidden)
        hidden = self.activation(hidden)
        return self.dropout(hidden)
