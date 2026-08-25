"""Output heads: Viseme, Soft Viseme Embedding and Strength.

The Strength Head reads the *soft* probability-weighted viseme embedding, so
gradient from the Strength loss flows back through the Viseme logits. Hard
``argmax`` appears only at inference time, never on the training path.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .fusion import build_activation


class VisemeHead(nn.Module):
    """``[B,T,640] -> 640->256 -> GELU -> Dropout -> 256->18 -> logits``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden_size = cfg.transformer.hidden_size
        head = cfg.viseme_head

        self.input_projection = nn.Linear(hidden_size, head.hidden_size)
        self.activation = build_activation(cfg.fusion.activation)
        self.dropout = nn.Dropout(head.dropout)
        self.classifier = nn.Linear(head.hidden_size, head.num_classes)
        self.num_classes = head.num_classes

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = self.dropout(self.activation(self.input_projection(hidden)))
        return self.classifier(projected)


class SoftVisemeEmbedding(nn.Module):
    """Trainable ``E in R^(18x32)``; returns ``softmax(logits) @ E``.

    Differentiable by construction — this is the only path by which the
    Strength loss can shape the viseme distribution.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        embedding = cfg.viseme_embedding
        if embedding.mode != "soft_probability_weighted":
            raise ValueError(
                "SoftVisemeEmbedding only supports mode='soft_probability_weighted'; "
                f"got {embedding.mode!r}"
            )
        self.num_embeddings = embedding.num_embeddings
        self.dim = embedding.dim
        self.table = nn.Parameter(torch.empty(embedding.num_embeddings, embedding.dim))
        nn.init.normal_(self.table, mean=0.0, std=0.02)

    def forward(self, viseme_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(probabilities [B,T,18], soft_embedding [B,T,32])``."""
        if viseme_logits.shape[-1] != self.num_embeddings:
            raise ValueError(
                f"expected {self.num_embeddings} viseme logits, got {viseme_logits.shape[-1]}"
            )
        probabilities = F.softmax(viseme_logits, dim=-1)
        soft_embedding = probabilities @ self.table
        return probabilities, soft_embedding


class StrengthHead(nn.Module):
    """``concat(H, e_v) -> 672 -> 256 -> 64 -> 1 -> Sigmoid``.

    Emits normalised strength in ``[0,1]``. Callers multiply by
    ``output_scale`` (100) for user-facing output.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        head = cfg.strength_head
        expected_input = cfg.transformer.hidden_size + cfg.viseme_embedding.dim
        if head.input_dim != expected_input:
            raise ValueError(
                f"StrengthHead input_dim {head.input_dim} != hidden {cfg.transformer.hidden_size} "
                f"+ soft viseme {cfg.viseme_embedding.dim}"
            )

        layers: list[nn.Module] = []
        in_dim = head.input_dim
        # Dropout after every hidden activation except the last, matching
        # docs/01 section 9 (dropout only after the 672->256 block for [256,64]).
        for position, hidden_dim in enumerate(head.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(build_activation(head.activation))
            if position < len(head.hidden_dims) - 1:
                layers.append(nn.Dropout(head.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))

        self.network = nn.Sequential(*layers)
        self.output_scale = head.output_scale
        self.input_dim = head.input_dim

    def forward(
        self, hidden: torch.Tensor, soft_viseme_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return normalised strength ``[B,T]`` in ``[0,1]``."""
        combined = torch.cat((hidden, soft_viseme_embedding), dim=-1)
        if combined.shape[-1] != self.input_dim:
            raise ValueError(
                f"StrengthHead expected input width {self.input_dim}, got {combined.shape[-1]}"
            )
        raw = self.network(combined).squeeze(-1)
        return torch.sigmoid(raw)
