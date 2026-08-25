"""ArticuLM-V1: the assembled phoneme-to-viseme + strength model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import ModelConfig
from ..data.vocab import FeatureVocabulary
from .embeddings import PhonemeFeatureEmbedding
from .fusion import FeatureFusion
from .heads import SoftVisemeEmbedding, StrengthHead, VisemeHead
from .tracing import is_graph_capture
from .transformer import ContextTransformer


@dataclass
class ArticuLMOutput:
    """Forward outputs. ``strength`` is only filled on the inference path."""

    # [B,T,640] final context hidden states (zero at padded positions).
    hidden_states: torch.Tensor
    # [B,T,18] pre-softmax viseme logits.
    viseme_logits: torch.Tensor
    # [B,T,18] softmax probabilities.
    viseme_probabilities: torch.Tensor
    # [B,T,32] probability-weighted soft viseme embedding.
    soft_viseme_embedding: torch.Tensor
    # [B,T] sigmoid output in [0,1] — the training target space.
    strength_norm: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.viseme_logits.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.viseme_logits.shape[1])

    def predicted_viseme_ids(self) -> torch.Tensor:
        """``argmax`` over logits. Inference only — never on the loss path."""
        return self.viseme_logits.argmax(dim=-1)

    def strength_scaled(self, scale: float = 100.0) -> torch.Tensor:
        return self.strength_norm * scale


@dataclass
class ParameterBreakdown:
    embeddings: int
    fusion: int
    transformer: int
    viseme_head: int
    soft_viseme_embedding: int
    strength_head: int

    @property
    def total(self) -> int:
        return (
            self.embeddings
            + self.fusion
            + self.transformer
            + self.viseme_head
            + self.soft_viseme_embedding
            + self.strength_head
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "embeddings": self.embeddings,
            "fusion": self.fusion,
            "transformer": self.transformer,
            "viseme_head": self.viseme_head,
            "soft_viseme_embedding": self.soft_viseme_embedding,
            "strength_head": self.strength_head,
            "total": self.total,
        }


class ArticuLMV1(nn.Module):
    """```
    feature ids -> embeddings -> concat 384 -> fusion 640
                -> 10-layer context transformer
                -> Viseme Head (18) -> softmax -> Soft Viseme Embedding (32)
                -> Strength Head (672 -> 256 -> 64 -> 1 -> sigmoid)
    ```"""

    def __init__(self, cfg: ModelConfig, vocab_sizes: dict[str, int]) -> None:
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.vocab_sizes = dict(vocab_sizes)

        self.embeddings = PhonemeFeatureEmbedding(cfg, vocab_sizes)
        self.fusion = FeatureFusion(cfg)
        self.transformer = ContextTransformer(cfg)
        self.viseme_head = VisemeHead(cfg)
        self.soft_viseme_embedding = SoftVisemeEmbedding(cfg)
        self.strength_head = StrengthHead(cfg)

        self.strength_scale = cfg.strength_head.output_scale
        self.max_seq_len = cfg.input.max_seq_len

    # ------------------------------------------------------------- forward

    def forward(
        self, feature_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> ArticuLMOutput:
        if not is_graph_capture() and feature_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {feature_ids.shape[1]} exceeds max_seq_len {self.max_seq_len}"
            )
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        embedded = self.embeddings(feature_ids)
        fused = self.fusion(embedded)
        hidden = self.transformer(fused, attention_mask)

        viseme_logits = self.viseme_head(hidden)
        probabilities, soft_embedding = self.soft_viseme_embedding(viseme_logits)
        strength_norm = self.strength_head(hidden, soft_embedding)

        return ArticuLMOutput(
            hidden_states=hidden,
            viseme_logits=viseme_logits,
            viseme_probabilities=probabilities,
            soft_viseme_embedding=soft_embedding,
            strength_norm=strength_norm,
        )

    @torch.no_grad()
    def predict(
        self, feature_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(viseme_ids [B,T], strength [B,T])`` scaled to 0..100."""
        was_training = self.training
        self.eval()
        try:
            output = self.forward(feature_ids, attention_mask)
        finally:
            if was_training:
                self.train()
        return output.predicted_viseme_ids(), output.strength_scaled(self.strength_scale)

    # -------------------------------------------------------- introspection

    def parameter_breakdown(self) -> ParameterBreakdown:
        def count(module: nn.Module | nn.Parameter) -> int:
            if isinstance(module, nn.Parameter):
                return module.numel()
            return sum(p.numel() for p in module.parameters())

        return ParameterBreakdown(
            embeddings=count(self.embeddings),
            fusion=count(self.fusion),
            transformer=count(self.transformer),
            viseme_head=count(self.viseme_head),
            soft_viseme_embedding=count(self.soft_viseme_embedding),
            strength_head=count(self.strength_head),
        )

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )

    def head_parameter_names(self) -> tuple[str, ...]:
        """Parameter names belonging to the output heads.

        Used for the split backbone/head learning rates recommended for
        Human Gold fine-tuning (docs/04_training_plan.md).
        """
        prefixes = ("viseme_head.", "soft_viseme_embedding", "strength_head.")
        return tuple(
            name for name, _ in self.named_parameters() if name.startswith(prefixes)
        )

    @classmethod
    def from_vocabulary(cls, cfg: ModelConfig, vocab: FeatureVocabulary) -> ArticuLMV1:
        return cls(cfg, vocab.sizes())
