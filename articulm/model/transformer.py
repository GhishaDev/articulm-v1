"""10-layer Pre-LN context transformer encoder with configurable positions.

Baseline (``config/model_v1_50m.yaml``):

```text
hidden 640 / 10 layers / 10 heads / head_dim 64 / FFN 2560
dropout 0.1 / attention_dropout 0.1 / GELU / pre_layer_norm / RoPE
```

Layer count and hidden size are never changed silently: mismatches are
rejected by :func:`articulm.config.ModelConfig.validate`.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig
from .fusion import build_activation
from .local_conv import build_local_conv_stack, local_conv_insertion_points
from .tracing import is_graph_capture


class RotaryPositionEmbedding(nn.Module):
    """RoPE applied to the query/key head dimension."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = base
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        # Build on the buffer's own device: after `.to("cuda")` the frequencies
        # live on the GPU, and a CPU `positions` would raise a device mismatch.
        device = self.inverse_frequency.device
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, self.inverse_frequency)  # [T, head_dim/2]
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)
        self._cached_len = seq_len

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        cos = self.cos_cache[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cache[:seq_len].to(device=device, dtype=dtype)
        # Broadcast over [B, heads, T, head_dim/2].
        return cos[None, None, :, :], sin[None, None, :, :]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Rotate ``[B, heads, T, head_dim]`` in interleaved even/odd pairs."""
        batch, heads, seq_len, dim = tensor.shape
        if dim != self.head_dim:
            raise ValueError(f"RoPE expected head_dim {self.head_dim}, got {dim}")
        cos, sin = self._cos_sin(seq_len, tensor.device, tensor.dtype)

        reshaped = tensor.reshape(batch, heads, seq_len, dim // 2, 2)
        even = reshaped[..., 0]
        odd = reshaped[..., 1]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).reshape(
            batch, heads, seq_len, dim
        )


class SinusoidalPositionEncoding(nn.Module):
    """Additive fixed sinusoidal encoding (alternative to RoPE)."""

    def __init__(self, hidden_size: int, max_seq_len: int) -> None:
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        encoding = torch.zeros(max_seq_len, hidden_size)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.shape[1]
        if seq_len > self.encoding.shape[0]:
            raise ValueError(
                f"sequence length {seq_len} exceeds sinusoidal cache "
                f"{self.encoding.shape[0]}"
            )
        return hidden + self.encoding[:seq_len].to(dtype=hidden.dtype)


class MultiHeadSelfAttention(nn.Module):
    """Standard MHSA with optional RoPE and a key-padding mask."""

    def __init__(self, cfg: ModelConfig, rotary: RotaryPositionEmbedding | None) -> None:
        super().__init__()
        tf = cfg.transformer
        self.num_heads = tf.num_heads
        self.head_dim = tf.head_dim
        self.hidden_size = tf.hidden_size
        self.attention_dropout = tf.attention_dropout

        self.query = nn.Linear(tf.hidden_size, tf.num_heads * tf.head_dim)
        self.key = nn.Linear(tf.hidden_size, tf.num_heads * tf.head_dim)
        self.value = nn.Linear(tf.hidden_size, tf.num_heads * tf.head_dim)
        self.output = nn.Linear(tf.num_heads * tf.head_dim, tf.hidden_size)
        self.rotary = rotary
        self.residual_dropout = nn.Dropout(tf.dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B,T,H]``; ``attention_mask`` ``[B,T]`` bool, True = real."""
        batch, seq_len, _ = hidden.shape

        query = self._split_heads(self.query(hidden))
        key = self._split_heads(self.key(hidden))
        value = self._split_heads(self.value(hidden))

        if self.rotary is not None:
            query = self.rotary(query)
            key = self.rotary(key)

        # Broadcast key-padding mask to [B, 1, 1, T]; True means "may attend".
        key_mask = attention_mask[:, None, None, :]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.residual_dropout(self.output(merged))


class FeedForward(nn.Module):
    """Position-wise FFN: ``H -> ffn_size -> H``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        tf = cfg.transformer
        self.up = nn.Linear(tf.hidden_size, tf.ffn_size)
        self.activation = build_activation(tf.activation)
        self.down = nn.Linear(tf.ffn_size, tf.hidden_size)
        self.dropout = nn.Dropout(tf.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.activation(self.up(hidden))))


class TransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm encoder block."""

    def __init__(self, cfg: ModelConfig, rotary: RotaryPositionEmbedding | None) -> None:
        super().__init__()
        hidden_size = cfg.transformer.hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(cfg, rotary)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(cfg)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), attention_mask)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class ContextTransformer(nn.Module):
    """Stack of Pre-LN blocks with optional interleaved local convolutions."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        tf = cfg.transformer
        self.hidden_size = tf.hidden_size
        self.num_layers = tf.num_layers
        self.position_type = cfg.position.type

        self.rotary: RotaryPositionEmbedding | None = None
        self.sinusoidal: SinusoidalPositionEncoding | None = None
        self.learned_positions: nn.Embedding | None = None
        if self.position_type == "rope":
            self.rotary = RotaryPositionEmbedding(tf.head_dim, cfg.input.max_seq_len)
        elif self.position_type == "sinusoidal":
            self.sinusoidal = SinusoidalPositionEncoding(tf.hidden_size, cfg.input.max_seq_len)
        elif self.position_type == "learned":
            self.learned_positions = nn.Embedding(cfg.input.max_seq_len, tf.hidden_size)
            nn.init.normal_(self.learned_positions.weight, mean=0.0, std=0.02)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(cfg, self.rotary) for _ in range(tf.num_layers)]
        )
        # Pre-LN stacks need a final norm before the heads read the residual.
        self.final_norm = nn.LayerNorm(tf.hidden_size)

        self.local_convs = build_local_conv_stack(cfg)
        self.local_conv_after = local_conv_insertion_points(cfg)
        if len(self.local_convs) != len(self.local_conv_after):
            raise ValueError(
                f"local conv count {len(self.local_convs)} does not match insertion "
                f"points {self.local_conv_after}"
            )
        self._conv_by_layer = {
            layer_index: conv_index
            for conv_index, layer_index in enumerate(self.local_conv_after)
        }
        # Eager-only value assertions; see model/tracing.py.
        self.validate_inputs = True

    @property
    def local_conv_enabled(self) -> bool:
        return len(self.local_convs) > 0

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``hidden`` ``[B,T,H]``, ``attention_mask`` ``[B,T]`` bool (True = real)."""
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                f"ContextTransformer expected hidden {self.hidden_size}, got {hidden.shape[-1]}"
            )
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} does not match "
                f"hidden {tuple(hidden.shape[:2])}"
            )
        # Value-dependent check: an all-PAD row makes attention undefined.
        # Skipped during graph capture, where tensor values are unavailable.
        if (
            self.validate_inputs
            and not is_graph_capture()
            and not attention_mask.any(dim=1).all()
        ):
            raise ValueError("every sequence must contain at least one unpadded token")

        keep = attention_mask.unsqueeze(-1).to(hidden.dtype)

        if self.sinusoidal is not None:
            hidden = self.sinusoidal(hidden)
        elif self.learned_positions is not None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            hidden = hidden + self.learned_positions(positions)[None, :, :]

        # Keep PAD positions at exactly zero so no module (in particular the
        # local conv) can pull padding into a real neighbour.
        hidden = hidden * keep

        for layer_index, layer in enumerate(self.layers):
            hidden = layer(hidden, attention_mask) * keep
            conv_index = self._conv_by_layer.get(layer_index)
            if conv_index is not None:
                hidden = self.local_convs[conv_index](hidden, attention_mask)

        return self.final_norm(hidden) * keep
