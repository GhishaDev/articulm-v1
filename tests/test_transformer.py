"""Context transformer: structure, padding mask correctness, RoPE."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from articulm.config import PositionConfig
from articulm.model.transformer import (
    ContextTransformer,
    MultiHeadSelfAttention,
    RotaryPositionEmbedding,
    TransformerEncoderLayer,
)


@pytest.fixture
def transformer(tiny_model_config) -> ContextTransformer:
    model = ContextTransformer(tiny_model_config)
    model.eval()
    return model


def test_baseline_stack_structure(model_config):
    model = ContextTransformer(model_config)
    assert len(model.layers) == 10
    assert model.hidden_size == 640
    for layer in model.layers:
        assert isinstance(layer, TransformerEncoderLayer)
        assert layer.attention.num_heads == 10
        assert layer.attention.head_dim == 64
        assert layer.ffn.up.out_features == 2560
        assert layer.ffn.down.out_features == 640
        # Pre-LN: normalisation happens before attention and before the FFN.
        assert isinstance(layer.attention_norm, torch.nn.LayerNorm)
        assert isinstance(layer.ffn_norm, torch.nn.LayerNorm)
    assert isinstance(model.final_norm, torch.nn.LayerNorm)


def test_layer_count_is_not_changed_silently(model_config):
    """A 10x640 baseline must stay 10x640 unless the config says otherwise."""
    model = ContextTransformer(model_config)
    assert model.num_layers == model_config.transformer.num_layers == 10


def test_forward_shape(transformer, tiny_model_config):
    hidden = torch.randn(3, 9, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(3, 9, dtype=torch.bool)
    out = transformer(hidden, mask)
    assert out.shape == hidden.shape


def test_padded_positions_are_exactly_zero(transformer, tiny_model_config):
    hidden = torch.randn(2, 6, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    out = transformer(hidden, mask)
    assert float(out[1, 4:].abs().max()) == 0.0


def test_padding_does_not_change_real_token_outputs(transformer, tiny_model_config):
    """Appending PAD must leave real positions bit-identical."""
    torch.manual_seed(0)
    hidden_size = tiny_model_config.transformer.hidden_size
    short = torch.randn(1, 5, hidden_size)
    short_mask = torch.ones(1, 5, dtype=torch.bool)

    padded = torch.cat([short, torch.randn(1, 4, hidden_size)], dim=1)
    padded_mask = torch.zeros(1, 9, dtype=torch.bool)
    padded_mask[0, :5] = True

    reference = transformer(short, short_mask)
    with_padding = transformer(padded, padded_mask)
    assert torch.allclose(reference, with_padding[:, :5], atol=1e-5)


def test_garbage_in_padded_slots_cannot_leak(transformer, tiny_model_config):
    """Huge values in PAD slots must not move real outputs."""
    hidden_size = tiny_model_config.transformer.hidden_size
    torch.manual_seed(1)
    hidden = torch.randn(1, 8, hidden_size)
    mask = torch.zeros(1, 8, dtype=torch.bool)
    mask[0, :4] = True

    clean = transformer(hidden.clone(), mask)
    poisoned = hidden.clone()
    poisoned[0, 4:] = 1e4
    dirty = transformer(poisoned, mask)
    assert torch.allclose(clean[:, :4], dirty[:, :4], atol=1e-4)


def test_all_padding_row_is_rejected(transformer, tiny_model_config):
    hidden = torch.randn(2, 5, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError, match="at least one unpadded token"):
        transformer(hidden, mask)


def test_mask_shape_mismatch_is_rejected(transformer, tiny_model_config):
    hidden = torch.randn(2, 5, tiny_model_config.transformer.hidden_size)
    with pytest.raises(ValueError, match="does not match"):
        transformer(hidden, torch.ones(2, 4, dtype=torch.bool))


def test_wrong_hidden_width_is_rejected(transformer):
    with pytest.raises(ValueError, match="expected hidden"):
        transformer(torch.randn(1, 4, 7), torch.ones(1, 4, dtype=torch.bool))


def test_attention_is_context_sensitive(transformer, tiny_model_config):
    """Changing a neighbour must change a token's output — that is the point
    of a context transformer.

    The perturbation must not be a constant offset across channels: Pre-LN
    correctly ignores those, since LayerNorm removes the per-token mean.
    """
    hidden_size = tiny_model_config.transformer.hidden_size
    torch.manual_seed(2)
    base = torch.randn(1, 6, hidden_size)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        first = transformer(base, mask)

        changed = base.clone()
        changed[0, 3] = torch.randn(hidden_size) * 3.0
        second = transformer(changed, mask)

    # Every other position must react to the changed neighbour.
    for position in (0, 1, 2, 4, 5):
        assert not torch.allclose(
            first[0, position], second[0, position], atol=1e-5
        ), f"position {position} ignored its context"


def test_gradient_flows_from_every_output_position_to_every_input_position(
    transformer, tiny_model_config
):
    """Attention must connect all unpadded positions, both directions."""
    hidden_size = tiny_model_config.transformer.hidden_size
    torch.manual_seed(3)
    hidden = torch.randn(1, 5, hidden_size, requires_grad=True)
    mask = torch.ones(1, 5, dtype=torch.bool)
    output = transformer(hidden, mask)
    # A random projection avoids LayerNorm's sum-invariance masking the signal.
    (output[0, 0] * torch.randn(hidden_size)).sum().backward()
    assert hidden.grad is not None
    for position in range(5):
        assert float(hidden.grad[0, position].abs().max()) > 0.0


# -- position encodings -----------------------------------------------------


def test_rope_is_used_by_default(model_config):
    model = ContextTransformer(model_config)
    assert model.position_type == "rope"
    assert isinstance(model.rotary, RotaryPositionEmbedding)
    assert model.sinusoidal is None
    assert model.learned_positions is None


def test_rope_preserves_shape_and_norm():
    rope = RotaryPositionEmbedding(head_dim=16, max_seq_len=32)
    tensor = torch.randn(2, 3, 12, 16)
    rotated = rope(tensor)
    assert rotated.shape == tensor.shape
    # Rotation is norm-preserving per 2D pair, hence per vector.
    assert torch.allclose(rotated.norm(dim=-1), tensor.norm(dim=-1), atol=1e-5)


def test_rope_position_zero_is_identity():
    rope = RotaryPositionEmbedding(head_dim=8, max_seq_len=16)
    tensor = torch.randn(1, 1, 4, 8)
    rotated = rope(tensor)
    assert torch.allclose(rotated[0, 0, 0], tensor[0, 0, 0], atol=1e-6)


def test_rope_differentiates_positions():
    rope = RotaryPositionEmbedding(head_dim=8, max_seq_len=16)
    repeated = torch.ones(1, 1, 4, 8)
    rotated = rope(repeated)
    assert not torch.allclose(rotated[0, 0, 0], rotated[0, 0, 1], atol=1e-4)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryPositionEmbedding(head_dim=7, max_seq_len=8)


def test_rope_extends_cache_beyond_initial_length():
    rope = RotaryPositionEmbedding(head_dim=8, max_seq_len=4)
    out = rope(torch.randn(1, 1, 12, 8))
    assert out.shape == (1, 1, 12, 8)


@pytest.mark.parametrize("position_type", ["sinusoidal", "learned", "none"])
def test_alternative_position_encodings_run(tiny_model_config, position_type):
    cfg = dataclasses.replace(
        tiny_model_config, position=PositionConfig(type=position_type)
    )
    model = ContextTransformer(cfg)
    model.eval()
    hidden = torch.randn(2, 7, cfg.transformer.hidden_size)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 5:] = False
    out = model(hidden, mask)
    assert out.shape == hidden.shape
    assert float(out[1, 5:].abs().max()) == 0.0


def test_attention_module_shapes(tiny_model_config):
    attention = MultiHeadSelfAttention(tiny_model_config, rotary=None)
    hidden = torch.randn(2, 5, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(2, 5, dtype=torch.bool)
    assert attention(hidden, mask).shape == hidden.shape


def test_dropout_is_disabled_in_eval(tiny_model_config):
    model = ContextTransformer(tiny_model_config)
    model.eval()
    hidden = torch.randn(1, 6, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(1, 6, dtype=torch.bool)
    assert torch.allclose(model(hidden, mask), model(hidden, mask))
