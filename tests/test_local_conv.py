"""Optional local coarticulation conv — ON and OFF must both work."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from articulm.config import ConfigError, LocalConvConfig, to_plain_dict
from articulm.model.local_conv import (
    LocalCoarticulationConv,
    build_local_conv_stack,
    local_conv_insertion_points,
)
from articulm.model.transformer import ContextTransformer


@pytest.fixture
def conv_on_config(tiny_model_config):
    return dataclasses.replace(
        tiny_model_config,
        local_conv=LocalConvConfig(
            enabled=True, every_n_layers=2, kernel_size=5, depthwise=True
        ),
    )


def test_baseline_is_off(model_config):
    assert model_config.local_conv.enabled is False
    assert len(build_local_conv_stack(model_config)) == 0
    assert local_conv_insertion_points(model_config) == ()
    assert ContextTransformer(model_config).local_conv_enabled is False


def test_insertion_points_every_two_layers(model_config):
    enabled = dataclasses.replace(
        model_config, local_conv=dataclasses.replace(model_config.local_conv, enabled=True)
    )
    # 10 layers, every 2 -> after blocks 2, 4, 6, 8, 10 (0-based 1,3,5,7,9).
    assert local_conv_insertion_points(enabled) == (1, 3, 5, 7, 9)
    assert len(build_local_conv_stack(enabled)) == 5


def test_transformer_with_conv_on(conv_on_config):
    model = ContextTransformer(conv_on_config)
    assert model.local_conv_enabled is True
    assert len(model.local_convs) == 1  # 2 layers, every 2
    assert model.local_conv_after == (1,)


def test_conv_on_and_off_produce_same_shapes(tiny_model_config, conv_on_config):
    hidden = torch.randn(2, 9, tiny_model_config.transformer.hidden_size)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[1, 6:] = False

    off = ContextTransformer(tiny_model_config).eval()
    on = ContextTransformer(conv_on_config).eval()
    with torch.no_grad():
        assert off(hidden, mask).shape == on(hidden, mask).shape


def test_conv_on_keeps_padding_at_zero(conv_on_config):
    model = ContextTransformer(conv_on_config).eval()
    hidden = torch.randn(2, 10, conv_on_config.transformer.hidden_size)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 4:] = False
    with torch.no_grad():
        out = model(hidden, mask)
    assert float(out[1, 4:].abs().max()) == 0.0


def test_conv_cannot_leak_padding_into_real_neighbours(conv_on_config):
    """The kernel spans 5 positions, so a PAD next to a real token would leak
    without explicit zeroing."""
    model = ContextTransformer(conv_on_config).eval()
    hidden_size = conv_on_config.transformer.hidden_size
    torch.manual_seed(0)
    hidden = torch.randn(1, 12, hidden_size)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, :6] = True

    with torch.no_grad():
        clean = model(hidden.clone(), mask)
        poisoned = hidden.clone()
        poisoned[0, 6:] = 1e4
        dirty = model(poisoned, mask)
    assert torch.allclose(clean[:, :6], dirty[:, :6], atol=1e-4)


def test_conv_is_depthwise_and_light(conv_on_config):
    conv = LocalCoarticulationConv(
        hidden_size=64, kernel_size=5, depthwise=True, dropout=0.0
    )
    assert conv.conv.groups == 64
    # depthwise: kernel*channels weights + channels biases
    assert conv.conv.weight.numel() == 64 * 5
    assert conv.conv.bias is not None and conv.conv.bias.numel() == 64

    dense = LocalCoarticulationConv(
        hidden_size=64, kernel_size=5, depthwise=False, dropout=0.0
    )
    assert dense.conv.groups == 1
    assert dense.conv.weight.numel() == 64 * 64 * 5
    assert conv.conv.weight.numel() < dense.conv.weight.numel()


def test_conv_uses_same_padding(conv_on_config):
    conv = LocalCoarticulationConv(hidden_size=8, kernel_size=5, dropout=0.0).eval()
    hidden = torch.randn(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    with torch.no_grad():
        assert conv(hidden, mask).shape == hidden.shape


def test_conv_is_residual(conv_on_config):
    """With zeroed conv weights the block must be the identity."""
    conv = LocalCoarticulationConv(hidden_size=8, kernel_size=5, dropout=0.0).eval()
    with torch.no_grad():
        conv.conv.weight.zero_()
        assert conv.conv.bias is not None
        conv.conv.bias.zero_()
    hidden = torch.randn(1, 6, 8)
    mask = torch.ones(1, 6, dtype=torch.bool)
    with torch.no_grad():
        out = conv(hidden, mask)
    # GELU(0) == 0, so the residual passes through untouched.
    assert torch.allclose(out, hidden, atol=1e-6)


def test_conv_mixes_local_context(conv_on_config):
    conv = LocalCoarticulationConv(hidden_size=8, kernel_size=5, dropout=0.0).eval()
    torch.manual_seed(4)
    hidden = torch.randn(1, 9, 8)
    mask = torch.ones(1, 9, dtype=torch.bool)
    with torch.no_grad():
        first = conv(hidden, mask)
        changed = hidden.clone()
        changed[0, 4] = torch.randn(8) * 3
        second = conv(changed, mask)
    # kernel 5 centred on 4 reaches positions 2..6.
    assert not torch.allclose(first[0, 3], second[0, 3], atol=1e-6)
    # ...and must not reach position 8.
    assert torch.allclose(first[0, 8], second[0, 8], atol=1e-6)


def test_conv_on_backward_runs(conv_on_config):
    model = ContextTransformer(conv_on_config)
    hidden = torch.randn(2, 8, conv_on_config.transformer.hidden_size, requires_grad=True)
    mask = torch.ones(2, 8, dtype=torch.bool)
    output = model(hidden, mask)
    (output * torch.randn_like(output)).sum().backward()
    assert hidden.grad is not None
    conv_grads = [
        p.grad for p in model.local_convs.parameters() if p.grad is not None
    ]
    assert conv_grads
    assert any(float(g.abs().max()) > 0 for g in conv_grads)


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError, match="odd"):
        LocalCoarticulationConv(hidden_size=8, kernel_size=4)


def test_even_kernel_rejected_by_config(tmp_path, model_config):
    import yaml

    from articulm.config import load_model_config

    broken = dataclasses.replace(model_config.local_conv, kernel_size=4)
    raw = to_plain_dict(dataclasses.replace(model_config, local_conv=broken))
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"model": raw}), encoding="utf-8")
    with pytest.raises(ConfigError, match="kernel_size must be odd"):
        load_model_config(path)
