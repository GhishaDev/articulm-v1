"""End-to-end model wiring, shapes, parameter count and padding isolation."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.nn import functional as F

from articulm.config import LocalConvConfig
from articulm.data.collator import PhonemeCollator
from articulm.model.articulm_v1 import ArticuLMV1


@pytest.fixture
def batch(dataset):
    return PhonemeCollator()(list(dataset.encoded))


@pytest.fixture
def tiny_model(tiny_model_config, vocab) -> ArticuLMV1:
    return ArticuLMV1.from_vocabulary(tiny_model_config, vocab)


@pytest.fixture(scope="module")
def baseline_model(model_config, vocab) -> ArticuLMV1:
    """The real 10x640 model, built once per module (it is ~50M params)."""
    return ArticuLMV1.from_vocabulary(model_config, vocab)


# -- shapes ---------------------------------------------------------------


def test_forward_shapes(baseline_model, batch):
    baseline_model.eval()
    with torch.no_grad():
        out = baseline_model(batch.feature_ids, batch.attention_mask)
    b, t = batch.batch_size, batch.max_length
    assert out.hidden_states.shape == (b, t, 640)
    assert out.viseme_logits.shape == (b, t, 18)
    assert out.viseme_probabilities.shape == (b, t, 18)
    assert out.soft_viseme_embedding.shape == (b, t, 32)
    assert out.strength_norm.shape == (b, t)
    assert out.batch_size == b
    assert out.seq_len == t


def test_strength_is_in_unit_range_then_scaled(baseline_model, batch):
    baseline_model.eval()
    with torch.no_grad():
        out = baseline_model(batch.feature_ids, batch.attention_mask)
    assert float(out.strength_norm.min()) >= 0.0
    assert float(out.strength_norm.max()) <= 1.0
    scaled = out.strength_scaled(100.0)
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 100.0


def test_predict_returns_ids_and_0_100_strength(baseline_model, batch):
    ids, strength = baseline_model.predict(batch.feature_ids, batch.attention_mask)
    assert ids.shape == (batch.batch_size, batch.max_length)
    assert ids.dtype == torch.long
    assert int(ids.min()) >= 0 and int(ids.max()) <= 17
    assert float(strength.min()) >= 0.0 and float(strength.max()) <= 100.0


def test_predict_restores_training_mode(baseline_model, batch):
    baseline_model.train()
    baseline_model.predict(batch.feature_ids, batch.attention_mask)
    assert baseline_model.training is True
    baseline_model.eval()


# -- parameter count -------------------------------------------------------


def test_parameter_count_is_about_50m(baseline_model):
    total = baseline_model.num_parameters()
    breakdown = baseline_model.parameter_breakdown()
    assert breakdown.total == total
    # 10 x 640 transformer alone is ~49.2M (docs/01: ~4.915M per layer).
    assert 49_000_000 <= breakdown.transformer <= 49_500_000
    assert 45_000_000 <= total <= 55_000_000


def test_transformer_per_layer_parameter_count(baseline_model):
    breakdown = baseline_model.parameter_breakdown()
    # Subtract the final Pre-LN norm (2 * hidden) before dividing.
    per_layer = (breakdown.transformer - 2 * 640) / 10
    assert per_layer == pytest.approx(4_923_520, rel=1e-6)


def test_head_parameter_names_cover_all_three_heads(baseline_model):
    names = baseline_model.head_parameter_names()
    assert any(n.startswith("viseme_head.") for n in names)
    assert any(n.startswith("strength_head.") for n in names)
    assert any(n.startswith("soft_viseme_embedding") for n in names)
    all_names = {n for n, _ in baseline_model.named_parameters()}
    assert set(names) <= all_names


# -- padding isolation -----------------------------------------------------


def test_padded_hidden_states_are_zero(tiny_model, batch):
    tiny_model.eval()
    with torch.no_grad():
        out = tiny_model(batch.feature_ids, batch.attention_mask)
    padded = ~batch.attention_mask
    if bool(padded.any()):
        assert float(out.hidden_states[padded].abs().max()) == 0.0


def test_padding_does_not_change_real_predictions(tiny_model, dataset):
    """Batching a short sentence with a long one must not alter its outputs."""
    tiny_model.eval()
    items = sorted(dataset.encoded, key=lambda item: item.length)
    short, long_item = items[0], items[-1]
    assert short.length < long_item.length

    collator = PhonemeCollator()
    alone = collator([short])
    together = collator([short, long_item])

    with torch.no_grad():
        solo = tiny_model(alone.feature_ids, alone.attention_mask)
        pair = tiny_model(together.feature_ids, together.attention_mask)

    length = short.length
    assert torch.allclose(
        solo.viseme_logits[0, :length], pair.viseme_logits[0, :length], atol=1e-4
    )
    assert torch.allclose(
        solo.strength_norm[0, :length], pair.strength_norm[0, :length], atol=1e-5
    )


def test_batch_order_does_not_change_predictions(tiny_model, dataset):
    tiny_model.eval()
    items = list(dataset.encoded)[:3]
    collator = PhonemeCollator()
    forward = collator(items)
    reversed_batch = collator(items[::-1])
    with torch.no_grad():
        a = tiny_model(forward.feature_ids, forward.attention_mask)
        b = tiny_model(reversed_batch.feature_ids, reversed_batch.attention_mask)
    for index, item in enumerate(items):
        mirrored = len(items) - 1 - index
        length = item.length
        assert torch.allclose(
            a.viseme_logits[index, :length],
            b.viseme_logits[mirrored, :length],
            atol=1e-4,
        )


# -- gradient flow through the whole model --------------------------------


def test_strength_only_loss_reaches_viseme_head(tiny_model, batch):
    """No hard argmax on the training path, verified on the full model."""
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    out = tiny_model(batch.feature_ids, batch.attention_mask)
    assert batch.strength_targets is not None
    mask = batch.loss_mask
    loss = F.smooth_l1_loss(out.strength_norm[mask], batch.strength_targets[mask])
    loss.backward()

    for name in (
        "viseme_head.classifier.weight",
        "viseme_head.input_projection.weight",
        "soft_viseme_embedding.table",
    ):
        parameter = dict(tiny_model.named_parameters())[name]
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().max()) > 0.0, name


def test_viseme_only_loss_does_not_touch_strength_head(tiny_model, batch):
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    out = tiny_model(batch.feature_ids, batch.attention_mask)
    assert batch.viseme_targets is not None
    mask = batch.loss_mask
    F.cross_entropy(
        out.viseme_logits[mask], batch.viseme_targets[mask]
    ).backward()
    for name, parameter in tiny_model.named_parameters():
        if name.startswith("strength_head."):
            assert parameter.grad is None or float(parameter.grad.abs().max()) == 0.0


def test_padding_contributes_no_gradient(tiny_model, dataset):
    """A gradient computed only on masked tokens must be identical whether or
    not extra padding is present."""
    tiny_model.eval()  # disable dropout for determinism
    items = sorted(dataset.encoded, key=lambda item: item.length)
    short, long_item = items[0], items[-1]
    collator = PhonemeCollator()

    def grad_for(batch, row: int, length: int) -> torch.Tensor:
        tiny_model.zero_grad(set_to_none=True)
        out = tiny_model(batch.feature_ids, batch.attention_mask)
        assert batch.viseme_targets is not None
        logits = out.viseme_logits[row, :length]
        targets = batch.viseme_targets[row, :length]
        F.cross_entropy(logits, targets).backward()
        return tiny_model.viseme_head.classifier.weight.grad.clone()

    alone = grad_for(collator([short]), 0, short.length)
    padded = grad_for(collator([short, long_item]), 0, short.length)
    assert torch.allclose(alone, padded, atol=1e-5)


# -- config-driven variants ------------------------------------------------


def test_local_conv_on_changes_parameter_count(tiny_model_config, vocab):
    off = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    on_config = dataclasses.replace(
        tiny_model_config,
        local_conv=LocalConvConfig(enabled=True, every_n_layers=2, kernel_size=5),
    )
    on = ArticuLMV1.from_vocabulary(on_config, vocab)
    assert on.num_parameters() > off.num_parameters()
    assert on.transformer.local_conv_enabled is True
    assert off.transformer.local_conv_enabled is False


def test_local_conv_on_full_baseline_adds_expected_params(model_config, vocab):
    off = ArticuLMV1.from_vocabulary(model_config, vocab)
    on_config = dataclasses.replace(
        model_config, local_conv=dataclasses.replace(model_config.local_conv, enabled=True)
    )
    on = ArticuLMV1.from_vocabulary(on_config, vocab)
    # 5 insertions x (LayerNorm 2*640 + depthwise conv 640*5 + bias 640)
    expected = 5 * (2 * 640 + 640 * 5 + 640)
    assert on.num_parameters() - off.num_parameters() == expected


def test_sequence_longer_than_max_seq_len_is_rejected(tiny_model, tiny_model_config):
    too_long = tiny_model_config.input.max_seq_len + 1
    from articulm.data.vocab import FEATURE_KEYS

    ids = torch.ones((1, too_long, len(FEATURE_KEYS)), dtype=torch.long)
    mask = torch.ones((1, too_long), dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        tiny_model(ids, mask)


def test_integer_attention_mask_is_accepted(tiny_model, batch):
    tiny_model.eval()
    with torch.no_grad():
        from_bool = tiny_model(batch.feature_ids, batch.attention_mask)
        from_int = tiny_model(batch.feature_ids, batch.attention_mask.long())
    assert torch.allclose(from_bool.viseme_logits, from_int.viseme_logits, atol=1e-6)


def test_model_is_deterministic_in_eval(tiny_model, batch):
    tiny_model.eval()
    with torch.no_grad():
        first = tiny_model(batch.feature_ids, batch.attention_mask)
        second = tiny_model(batch.feature_ids, batch.attention_mask)
    assert torch.equal(first.viseme_logits, second.viseme_logits)
    assert torch.equal(first.strength_norm, second.strength_norm)
