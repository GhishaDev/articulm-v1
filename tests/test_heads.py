"""Viseme head, Soft Viseme Embedding and Strength head.

The gradient-flow tests here are the load-bearing ones: they prove the
Strength loss reaches the Viseme logits through the *soft* embedding, i.e.
that no hard argmax sits on the training path.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.nn import functional as F

from articulm.model.heads import SoftVisemeEmbedding, StrengthHead, VisemeHead


@pytest.fixture
def viseme_head(model_config) -> VisemeHead:
    return VisemeHead(model_config)


@pytest.fixture
def soft_embedding(model_config) -> SoftVisemeEmbedding:
    return SoftVisemeEmbedding(model_config)


@pytest.fixture
def strength_head(model_config) -> StrengthHead:
    return StrengthHead(model_config)


# -- Viseme head ------------------------------------------------------------


def test_viseme_head_layer_widths(viseme_head):
    assert viseme_head.input_projection.in_features == 640
    assert viseme_head.input_projection.out_features == 256
    assert viseme_head.classifier.in_features == 256
    assert viseme_head.classifier.out_features == 18
    assert isinstance(viseme_head.activation, torch.nn.GELU)
    assert isinstance(viseme_head.dropout, torch.nn.Dropout)


def test_viseme_head_outputs_logits_not_probabilities(viseme_head):
    viseme_head.eval()
    with torch.no_grad():
        logits = viseme_head(torch.randn(2, 5, 640) * 3.0)
    assert logits.shape == (2, 5, 18)
    # Raw logits are unbounded and do not sum to one.
    assert not torch.allclose(
        logits.sum(dim=-1), torch.ones(2, 5), atol=1e-3
    )


def test_viseme_head_argmax_is_the_inference_rule(viseme_head):
    viseme_head.eval()
    with torch.no_grad():
        logits = viseme_head(torch.randn(1, 4, 640))
    predicted = logits.argmax(dim=-1)
    assert predicted.shape == (1, 4)
    assert int(predicted.min()) >= 0 and int(predicted.max()) <= 17


# -- Soft Viseme Embedding --------------------------------------------------


def test_soft_embedding_table_is_18_by_32(soft_embedding):
    assert soft_embedding.table.shape == (18, 32)
    assert soft_embedding.table.requires_grad is True
    assert soft_embedding.num_embeddings == 18
    assert soft_embedding.dim == 32


def test_soft_embedding_output_shape(soft_embedding):
    probabilities, embedded = soft_embedding(torch.randn(3, 7, 18))
    assert probabilities.shape == (3, 7, 18)
    assert embedded.shape == (3, 7, 32)


def test_probabilities_are_a_softmax(soft_embedding):
    logits = torch.randn(2, 4, 18)
    probabilities, _ = soft_embedding(logits)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(probabilities, F.softmax(logits, dim=-1), atol=1e-6)


def test_soft_embedding_is_probability_weighted_sum(soft_embedding):
    logits = torch.randn(2, 3, 18)
    probabilities, embedded = soft_embedding(logits)
    expected = probabilities @ soft_embedding.table
    assert torch.allclose(embedded, expected, atol=1e-6)


def test_soft_embedding_is_not_a_hard_lookup(soft_embedding):
    """A one-hot lookup would ignore the runner-up class entirely."""
    logits = torch.zeros(1, 1, 18)
    logits[0, 0, 3] = 1.0
    logits[0, 0, 7] = 0.9
    _, soft = soft_embedding(logits)
    hard = soft_embedding.table[3].unsqueeze(0).unsqueeze(0)
    assert not torch.allclose(soft, hard, atol=1e-3)


def test_soft_embedding_gradient_reaches_logits(soft_embedding):
    logits = torch.randn(2, 4, 18, requires_grad=True)
    _, embedded = soft_embedding(logits)
    (embedded * torch.randn_like(embedded)).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().max()) > 0.0


def test_soft_embedding_rejects_hard_argmax_mode(model_config):
    broken = dataclasses.replace(model_config.viseme_embedding, mode="hard_argmax")
    with pytest.raises(ValueError, match="soft_probability_weighted"):
        SoftVisemeEmbedding(dataclasses.replace(model_config, viseme_embedding=broken))


def test_soft_embedding_rejects_wrong_logit_width(soft_embedding):
    with pytest.raises(ValueError, match="expected 18 viseme logits"):
        soft_embedding(torch.randn(1, 2, 8))


# -- Strength head ----------------------------------------------------------


def test_strength_head_layer_widths(strength_head):
    linears = [m for m in strength_head.network if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [
        (672, 256),
        (256, 64),
        (64, 1),
    ]


def test_strength_head_dropout_placement(strength_head):
    """docs/01 section 9: dropout after the 672->256 block only."""
    kinds = [type(m).__name__ for m in strength_head.network]
    assert kinds == ["Linear", "GELU", "Dropout", "Linear", "GELU", "Linear"]


def test_strength_head_outputs_0_to_1(strength_head):
    strength_head.eval()
    with torch.no_grad():
        out = strength_head(torch.randn(2, 6, 640) * 20, torch.randn(2, 6, 32) * 20)
    assert out.shape == (2, 6)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_strength_head_scales_to_0_100(strength_head):
    assert strength_head.output_scale == pytest.approx(100.0)
    strength_head.eval()
    with torch.no_grad():
        scaled = strength_head(torch.randn(1, 3, 640), torch.randn(1, 3, 32)) * 100.0
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 100.0


def test_strength_head_consumes_the_soft_viseme_embedding(strength_head):
    """Changing only the soft viseme embedding must change the prediction."""
    strength_head.eval()
    hidden = torch.randn(1, 3, 640)
    with torch.no_grad():
        first = strength_head(hidden, torch.zeros(1, 3, 32))
        second = strength_head(hidden, torch.ones(1, 3, 32))
    assert not torch.allclose(first, second, atol=1e-5)


def test_strength_head_rejects_wrong_input_width(model_config):
    broken = dataclasses.replace(model_config.strength_head, input_dim=700)
    with pytest.raises(ValueError, match="input_dim"):
        StrengthHead(dataclasses.replace(model_config, strength_head=broken))


# -- the full head chain ----------------------------------------------------


def test_strength_loss_backpropagates_through_soft_viseme_path(
    viseme_head, soft_embedding, strength_head
):
    """The non-negotiable one: Strength supervision must shape viseme logits.

    Gradient must reach the Viseme Head weights via the soft embedding even
    when no Viseme loss is present at all.
    """
    hidden = torch.randn(2, 5, 640)
    logits = viseme_head(hidden)
    _, soft = soft_embedding(logits)
    strength = strength_head(hidden, soft)

    target = torch.rand_like(strength)
    F.smooth_l1_loss(strength, target).backward()

    classifier_grad = viseme_head.classifier.weight.grad
    projection_grad = viseme_head.input_projection.weight.grad
    table_grad = soft_embedding.table.grad

    assert classifier_grad is not None and float(classifier_grad.abs().max()) > 0.0
    assert projection_grad is not None and float(projection_grad.abs().max()) > 0.0
    assert table_grad is not None and float(table_grad.abs().max()) > 0.0


def test_detached_argmax_path_would_kill_the_gradient(
    viseme_head, soft_embedding, strength_head
):
    """Contrast case: a hard lookup severs the gradient we just verified.

    This documents *why* hard argmax is forbidden before the Strength Head.
    """
    hidden = torch.randn(2, 5, 640)
    logits = viseme_head(hidden)
    hard_ids = logits.argmax(dim=-1)
    hard = soft_embedding.table[hard_ids]
    strength = strength_head(hidden, hard)
    F.smooth_l1_loss(strength, torch.rand_like(strength)).backward()

    assert viseme_head.classifier.weight.grad is None or float(
        viseme_head.classifier.weight.grad.abs().max()
    ) == 0.0
