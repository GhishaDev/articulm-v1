"""Loss weighting, target normalisation and padding exclusion."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.nn import functional as F

from articulm.config import LossConfig, StrengthLossConfig, VisemeLossConfig
from articulm.data.collator import PhonemeCollator
from articulm.model.articulm_v1 import ArticuLMOutput, ArticuLMV1
from articulm.training.losses import ArticuLMLoss

SYNTHETIC_LOSS = LossConfig(
    viseme=VisemeLossConfig(weight=1.0, label_smoothing=0.05),
    strength=StrengthLossConfig(weight=0.3, beta=0.1),
)
HUMAN_GOLD_LOSS = LossConfig(
    viseme=VisemeLossConfig(weight=1.0, label_smoothing=0.05),
    strength=StrengthLossConfig(weight=1.0, beta=0.1),
)


@pytest.fixture
def batch(dataset):
    return PhonemeCollator()(list(dataset.encoded))


@pytest.fixture
def model(tiny_model_config, vocab) -> ArticuLMV1:
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    model.eval()
    return model


@pytest.fixture
def output(model, batch) -> ArticuLMOutput:
    with torch.no_grad():
        return model(batch.feature_ids, batch.attention_mask)


def test_synthetic_weights_match_docs():
    """docs/05: L = 1.0*L_viseme + 0.3*L_strength."""
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    assert loss_fn.viseme_weight == pytest.approx(1.0)
    assert loss_fn.strength_weight == pytest.approx(0.3)
    assert loss_fn.label_smoothing == pytest.approx(0.05)


def test_human_gold_weights_match_docs():
    """docs/05: L = 1.0*L_viseme + 1.0*L_strength."""
    loss_fn = ArticuLMLoss(HUMAN_GOLD_LOSS)
    assert loss_fn.viseme_weight == pytest.approx(1.0)
    assert loss_fn.strength_weight == pytest.approx(1.0)


def test_total_is_the_weighted_sum(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    breakdown = loss_fn(output, batch)
    expected = 1.0 * float(breakdown.viseme) + 0.3 * float(breakdown.strength)
    assert float(breakdown.total) == pytest.approx(expected, rel=1e-6)


def test_losses_are_finite_and_positive(output, batch):
    breakdown = ArticuLMLoss(SYNTHETIC_LOSS)(output, batch)
    for value in (breakdown.total, breakdown.viseme, breakdown.strength):
        assert torch.isfinite(value)
        assert float(value) >= 0.0


def test_supervised_token_count_excludes_padding(output, batch, dataset):
    breakdown = ArticuLMLoss(SYNTHETIC_LOSS)(output, batch)
    assert breakdown.num_supervised_tokens == sum(dataset.lengths)
    assert breakdown.num_supervised_tokens < batch.batch_size * batch.max_length


# -- padding exclusion -----------------------------------------------------


def test_viseme_loss_matches_manual_masked_cross_entropy(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    computed = loss_fn.viseme_loss(
        output.viseme_logits, batch.viseme_targets, batch.loss_mask
    )
    mask = batch.loss_mask
    expected = F.cross_entropy(
        output.viseme_logits[mask], batch.viseme_targets[mask], label_smoothing=0.05
    )
    assert float(computed) == pytest.approx(float(expected), rel=1e-6)


def test_strength_loss_matches_manual_masked_huber(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    computed, _ = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, batch.loss_mask, None
    )
    mask = batch.loss_mask
    expected = F.smooth_l1_loss(
        output.strength_norm[mask], batch.strength_targets[mask], beta=0.1
    )
    assert float(computed) == pytest.approx(float(expected), rel=1e-6)


def test_garbage_in_padded_slots_does_not_change_the_loss(model, dataset):
    """Corrupt every padded target; the loss must be unchanged."""
    collator = PhonemeCollator()
    batch = collator(list(dataset.encoded))
    with torch.no_grad():
        out = model(batch.feature_ids, batch.attention_mask)
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    reference = float(loss_fn(out, batch).total)

    poisoned = dataclasses.replace(
        batch,
        viseme_targets=batch.viseme_targets.clone(),
        strength_targets=batch.strength_targets.clone(),
    )
    padded = ~batch.attention_mask
    poisoned.viseme_targets[padded] = 7  # a valid class, in the wrong place
    poisoned.strength_targets[padded] = 999.0
    assert float(loss_fn(out, poisoned).total) == pytest.approx(reference, rel=1e-9)


def test_loss_mask_wider_than_attention_mask_is_rejected(output, batch):
    broken = dataclasses.replace(batch, loss_mask=torch.ones_like(batch.attention_mask))
    if bool((~batch.attention_mask).any()):
        with pytest.raises(ValueError, match="outside attention_mask"):
            ArticuLMLoss(SYNTHETIC_LOSS)(output, broken)


def test_ignore_index_targets_would_be_caught(output, batch):
    """If IGNORE_INDEX ever survived the mask, the loss must complain loudly
    rather than silently training on class -100."""
    from articulm.data.collator import IGNORE_INDEX

    targets = batch.viseme_targets.clone()
    targets[batch.loss_mask] = IGNORE_INDEX
    with pytest.raises(ValueError, match=r"outside .0,num_classes."):
        ArticuLMLoss(SYNTHETIC_LOSS).viseme_loss(
            output.viseme_logits, targets, batch.loss_mask
        )


def test_empty_mask_yields_zero_not_nan(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    empty = torch.zeros_like(batch.loss_mask)
    viseme = loss_fn.viseme_loss(output.viseme_logits, batch.viseme_targets, empty)
    strength, weight_sum = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, empty, None
    )
    assert float(viseme) == 0.0
    assert float(strength) == 0.0
    assert weight_sum == 0.0


# -- target normalisation --------------------------------------------------


def test_strength_targets_are_already_normalised(batch):
    assert batch.strength_targets is not None
    mask = batch.loss_mask
    values = batch.strength_targets[mask]
    assert float(values.min()) >= 0.0
    assert float(values.max()) <= 1.0


def test_perfect_prediction_gives_zero_strength_loss(batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    loss, _ = loss_fn.strength_loss(
        batch.strength_targets, batch.strength_targets, batch.loss_mask, None
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-9)


# -- source weighting ------------------------------------------------------


def test_source_weights_scale_the_strength_loss(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    unweighted, _ = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, batch.loss_mask, None
    )
    half = torch.where(
        batch.loss_mask, torch.full_like(batch.strength_targets, 0.5), 0.0
    )
    weighted, weight_sum = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, batch.loss_mask, half
    )
    # A uniform weight is a normalised reweighting, so the mean is unchanged.
    assert float(weighted) == pytest.approx(float(unweighted), rel=1e-5)
    assert weight_sum == pytest.approx(0.5 * batch.num_supervised_tokens)


def test_zero_weight_tokens_are_excluded(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    mask = batch.loss_mask
    weights = torch.zeros_like(batch.strength_targets)
    # Keep only the first token of the first sequence.
    weights[0, 0] = 1.0
    weighted, weight_sum = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, mask, weights
    )
    single = F.smooth_l1_loss(
        output.strength_norm[0, 0], batch.strength_targets[0, 0], beta=0.1
    )
    assert weight_sum == pytest.approx(1.0)
    assert float(weighted) == pytest.approx(float(single), rel=1e-6)


def test_all_zero_weights_yield_zero(output, batch):
    loss_fn = ArticuLMLoss(SYNTHETIC_LOSS)
    weights = torch.zeros_like(batch.strength_targets)
    loss, weight_sum = loss_fn.strength_loss(
        output.strength_norm, batch.strength_targets, batch.loss_mask, weights
    )
    assert float(loss) == 0.0
    assert weight_sum == 0.0


# -- loss variants ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["smooth_l1", "huber", "l1", "mse"])
def test_supported_strength_losses(output, batch, kind):
    config = dataclasses.replace(
        SYNTHETIC_LOSS, strength=StrengthLossConfig(type=kind, weight=0.3, beta=0.1)
    )
    breakdown = ArticuLMLoss(config)(output, batch)
    assert torch.isfinite(breakdown.total)


def test_unsupported_strength_loss_is_rejected(output, batch):
    config = dataclasses.replace(
        SYNTHETIC_LOSS, strength=StrengthLossConfig(type="quantile", weight=0.3)
    )
    with pytest.raises(ValueError, match="unsupported strength loss"):
        ArticuLMLoss(config)(output, batch)


def test_label_smoothing_raises_the_floor(output, batch):
    """With smoothing, even a perfect prediction has non-zero CE."""
    smoothed = ArticuLMLoss(SYNTHETIC_LOSS)
    unsmoothed = ArticuLMLoss(
        dataclasses.replace(
            SYNTHETIC_LOSS, viseme=VisemeLossConfig(weight=1.0, label_smoothing=0.0)
        )
    )
    mask = batch.loss_mask
    confident = F.one_hot(
        batch.viseme_targets.clamp(min=0), num_classes=18
    ).float() * 50.0
    with_smoothing = smoothed.viseme_loss(confident, batch.viseme_targets, mask)
    without = unsmoothed.viseme_loss(confident, batch.viseme_targets, mask)
    assert float(with_smoothing) > float(without)
    assert float(without) == pytest.approx(0.0, abs=1e-5)


def test_unlabelled_batch_is_rejected(model, output, batch):
    stripped = dataclasses.replace(batch, viseme_targets=None, strength_targets=None)
    with pytest.raises(ValueError, match="requires a labelled batch"):
        ArticuLMLoss(SYNTHETIC_LOSS)(output, stripped)


def test_backward_produces_gradients_for_both_heads(tiny_model_config, vocab, batch):
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    model.train()
    out = model(batch.feature_ids, batch.attention_mask)
    ArticuLMLoss(SYNTHETIC_LOSS)(out, batch).total.backward()
    named = dict(model.named_parameters())
    for name in (
        "viseme_head.classifier.weight",
        "strength_head.network.0.weight",
        "soft_viseme_embedding.table",
        "fusion.projection.weight",
    ):
        assert named[name].grad is not None, name
        assert float(named[name].grad.abs().max()) > 0.0, name
