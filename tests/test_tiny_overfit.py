"""Tiny-overfit gate (docs/04 Stage 1, docs/09 M2).

This is the gate that must pass before any long training run:

* Viseme train accuracy approaches a near-perfect fit
* Strength loss falls strongly
* no NaN / Inf
* no padding leakage
* checkpoint reload reproduces predictions

The test here uses the small config so it runs in seconds. The full 50M gate
is a separate command, documented in the README:

```bash
python -m articulm.train --config config/train_tiny_overfit.yaml
```
"""

from __future__ import annotations

import math

import pytest
import torch

from articulm.config import (
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    StrengthLossConfig,
    VisemeLossConfig,
)
from articulm.data.collator import PhonemeCollator, build_dataloader
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.training.checkpoint import (
    TrainingState,
    load_checkpoint,
    restore_into,
    save_checkpoint,
)
from articulm.training.losses import ArticuLMLoss
from articulm.training.metrics import masked_accuracy, masked_strength_mae
from articulm.training.optimizer import build_optimizer
from articulm.training.scheduler import build_scheduler

STEPS = 220
# Tiny overfit intentionally disables label smoothing so the CE floor is 0.
TINY_LOSS = LossConfig(
    viseme=VisemeLossConfig(weight=1.0, label_smoothing=0.0),
    strength=StrengthLossConfig(weight=1.0, beta=0.1),
)


@pytest.fixture(scope="module")
def overfit_run(request):
    """Train the small model to convergence on the fixture sentences."""
    import dataclasses
    from pathlib import Path

    from articulm.config import (
        ArticulatoryEmbeddingDims,
        EmbeddingDims,
        InputConfig,
        StrengthHeadConfig,
        TransformerConfig,
        load_data_config,
        load_model_config,
    )
    from articulm.data.dataset import PhonemeSequenceDataset
    from articulm.data.schema import load_samples
    from articulm.data.vocab import build_vocabulary

    root = Path(__file__).resolve().parent
    data_cfg = load_data_config(root.parent / "config" / "data_v1.yaml")
    samples: list = []
    for name in ("sample_zh.jsonl", "sample_en.jsonl", "sample_mixed.jsonl"):
        samples.extend(load_samples(root / "fixtures" / name, data_cfg))
    vocab = build_vocabulary(samples, data_cfg)
    dataset = PhonemeSequenceDataset(samples, vocab)

    base = load_model_config(root.parent / "config" / "model_v1_50m.yaml")
    articulatory = ArticulatoryEmbeddingDims(
        type=4, height=4, backness=4, rounded=2, place=6, manner=6, voiced=2, aspirated=2
    )
    dims = EmbeddingDims(
        phoneme=32,
        language=4,
        surface_tone=8,
        stress=4,
        syllable_role=8,
        articulatory=articulatory,
        boundary=10,
    )
    cfg = dataclasses.replace(
        base,
        input=InputConfig(
            max_seq_len=128, embedding_dims=dims, fused_input_dim=dims.total
        ),
        fusion=dataclasses.replace(base.fusion, output_dim=128, dropout=0.0),
        transformer=TransformerConfig(
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            head_dim=32,
            ffn_size=256,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        viseme_head=dataclasses.replace(base.viseme_head, hidden_size=64, dropout=0.0),
        strength_head=StrengthHeadConfig(
            input_dim=128 + base.viseme_embedding.dim,
            hidden_dims=(64, 32),
            dropout=0.0,
        ),
    )

    torch.manual_seed(1234)
    model = ArticuLMV1.from_vocabulary(cfg, vocab)
    optimizer = build_optimizer(
        model,
        OptimizerConfig(learning_rate=3e-3, weight_decay=0.0),
        head_parameter_names=model.head_parameter_names(),
    )
    scheduler = build_scheduler(optimizer, SchedulerConfig(type="cosine", warmup_ratio=0.1), STEPS)
    loss_fn = ArticuLMLoss(TINY_LOSS)
    loader = build_dataloader(
        dataset, strategy="fixed_samples", batch_size=len(dataset), shuffle=False
    )
    batch = next(iter(loader))

    history: list[dict[str, float]] = []
    model.train()
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.feature_ids, batch.attention_mask)
        breakdown = loss_fn(output, batch)
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        history.append(
            {
                **breakdown.as_floats(),
                "accuracy": masked_accuracy(
                    output.viseme_logits.detach(), batch.viseme_targets, batch.loss_mask
                ),
                "mae": masked_strength_mae(
                    output.strength_norm.detach(),
                    batch.strength_targets,
                    batch.loss_mask,
                    100.0,
                ),
            }
        )

    return {
        "model": model,
        "config": cfg,
        "data_config": data_cfg,
        "vocab": vocab,
        "dataset": dataset,
        "batch": batch,
        "history": history,
    }


def test_viseme_accuracy_approaches_a_perfect_fit(overfit_run):
    history = overfit_run["history"]
    assert history[0]["accuracy"] < 0.5, "model started already fitted; test is vacuous"
    assert history[-1]["accuracy"] >= 0.98, (
        f"tiny overfit failed to fit viseme labels: "
        f"{history[0]['accuracy']:.3f} -> {history[-1]['accuracy']:.3f}"
    )


def test_viseme_loss_falls_strongly(overfit_run):
    history = overfit_run["history"]
    first, last = history[0]["viseme_loss"], history[-1]["viseme_loss"]
    assert last < first * 0.05, f"viseme loss only fell {first:.4f} -> {last:.4f}"


def test_strength_loss_falls_strongly(overfit_run):
    history = overfit_run["history"]
    first, last = history[0]["strength_loss"], history[-1]["strength_loss"]
    assert last < first * 0.5, f"strength loss only fell {first:.4f} -> {last:.4f}"


def test_strength_mae_falls_to_a_small_value(overfit_run):
    history = overfit_run["history"]
    assert history[-1]["mae"] < 5.0, (
        f"strength MAE did not converge: "
        f"{history[0]['mae']:.2f} -> {history[-1]['mae']:.2f} (0..100 units)"
    )


def test_no_nan_or_inf_anywhere_in_training(overfit_run):
    for step, entry in enumerate(overfit_run["history"]):
        for key, value in entry.items():
            assert math.isfinite(value), f"non-finite {key} at step {step}: {value}"


def test_all_parameters_are_finite_after_training(overfit_run):
    for name, parameter in overfit_run["model"].named_parameters():
        assert torch.isfinite(parameter).all(), name


def test_no_padding_leakage_after_fitting(overfit_run):
    model = overfit_run["model"]
    batch = overfit_run["batch"]
    model.eval()
    with torch.no_grad():
        output = model(batch.feature_ids, batch.attention_mask)
    padded = ~batch.attention_mask
    assert bool(padded.any()), "fixture batch has no padding; test is vacuous"
    assert float(output.hidden_states[padded].abs().max()) == 0.0


def test_padding_does_not_change_fitted_predictions(overfit_run):
    """Re-batch each sentence alone; predictions must not move."""
    model = overfit_run["model"]
    dataset = overfit_run["dataset"]
    batch = overfit_run["batch"]
    model.eval()
    collator = PhonemeCollator()
    with torch.no_grad():
        together = model(batch.feature_ids, batch.attention_mask)
        for row, item in enumerate(dataset.encoded):
            alone = collator([item])
            solo = model(alone.feature_ids, alone.attention_mask)
            length = item.length
            assert torch.allclose(
                solo.viseme_logits[0, :length],
                together.viseme_logits[row, :length],
                atol=1e-4,
            ), item.sample_id


def test_checkpoint_reload_reproduces_fitted_predictions(overfit_run, tmp_path):
    model = overfit_run["model"]
    batch = overfit_run["batch"]
    model.eval()
    with torch.no_grad():
        before = model(batch.feature_ids, batch.attention_mask)

    save_checkpoint(
        tmp_path / "tiny.pt",
        model=model,
        model_config=overfit_run["config"],
        data_config=overfit_run["data_config"],
        vocab=overfit_run["vocab"],
        state=TrainingState(global_step=STEPS),
        seed=1234,
    )
    loaded = load_checkpoint(tmp_path / "tiny.pt")
    reloaded = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    restore_into(loaded, model=reloaded)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(batch.feature_ids, batch.attention_mask)

    assert torch.equal(
        before.viseme_logits.argmax(-1), after.viseme_logits.argmax(-1)
    )
    assert torch.allclose(before.strength_norm, after.strength_norm, atol=1e-6)


def test_fitted_model_predicts_the_training_labels(overfit_run):
    """The end-to-end check: argmax must reproduce the training labels."""
    model = overfit_run["model"]
    batch = overfit_run["batch"]
    model.eval()
    with torch.no_grad():
        ids, strength = model.predict(batch.feature_ids, batch.attention_mask)

    mask = batch.loss_mask
    assert batch.viseme_targets is not None and batch.strength_targets is not None
    matched = (ids[mask] == batch.viseme_targets[mask]).float().mean()
    assert float(matched) >= 0.98

    target_strength = batch.strength_targets[mask] * 100.0
    assert float((strength[mask] - target_strength).abs().mean()) < 5.0
