"""Checkpoint completeness, reload parity, resume and rotation."""

from __future__ import annotations

import pytest
import torch

from articulm.config import SchedulerConfig
from articulm.data.collator import PhonemeCollator
from articulm.model.articulm_v1 import ArticuLMV1
from articulm.training.checkpoint import (
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_FORMAT_VERSION,
    LAST_CHECKPOINT_NAME,
    CheckpointError,
    TrainingState,
    is_better,
    load_checkpoint,
    restore_into,
    rotate_step_checkpoints,
    save_checkpoint,
)
from articulm.training.optimizer import build_optimizer
from articulm.training.scheduler import build_scheduler


@pytest.fixture
def batch(dataset):
    return PhonemeCollator()(list(dataset.encoded))


@pytest.fixture
def trained_bundle(tiny_model_config, data_config, vocab, batch):
    """A model that has taken a few real steps, plus its optimizer state."""
    from articulm.config import LossConfig, OptimizerConfig
    from articulm.training.losses import ArticuLMLoss

    torch.manual_seed(11)
    model = ArticuLMV1.from_vocabulary(tiny_model_config, vocab)
    optimizer = build_optimizer(
        model,
        OptimizerConfig(learning_rate=1e-3),
        head_parameter_names=model.head_parameter_names(),
    )
    scheduler = build_scheduler(optimizer, SchedulerConfig(warmup_ratio=0.1), 20)
    loss_fn = ArticuLMLoss(LossConfig())

    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(batch.feature_ids, batch.attention_mask), batch).total.backward()
        optimizer.step()
        scheduler.step()

    return model, optimizer, scheduler


def test_checkpoint_contains_everything_the_plan_requires(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab
):
    """docs/04: model, optimizer, scheduler, scaler, step/epoch, configs,
    vocab, seed."""
    model, optimizer, scheduler = trained_bundle
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    state = TrainingState(global_step=3, epoch=1, best_metric=0.42, best_step=3)

    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=state,
        seed=1234,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_config=None,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in (
        "format_version",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "global_step",
        "epoch",
        "model_config",
        "data_config",
        "vocab",
        "seed",
        "rng_state",
    ):
        assert key in payload, key
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["global_step"] == 3
    assert payload["epoch"] == 1
    assert payload["seed"] == 1234


def test_load_restores_configs_and_vocab(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab
):
    model, optimizer, scheduler = trained_bundle
    save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=3),
        seed=7,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    assert loaded.model_config == tiny_model_config
    assert loaded.data_config == data_config
    assert loaded.vocab.sizes() == vocab.sizes()
    assert loaded.state.global_step == 3
    assert loaded.seed == 7


def test_reload_reproduces_predictions_exactly(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab, batch
):
    """M2 acceptance criterion: save/load parity."""
    model, optimizer, scheduler = trained_bundle
    model.eval()
    with torch.no_grad():
        before = model(batch.feature_ids, batch.attention_mask)

    save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=3),
        seed=7,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    fresh = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    restore_into(loaded, model=fresh)
    fresh.eval()
    with torch.no_grad():
        after = fresh(batch.feature_ids, batch.attention_mask)

    assert torch.equal(before.viseme_logits, after.viseme_logits)
    assert torch.equal(before.strength_norm, after.strength_norm)
    assert torch.equal(
        before.viseme_logits.argmax(-1), after.viseme_logits.argmax(-1)
    )


def test_resume_restores_optimizer_and_scheduler_state(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab
):
    model, optimizer, scheduler = trained_bundle
    save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=3, epoch=1),
        seed=7,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    expected_lr = optimizer.param_groups[0]["lr"]
    expected_last_epoch = scheduler.last_epoch

    from articulm.config import OptimizerConfig

    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    fresh = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    fresh_optimizer = build_optimizer(
        fresh,
        OptimizerConfig(learning_rate=1e-3),
        head_parameter_names=fresh.head_parameter_names(),
    )
    fresh_scheduler = build_scheduler(
        fresh_optimizer, SchedulerConfig(warmup_ratio=0.1), 20
    )
    restore_into(
        loaded, model=fresh, optimizer=fresh_optimizer, scheduler=fresh_scheduler
    )

    assert fresh_scheduler.last_epoch == expected_last_epoch
    assert fresh_optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    # Adam moment buffers must survive, otherwise resume restarts momentum.
    assert fresh_optimizer.state_dict()["state"]
    assert loaded.state.global_step == 3
    assert loaded.state.epoch == 1


def test_resumed_model_continues_rather_than_restarting(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab, batch
):
    """Same data + resumed state must give the same next step as never saving."""
    from articulm.config import LossConfig, OptimizerConfig
    from articulm.training.losses import ArticuLMLoss

    model, optimizer, scheduler = trained_bundle
    loss_fn = ArticuLMLoss(LossConfig())
    save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(global_step=3),
        seed=7,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Continue in-place.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(batch.feature_ids, batch.attention_mask), batch).total.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        continued = model(batch.feature_ids, batch.attention_mask).viseme_logits

    # Continue from the checkpoint.
    loaded = load_checkpoint(tmp_path / "ckpt.pt")
    fresh = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    fresh_optimizer = build_optimizer(
        fresh,
        OptimizerConfig(learning_rate=1e-3),
        head_parameter_names=fresh.head_parameter_names(),
    )
    fresh_scheduler = build_scheduler(
        fresh_optimizer, SchedulerConfig(warmup_ratio=0.1), 20
    )
    restore_into(
        loaded, model=fresh, optimizer=fresh_optimizer, scheduler=fresh_scheduler
    )
    fresh.train()
    fresh_optimizer.zero_grad(set_to_none=True)
    loss_fn(fresh(batch.feature_ids, batch.attention_mask), batch).total.backward()
    fresh_optimizer.step()
    fresh.eval()
    with torch.no_grad():
        resumed = fresh(batch.feature_ids, batch.attention_mask).viseme_logits

    assert torch.allclose(continued, resumed, atol=1e-5)


def test_missing_checkpoint_is_rejected(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_wrong_format_version_is_rejected(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab
):
    model, optimizer, _ = trained_bundle
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(),
        seed=1,
        optimizer=optimizer,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = "articulm_v0"
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="format"):
        load_checkpoint(path)


def test_save_is_atomic_and_leaves_no_temp_file(
    tmp_path, trained_bundle, tiny_model_config, data_config, vocab
):
    model, _, _ = trained_bundle
    save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        model_config=tiny_model_config,
        data_config=data_config,
        vocab=vocab,
        state=TrainingState(),
        seed=1,
    )
    assert (tmp_path / "ckpt.pt").is_file()
    assert not list(tmp_path.glob("*.tmp"))


# -- rotation --------------------------------------------------------------


def test_rotation_keeps_the_newest_step_checkpoints(tmp_path):
    for step in (10, 20, 30, 40):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    removed = rotate_step_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_00000030.pt", "step_00000040.pt"]
    assert len(removed) == 2


def test_rotation_never_deletes_last_or_best(tmp_path):
    (tmp_path / LAST_CHECKPOINT_NAME).write_bytes(b"x")
    (tmp_path / BEST_CHECKPOINT_NAME).write_bytes(b"x")
    for step in (10, 20, 30):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    rotate_step_checkpoints(tmp_path, keep_last_n=1)
    assert (tmp_path / LAST_CHECKPOINT_NAME).is_file()
    assert (tmp_path / BEST_CHECKPOINT_NAME).is_file()
    assert sorted(p.name for p in tmp_path.glob("step_*.pt")) == ["step_00000030.pt"]


def test_rotation_rejects_zero_keep(tmp_path):
    with pytest.raises(ValueError, match="keep_last_n"):
        rotate_step_checkpoints(tmp_path, keep_last_n=0)


def test_rotation_on_missing_directory_is_a_noop(tmp_path):
    assert rotate_step_checkpoints(tmp_path / "absent", keep_last_n=3) == []


# -- best-metric comparison ------------------------------------------------


@pytest.mark.parametrize(
    "candidate,incumbent,higher,expected",
    [
        (0.5, None, True, True),
        (0.6, 0.5, True, True),
        (0.4, 0.5, True, False),
        (1.0, 2.0, False, True),
        (3.0, 2.0, False, False),
    ],
)
def test_is_better(candidate, incumbent, higher, expected):
    assert is_better(candidate, incumbent, higher_is_better=higher) is expected
