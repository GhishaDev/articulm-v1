"""Checkpoint save / load / resume.

Every checkpoint carries enough state to reproduce a run byte-for-byte where
the hardware allows (docs/04_training_plan.md):

```text
model / optimizer / scheduler / scaler / global_step / epoch
model_config / data_config / train_config / vocab / seed / rng states
```

Checkpoints are never deleted except by explicit ``keep_last_n`` rotation of
step checkpoints; ``last.pt`` and ``best.pt`` are always preserved.
"""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import (
    DataConfig,
    ModelConfig,
    data_config_from_dict,
    model_config_from_dict,
    to_plain_dict,
)
from ..data.vocab import FeatureVocabulary

CHECKPOINT_FORMAT_VERSION = "articulm_v1_checkpoint_v1"

LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"


class CheckpointError(RuntimeError):
    pass


@dataclass
class TrainingState:
    """Mutable progress counters persisted alongside the weights."""

    global_step: int = 0
    epoch: int = 0
    best_metric: float | None = None
    best_step: int | None = None
    evaluations_without_improvement: int = 0


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        # A device-count mismatch between the saving and resuming host must not
        # abort the resume; losing CUDA RNG alignment is the lesser problem.
        with contextlib.suppress(RuntimeError, ValueError):
            torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_config: ModelConfig,
    data_config: DataConfig,
    vocab: FeatureVocabulary,
    state: TrainingState,
    seed: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    train_config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a full checkpoint atomically (temp file then rename)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "best_step": state.best_step,
        "evaluations_without_improvement": state.evaluations_without_improvement,
        "model_config": to_plain_dict(model_config),
        "data_config": to_plain_dict(data_config),
        "train_config": to_plain_dict(train_config) if train_config is not None else None,
        "vocab": vocab.to_dict(),
        "seed": seed,
        "rng_state": _rng_state(),
        "extra": extra or {},
    }

    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


@dataclass
class LoadedCheckpoint:
    model_config: ModelConfig
    data_config: DataConfig
    vocab: FeatureVocabulary
    state: TrainingState
    seed: int
    train_config: dict[str, Any] | None
    extra: dict[str, Any]
    payload: dict[str, Any]


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> LoadedCheckpoint:
    """Read a checkpoint without touching any live model."""
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source}")

    payload = torch.load(source, map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format {version!r} != expected {CHECKPOINT_FORMAT_VERSION!r}"
        )

    return LoadedCheckpoint(
        model_config=model_config_from_dict(payload["model_config"]),
        data_config=data_config_from_dict(payload["data_config"]),
        vocab=FeatureVocabulary.from_dict(payload["vocab"]),
        state=TrainingState(
            global_step=int(payload.get("global_step", 0)),
            epoch=int(payload.get("epoch", 0)),
            best_metric=payload.get("best_metric"),
            best_step=payload.get("best_step"),
            evaluations_without_improvement=int(
                payload.get("evaluations_without_improvement", 0)
            ),
        ),
        seed=int(payload.get("seed", 0)),
        train_config=payload.get("train_config"),
        extra=payload.get("extra") or {},
        payload=payload,
    )


def restore_into(
    loaded: LoadedCheckpoint,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    strict: bool = True,
    restore_rng: bool = True,
) -> None:
    """Load weights and optimizer/scheduler/scaler state into live objects."""
    missing, unexpected = model.load_state_dict(
        loaded.payload["model_state_dict"], strict=strict
    )
    if strict and (missing or unexpected):
        raise CheckpointError(
            f"state dict mismatch: missing={list(missing)} unexpected={list(unexpected)}"
        )

    if optimizer is not None and loaded.payload.get("optimizer_state_dict"):
        optimizer.load_state_dict(loaded.payload["optimizer_state_dict"])
    if scheduler is not None and loaded.payload.get("scheduler_state_dict"):
        scheduler.load_state_dict(loaded.payload["scheduler_state_dict"])
    if scaler is not None and loaded.payload.get("scaler_state_dict"):
        scaler.load_state_dict(loaded.payload["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(loaded.payload.get("rng_state"))


def rotate_step_checkpoints(directory: str | Path, keep_last_n: int) -> list[Path]:
    """Delete the oldest ``step_*.pt`` files beyond ``keep_last_n``.

    Only rotates step checkpoints. ``last.pt`` and ``best.pt`` are never
    touched, and datasets are never touched.
    """
    if keep_last_n < 1:
        raise ValueError("keep_last_n must be >= 1")
    folder = Path(directory)
    if not folder.is_dir():
        return []

    def step_of(candidate: Path) -> int:
        try:
            return int(candidate.stem.split("_")[-1])
        except ValueError:
            return -1

    step_files = sorted(
        (p for p in folder.glob("step_*.pt") if step_of(p) >= 0), key=step_of
    )
    removed: list[Path] = []
    while len(step_files) > keep_last_n:
        oldest = step_files.pop(0)
        oldest.unlink()
        removed.append(oldest)
    return removed


def is_better(candidate: float, incumbent: float | None, *, higher_is_better: bool) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent if higher_is_better else candidate < incumbent
