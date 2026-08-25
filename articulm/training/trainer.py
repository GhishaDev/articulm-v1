"""Lightweight PyTorch trainer for ArticuLM-V1.

Supports AdamW, warmup + decay scheduling, gradient clipping, mixed precision
(fp16 with GradScaler / bf16 without), gradient accumulation, periodic
validation, checkpoint save/rotate and resume.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import TrainRunConfig
from ..data.collator import Batch
from ..data.vocab import FeatureVocabulary
from ..model.articulm_v1 import ArticuLMV1
from ..runtime import (
    PrecisionPlan,
    StructuredLogger,
    describe_hardware,
    resolve_device,
    resolve_precision,
)
from .checkpoint import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    CheckpointError,
    LoadedCheckpoint,
    TrainingState,
    is_better,
    restore_into,
    rotate_step_checkpoints,
    save_checkpoint,
)
from .losses import ArticuLMLoss
from .metrics import MetricsAccumulator, MetricsReport, masked_accuracy, masked_strength_mae
from .optimizer import build_optimizer, parameter_group_summary
from .scheduler import build_scheduler, current_learning_rate


@dataclass
class TrainingSummary:
    """What the CLI reports when a run finishes."""

    run_id: str
    global_step: int
    epochs_completed: int
    final_train_loss: float
    final_train_viseme_accuracy: float
    final_train_strength_mae: float
    best_metric: float | None
    best_step: int | None
    validation: dict[str, float] | None
    checkpoint_dir: str
    stopped_because: str
    saw_non_finite_loss: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "global_step": self.global_step,
            "epochs_completed": self.epochs_completed,
            "final_train_loss": self.final_train_loss,
            "final_train_viseme_accuracy": self.final_train_viseme_accuracy,
            "final_train_strength_mae": self.final_train_strength_mae,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "validation": self.validation,
            "checkpoint_dir": self.checkpoint_dir,
            "stopped_because": self.stopped_because,
            "saw_non_finite_loss": self.saw_non_finite_loss,
        }


class Trainer:
    """Owns the training loop for one run."""

    def __init__(
        self,
        *,
        config: TrainRunConfig,
        model: ArticuLMV1,
        vocab: FeatureVocabulary,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        run_dir: str | Path,
        logger: StructuredLogger,
        device: torch.device | None = None,
        precision: PrecisionPlan | None = None,
    ) -> None:
        self.config = config
        self.training = config.training
        self.vocab = vocab
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.logger = logger

        self.device = device or resolve_device()
        self.hardware = describe_hardware(self.device)
        self.precision = precision or resolve_precision(self.training.precision, self.hardware)

        self.model = model.to(self.device)
        self.loss_fn = ArticuLMLoss(
            self.training.loss, num_classes=config.model.viseme_head.num_classes
        )
        self.optimizer = build_optimizer(
            self.model,
            self.training.optimizer,
            head_parameter_names=self.model.head_parameter_names(),
        )
        self.total_optimizer_steps = self._plan_total_steps()
        self.scheduler = build_scheduler(
            self.optimizer, self.training.scheduler, self.total_optimizer_steps
        )
        self.scaler = torch.amp.GradScaler(
            self.device.type, enabled=self.precision.use_grad_scaler
        )
        self.state = TrainingState()
        self.strength_scale = config.model.strength_head.output_scale
        self._saw_non_finite_loss = False
        # Last completed logging window, so the run summary is never NaN just
        # because the loop broke immediately after a log flush.
        self._last_window_metrics: dict[str, float] = {
            "loss": float("nan"),
            "accuracy": 0.0,
            "mae": float("nan"),
        }

        self.logger.event(
            "trainer_init",
            "trainer ready",
            device=str(self.device),
            device_name=self.hardware.device_name,
            precision=self.precision.name,
            precision_reason=self.precision.reason,
            total_optimizer_steps=self.total_optimizer_steps,
            accumulation_steps=self.training.batching.gradient_accumulation_steps,
            parameters=self.model.num_parameters(),
            parameter_groups=parameter_group_summary(self.optimizer),
        )

    # ------------------------------------------------------------- planning

    def _plan_total_steps(self) -> int:
        """Total optimizer steps, from max_steps and/or max_epochs."""
        accumulation = self.training.batching.gradient_accumulation_steps
        try:
            batches_per_epoch = len(self.train_loader)
        except TypeError:  # pragma: no cover - iterable-only loaders
            batches_per_epoch = 0

        candidates: list[int] = []
        if self.training.max_steps is not None:
            candidates.append(self.training.max_steps)
        if self.training.max_epochs is not None and batches_per_epoch:
            candidates.append(
                self.training.max_epochs
                * max(1, math.ceil(batches_per_epoch / accumulation))
            )
        if not candidates:
            raise ValueError("cannot plan a run without max_steps or max_epochs")
        return max(1, min(candidates))

    def _epoch_limit(self) -> int:
        if self.training.max_epochs is not None:
            return self.training.max_epochs
        return 1 << 30

    # ------------------------------------------------------------- resuming

    def resume(self, loaded: LoadedCheckpoint) -> None:
        """Restore model/optimizer/scheduler/scaler/step state in place."""
        restore_into(
            loaded,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.state = loaded.state
        self.logger.event(
            "resume",
            "resumed from checkpoint",
            global_step=self.state.global_step,
            epoch=self.state.epoch,
            best_metric=self.state.best_metric,
        )

    def warm_start(self, loaded: LoadedCheckpoint) -> None:
        """Load model weights only; optimizer/scheduler/scaler/step stay fresh.

        The fine-tuning entry point, unlike ``resume`` which restores the full
        training state for crash recovery. The checkpoint vocabulary must match
        this run's vocabulary exactly, otherwise the loaded embeddings would be
        silently misaligned.
        """
        if loaded.vocab.to_dict() != self.vocab.to_dict():
            raise CheckpointError(
                "init_from checkpoint vocabulary differs from this run's "
                "vocabulary; embeddings would be misaligned. Re-run with "
                "--vocab pointing at the frozen vocabulary the checkpoint "
                "was trained with."
            )
        restore_into(
            loaded,
            model=self.model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            restore_rng=False,
        )
        self.logger.event(
            "init_from",
            "warm start from checkpoint (weights only, fresh optimizer)",
            source_global_step=loaded.state.global_step,
            source_best_metric=loaded.state.best_metric,
            source_training_stage=(loaded.train_config or {}).get("stage"),
        )

    # ------------------------------------------------------------- training

    def train(self) -> TrainingSummary:
        self.model.train()
        stop_reason = "max_steps_reached"
        last_metrics = {"loss": float("nan"), "accuracy": 0.0, "mae": float("nan")}
        validation_summary: dict[str, float] | None = None
        epoch = self.state.epoch

        while self.state.global_step < self.total_optimizer_steps and epoch < self._epoch_limit():
            self._set_epoch(epoch)
            epoch_result = self._run_epoch(epoch)
            last_metrics = epoch_result["metrics"]
            if epoch_result["validation"] is not None:
                validation_summary = epoch_result["validation"]
            epoch += 1
            self.state.epoch = epoch
            if epoch_result["stop_reason"]:
                stop_reason = epoch_result["stop_reason"]
                break
        else:
            stop_reason = (
                "max_epochs_reached"
                if epoch >= self._epoch_limit()
                else "max_steps_reached"
            )

        # Final validation + checkpoint so a run always ends with usable state.
        if self.validation_loader is not None:
            report = self.evaluate(self.validation_loader)
            validation_summary = self._with_composite_metric(report.scalar_summary("val_"))
            self._track_best(validation_summary)
            self.logger.event("validation_final", "final validation", **validation_summary)

        self._save(LAST_CHECKPOINT_NAME)

        return TrainingSummary(
            run_id=self.logger.run_id,
            global_step=self.state.global_step,
            epochs_completed=self.state.epoch,
            final_train_loss=last_metrics["loss"],
            final_train_viseme_accuracy=last_metrics["accuracy"],
            final_train_strength_mae=last_metrics["mae"],
            best_metric=self.state.best_metric,
            best_step=self.state.best_step,
            validation=validation_summary,
            checkpoint_dir=str(self.checkpoint_dir),
            stopped_because=stop_reason,
            saw_non_finite_loss=self._saw_non_finite_loss,
        )

    def _set_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _run_epoch(self, epoch: int) -> dict[str, Any]:
        accumulation = self.training.batching.gradient_accumulation_steps
        clip_norm = self.training.gradient_clip_norm
        log_every = self.training.logging.every_steps
        eval_every = self.training.evaluation.every_steps
        save_every = self.training.checkpoint.every_steps

        running_loss = 0.0
        running_viseme = 0.0
        running_strength = 0.0
        running_accuracy = 0.0
        running_mae = 0.0
        running_tokens = 0
        micro_batches = 0
        stop_reason = ""
        validation_summary: dict[str, float] | None = None
        window_start = time.monotonic()

        self.optimizer.zero_grad(set_to_none=True)

        for micro_index, batch in enumerate(self._iterate(self.train_loader)):
            batch = batch.to(self.device)
            loss_breakdown, accuracy, mae = self._forward_backward(batch, accumulation)

            running_loss += loss_breakdown["loss"]
            running_viseme += loss_breakdown["viseme_loss"]
            running_strength += loss_breakdown["strength_loss"]
            running_accuracy += accuracy
            running_mae += mae
            running_tokens += batch.num_supervised_tokens
            micro_batches += 1

            if (micro_index + 1) % accumulation != 0:
                continue

            grad_norm = self._optimizer_step(clip_norm)
            self.state.global_step += 1
            step = self.state.global_step

            if step % log_every == 0:
                elapsed = max(time.monotonic() - window_start, 1e-9)
                self._last_window_metrics = {
                    "loss": running_loss / micro_batches,
                    "accuracy": running_accuracy / micro_batches,
                    "mae": running_mae / micro_batches,
                }
                self.logger.event(
                    "train_step",
                    "training step",
                    epoch=epoch,
                    step=step,
                    loss=running_loss / micro_batches,
                    viseme_loss=running_viseme / micro_batches,
                    strength_loss=running_strength / micro_batches,
                    viseme_accuracy=running_accuracy / micro_batches,
                    strength_mae=running_mae / micro_batches,
                    lr=current_learning_rate(self.optimizer),
                    grad_norm=grad_norm,
                    tokens_per_s=running_tokens / elapsed,
                    supervised_tokens=running_tokens,
                    precision=self.precision.name,
                )
                running_loss = running_viseme = running_strength = 0.0
                running_accuracy = running_mae = 0.0
                running_tokens = 0
                micro_batches = 0
                window_start = time.monotonic()

            if self.validation_loader is not None and step % eval_every == 0:
                report = self.evaluate(self.validation_loader)
                validation_summary = self._with_composite_metric(report.scalar_summary("val_"))
                self.logger.event(
                    "validation", "validation", step=step, **validation_summary
                )
                improved = self._track_best(validation_summary)
                self.model.train()
                if not improved and self._should_early_stop():
                    stop_reason = "early_stopping"

            if step % save_every == 0:
                self._save(f"step_{step:08d}.pt")
                self._save(LAST_CHECKPOINT_NAME)
                rotate_step_checkpoints(
                    self.checkpoint_dir, self.training.checkpoint.keep_last_n
                )

            if stop_reason or step >= self.total_optimizer_steps:
                if not stop_reason:
                    stop_reason = "max_steps_reached"
                break

        if micro_batches:
            metrics = {
                "loss": running_loss / micro_batches,
                "accuracy": running_accuracy / micro_batches,
                "mae": running_mae / micro_batches,
            }
        else:
            # The window was flushed by the last log; reuse those values.
            metrics = dict(self._last_window_metrics)

        return {
            "metrics": metrics,
            "validation": validation_summary,
            "stop_reason": stop_reason,
        }

    def _iterate(self, loader: DataLoader) -> Iterator[Batch]:
        yield from loader

    def _forward_backward(
        self, batch: Batch, accumulation: int
    ) -> tuple[dict[str, float], float, float]:
        autocast_context = torch.autocast(
            device_type=self.device.type,
            dtype=self.precision.autocast_dtype or torch.float32,
            enabled=self.precision.autocast_enabled,
        )
        with autocast_context:
            output = self.model(batch.feature_ids, batch.attention_mask)
            breakdown = self.loss_fn(output, batch)

        loss = breakdown.total
        if not torch.isfinite(loss):
            self._saw_non_finite_loss = True
            self.logger.warning(
                f"non-finite loss at step {self.state.global_step}; skipping this micro-batch"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return breakdown.as_floats(), 0.0, float("nan")

        scaled = loss / accumulation
        if self.scaler.is_enabled():
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        with torch.no_grad():
            accuracy = masked_accuracy(
                output.viseme_logits.detach(), batch.viseme_targets, batch.loss_mask
            )
            mae = masked_strength_mae(
                output.strength_norm.detach(),
                batch.strength_targets,
                batch.loss_mask,
                self.strength_scale,
            )
        return breakdown.as_floats(), accuracy, mae

    def _optimizer_step(self, clip_norm: float) -> float:
        grad_norm = float("nan")
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        if clip_norm > 0:
            grad_norm = float(
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
            )
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    # ----------------------------------------------------------- evaluating

    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, *, keep_errors: bool = True
    ) -> MetricsReport:
        """Run metrics over a loader. Padding is excluded by ``loss_mask``."""
        was_training = self.model.training
        self.model.eval()
        accumulator = MetricsAccumulator(
            num_classes=self.config.model.viseme_head.num_classes,
            strength_scale=self.strength_scale,
            keep_errors=keep_errors,
        )
        try:
            for batch in loader:
                batch = batch.to(self.device)
                # Evaluation always runs in fp32 so metrics are not distorted
                # by autocast rounding.
                output = self.model(batch.feature_ids, batch.attention_mask)
                accumulator.update_from_batch(output, batch)
        finally:
            if was_training:
                self.model.train()
        return accumulator.compute()

    # ---------------------------------------------------------- bookkeeping

    def _with_composite_metric(self, summary: dict[str, float]) -> dict[str, float]:
        """Add ``val_composite`` = macro F1 - alpha x MAE/scale to a validation
        summary so checkpoint selection can use a metric that still separates
        candidates once macro F1 sits at its noise ceiling."""
        f1_key = "val_viseme_macro_f1"
        mae_key = "val_strength_mae"
        if f1_key in summary and mae_key in summary:
            alpha = self.training.checkpoint.best_composite_alpha
            summary = dict(summary)
            summary["val_composite"] = (
                summary[f1_key] - alpha * summary[mae_key] / self.strength_scale
            )
        return summary

    def _track_best(self, validation: dict[str, float]) -> bool:
        key = self.training.checkpoint.save_best_by
        if key not in validation:
            self.logger.warning(
                f"checkpoint.save_best_by={key!r} is not in the validation metrics "
                f"{sorted(validation)}; best checkpoint not updated"
            )
            return False
        candidate = validation[key]
        higher_is_better = self.training.checkpoint.higher_is_better
        if "mae" in key or "rmse" in key or key.endswith("_loss"):
            higher_is_better = False
        if is_better(candidate, self.state.best_metric, higher_is_better=higher_is_better):
            self.state.best_metric = candidate
            self.state.best_step = self.state.global_step
            self.state.evaluations_without_improvement = 0
            self._save(BEST_CHECKPOINT_NAME)
            self.logger.event(
                "checkpoint_best",
                "new best checkpoint",
                metric=key,
                value=candidate,
                step=self.state.global_step,
            )
            return True
        self.state.evaluations_without_improvement += 1
        return False

    def _should_early_stop(self) -> bool:
        early = self.training.early_stopping
        if not early.enabled:
            return False
        return self.state.evaluations_without_improvement >= early.patience_evaluations

    def _save(self, filename: str) -> Path:
        path = save_checkpoint(
            self.checkpoint_dir / filename,
            model=self.model,
            model_config=self.config.model,
            data_config=self.config.data,
            vocab=self.vocab,
            state=self.state,
            seed=self.config.experiment.seed,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.scaler.is_enabled() else None,
            train_config=self.config.training,
            extra={
                "experiment_name": self.config.experiment.name,
                "run_id": self.logger.run_id,
                "precision": self.precision.name,
                "device": str(self.device),
                "parameters": self.model.num_parameters(),
            },
        )
        self.logger.event(
            "checkpoint_saved", "checkpoint saved", path=str(path), step=self.state.global_step
        )
        return path
