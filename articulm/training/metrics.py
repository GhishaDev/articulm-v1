"""Viseme and Strength metrics with slice breakdowns.

All metrics are computed over unpadded, supervised tokens only. Accumulators
take already-masked flat arrays, so a padding leak cannot reach them.

Viseme  : accuracy, macro F1, weighted F1, per-class precision/recall/F1,
          confusion matrix
Strength: MAE, RMSE, median absolute error, per-viseme MAE (in 0..100 units)
Slices  : language, surface_tone, stress, syllable_role, phrase position,
          sequence-length bucket
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..data.collator import SLICE_FIELDS, Batch
from ..model.articulm_v1 import ArticuLMOutput


@dataclass
class VisemeMetrics:
    """Confusion-matrix based classification metrics."""

    num_classes: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_precision: list[float]
    per_class_recall: list[float]
    per_class_f1: list[float]
    support: list[int]
    confusion_matrix: list[list[int]]

    def as_dict(self, *, include_matrix: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
        }
        if include_matrix:
            out["confusion_matrix"] = self.confusion_matrix
            out["per_class"] = [
                {
                    "viseme_id": index,
                    "precision": self.per_class_precision[index],
                    "recall": self.per_class_recall[index],
                    "f1": self.per_class_f1[index],
                    "support": self.support[index],
                }
                for index in range(self.num_classes)
            ]
        return out


@dataclass
class StrengthMetrics:
    """Regression metrics reported in user-facing 0..100 units."""

    mae: float
    rmse: float
    median_absolute_error: float
    count: int
    per_viseme_mae: dict[int, float] = field(default_factory=dict)

    def as_dict(self, *, include_per_viseme: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "mae": self.mae,
            "rmse": self.rmse,
            "median_absolute_error": self.median_absolute_error,
            "count": self.count,
        }
        if include_per_viseme:
            out["per_viseme_mae"] = {str(k): v for k, v in sorted(self.per_viseme_mae.items())}
        return out


class _ClassificationAccumulator:
    """Accumulates a confusion matrix incrementally."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        if predictions.shape != targets.shape:
            raise ValueError("prediction/target shape mismatch")
        if predictions.size == 0:
            return
        flat = targets.astype(np.int64) * self.num_classes + predictions.astype(np.int64)
        counts = np.bincount(flat, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    @property
    def count(self) -> int:
        return int(self.matrix.sum())

    def compute(self) -> VisemeMetrics:
        matrix = self.matrix
        total = matrix.sum()
        true_positive = np.diag(matrix).astype(np.float64)
        predicted = matrix.sum(axis=0).astype(np.float64)
        actual = matrix.sum(axis=1).astype(np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(predicted > 0, true_positive / predicted, 0.0)
            recall = np.where(actual > 0, true_positive / actual, 0.0)
            denominator = precision + recall
            f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

        # Macro averages count only classes that actually occur in the data,
        # so absent classes do not silently drag the average toward zero.
        present = actual > 0
        num_present = int(present.sum())
        macro_precision = float(precision[present].mean()) if num_present else 0.0
        macro_recall = float(recall[present].mean()) if num_present else 0.0
        macro_f1 = float(f1[present].mean()) if num_present else 0.0
        weighted_f1 = float((f1 * actual).sum() / total) if total else 0.0
        accuracy = float(true_positive.sum() / total) if total else 0.0

        return VisemeMetrics(
            num_classes=self.num_classes,
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            per_class_precision=[float(v) for v in precision],
            per_class_recall=[float(v) for v in recall],
            per_class_f1=[float(v) for v in f1],
            support=[int(v) for v in actual],
            confusion_matrix=matrix.tolist(),
        )


class _RegressionAccumulator:
    """Accumulates strength errors in 0..100 units."""

    def __init__(self, num_classes: int, *, keep_errors: bool = True) -> None:
        self.num_classes = num_classes
        self.keep_errors = keep_errors
        self.sum_absolute = 0.0
        self.sum_squared = 0.0
        self.count = 0
        self.per_class_absolute = np.zeros(num_classes, dtype=np.float64)
        self.per_class_count = np.zeros(num_classes, dtype=np.int64)
        self._errors: list[np.ndarray] = []

    def update(
        self, predictions: np.ndarray, targets: np.ndarray, target_visemes: np.ndarray
    ) -> None:
        if predictions.size == 0:
            return
        absolute = np.abs(predictions - targets)
        self.sum_absolute += float(absolute.sum())
        self.sum_squared += float(np.square(predictions - targets).sum())
        self.count += int(absolute.size)
        if self.keep_errors:
            self._errors.append(absolute.astype(np.float32))

        classes = target_visemes.astype(np.int64)
        np.add.at(self.per_class_absolute, classes, absolute)
        np.add.at(self.per_class_count, classes, 1)

    def compute(self) -> StrengthMetrics:
        if self.count == 0:
            return StrengthMetrics(mae=0.0, rmse=0.0, median_absolute_error=0.0, count=0)
        mae = self.sum_absolute / self.count
        rmse = float(np.sqrt(self.sum_squared / self.count))
        if self.keep_errors and self._errors:
            median = float(np.median(np.concatenate(self._errors)))
        else:
            median = float("nan")
        per_viseme = {
            index: float(self.per_class_absolute[index] / self.per_class_count[index])
            for index in range(self.num_classes)
            if self.per_class_count[index] > 0
        }
        return StrengthMetrics(
            mae=mae,
            rmse=rmse,
            median_absolute_error=median,
            count=self.count,
            per_viseme_mae=per_viseme,
        )


@dataclass
class MetricsReport:
    """Overall metrics plus per-slice breakdowns."""

    viseme: VisemeMetrics
    strength: StrengthMetrics
    num_tokens: int
    num_sequences: int
    slices: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)

    def as_dict(self, *, include_details: bool = True) -> dict[str, object]:
        return {
            "num_tokens": self.num_tokens,
            "num_sequences": self.num_sequences,
            "viseme": self.viseme.as_dict(include_matrix=include_details),
            "strength": self.strength.as_dict(include_per_viseme=include_details),
            "slices": self.slices if include_details else {},
        }

    def scalar_summary(self, prefix: str = "") -> dict[str, float]:
        """Flat scalars suitable for logging and checkpoint selection."""
        return {
            f"{prefix}viseme_accuracy": self.viseme.accuracy,
            f"{prefix}viseme_macro_f1": self.viseme.macro_f1,
            f"{prefix}viseme_weighted_f1": self.viseme.weighted_f1,
            f"{prefix}strength_mae": self.strength.mae,
            f"{prefix}strength_rmse": self.strength.rmse,
        }


class MetricsAccumulator:
    """Streaming metric accumulation over batches."""

    def __init__(
        self,
        *,
        num_classes: int = 18,
        strength_scale: float = 100.0,
        slice_fields: tuple[str, ...] = SLICE_FIELDS,
        keep_errors: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.strength_scale = strength_scale
        self.slice_fields = slice_fields
        self.keep_errors = keep_errors

        self._viseme = _ClassificationAccumulator(num_classes)
        self._strength = _RegressionAccumulator(num_classes, keep_errors=keep_errors)
        self._slice_viseme: dict[str, dict[str, _ClassificationAccumulator]] = {
            name: {} for name in slice_fields
        }
        self._slice_strength: dict[str, dict[str, _RegressionAccumulator]] = {
            name: {} for name in slice_fields
        }
        self.num_sequences = 0

    # ------------------------------------------------------------- updating

    def update_from_batch(self, output: ArticuLMOutput, batch: Batch) -> None:
        """Accumulate one batch, using ``loss_mask`` to drop padding."""
        if batch.viseme_targets is None or batch.strength_targets is None:
            raise ValueError("metrics require a labelled batch")

        mask = batch.loss_mask.bool()
        predicted_visemes = output.viseme_logits.argmax(dim=-1)[mask]
        target_visemes = batch.viseme_targets[mask]
        predicted_strength = output.strength_norm[mask] * self.strength_scale
        target_strength = batch.strength_targets[mask] * self.strength_scale

        self.update(
            predicted_visemes.detach().to("cpu").numpy(),
            target_visemes.detach().to("cpu").numpy(),
            predicted_strength.detach().float().to("cpu").numpy(),
            target_strength.detach().float().to("cpu").numpy(),
            slices=batch.slices,
            num_sequences=batch.batch_size,
        )

    def update(
        self,
        predicted_visemes: np.ndarray,
        target_visemes: np.ndarray,
        predicted_strength: np.ndarray,
        target_strength: np.ndarray,
        *,
        slices: dict[str, tuple[str, ...]] | None = None,
        num_sequences: int = 0,
    ) -> None:
        self._viseme.update(predicted_visemes, target_visemes)
        self._strength.update(predicted_strength, target_strength, target_visemes)
        self.num_sequences += num_sequences

        if not slices:
            return
        num_tokens = predicted_visemes.shape[0]
        for name in self.slice_fields:
            values = slices.get(name)
            if values is None:
                continue
            if len(values) != num_tokens:
                raise ValueError(
                    f"slice '{name}' has {len(values)} entries but the mask selected "
                    f"{num_tokens} tokens; slice metadata is misaligned with the batch"
                )
            value_array = np.asarray(values, dtype=object)
            for value in np.unique(value_array):
                selection = value_array == value
                key = str(value)
                viseme_acc = self._slice_viseme[name].setdefault(
                    key, _ClassificationAccumulator(self.num_classes)
                )
                strength_acc = self._slice_strength[name].setdefault(
                    key, _RegressionAccumulator(self.num_classes, keep_errors=False)
                )
                viseme_acc.update(predicted_visemes[selection], target_visemes[selection])
                strength_acc.update(
                    predicted_strength[selection],
                    target_strength[selection],
                    target_visemes[selection],
                )

    # ------------------------------------------------------------ computing

    def compute(self) -> MetricsReport:
        slice_report: dict[str, dict[str, dict[str, object]]] = {}
        for name in self.slice_fields:
            per_value: dict[str, dict[str, object]] = {}
            for key, viseme_acc in sorted(self._slice_viseme[name].items()):
                viseme = viseme_acc.compute()
                strength = self._slice_strength[name][key].compute()
                per_value[key] = {
                    "num_tokens": viseme_acc.count,
                    "viseme_accuracy": viseme.accuracy,
                    "viseme_macro_f1": viseme.macro_f1,
                    "strength_mae": strength.mae,
                    "strength_rmse": strength.rmse,
                }
            if per_value:
                slice_report[name] = per_value

        return MetricsReport(
            viseme=self._viseme.compute(),
            strength=self._strength.compute(),
            num_tokens=self._viseme.count,
            num_sequences=self.num_sequences,
            slices=slice_report,
        )


def masked_accuracy(
    viseme_logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> float:
    """Quick accuracy for training-loop logging."""
    selected = mask.bool()
    if not selected.any():
        return 0.0
    predictions = viseme_logits.argmax(dim=-1)[selected]
    return float((predictions == targets[selected]).float().mean())


def masked_strength_mae(
    strength_norm: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 100.0,
) -> float:
    """Quick MAE in 0..100 units for training-loop logging."""
    selected = mask.bool()
    if not selected.any():
        return 0.0
    error = (strength_norm[selected] - targets[selected]).abs().mean()
    return float(error) * scale
