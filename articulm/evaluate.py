"""Evaluation CLI.

```bash
python -m articulm.evaluate --checkpoint runs/.../checkpoints/best.pt \
    --data data/test.jsonl --label-set synthetic --out-dir reports/e2_test
```

Artifacts written to ``--out-dir``:

```text
metrics.json            overall + slice metrics
per_class.csv           per-viseme precision / recall / F1 / support / MAE
confusion_matrix.csv    16x16 counts
strength_report.csv     per-viseme strength error summary
failure_cases.jsonl     worst mispredictions for inspection
```

``--label-set`` is recorded verbatim in ``metrics.json``. Synthetic and Human
Gold results are separate invocations and must be reported separately
(docs/05, docs/06); this CLI never merges them.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, load_data_config
from .data.collator import build_dataloader
from .data.dataset import PhonemeSequenceDataset
from .data.schema import load_samples
from .data.validate import build_report
from .data.vocab import FeatureVocabulary
from .model.articulm_v1 import ArticuLMV1
from .runtime import describe_hardware, resolve_device
from .training.checkpoint import load_checkpoint
from .training.metrics import MetricsAccumulator, MetricsReport

LABEL_SETS = ("synthetic", "human_gold", "unspecified")


@dataclass
class FailureCase:
    sample_id: str
    position: int
    phoneme: str
    language: str
    target_viseme: int
    predicted_viseme: int
    target_strength: float
    predicted_strength: float

    @property
    def strength_error(self) -> float:
        return abs(self.predicted_strength - self.target_strength)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "position": self.position,
            "phoneme": self.phoneme,
            "language": self.language,
            "target_viseme_id": self.target_viseme,
            "predicted_viseme_id": self.predicted_viseme,
            "target_strength": round(self.target_strength, 3),
            "predicted_strength": round(self.predicted_strength, 3),
            "strength_absolute_error": round(self.strength_error, 3),
            "viseme_correct": self.target_viseme == self.predicted_viseme,
        }


@dataclass
class EvaluationResult:
    report: MetricsReport
    failures: list[FailureCase]
    label_set: str
    dataset_path: str
    checkpoint_path: str
    num_sequences: int


@torch.no_grad()
def evaluate_dataset(
    model: ArticuLMV1,
    dataset: PhonemeSequenceDataset,
    *,
    device: torch.device,
    batch_size: int = 16,
    max_seq_len: int | None = None,
    num_classes: int = 18,
    strength_scale: float = 100.0,
    max_failures: int = 500,
) -> tuple[MetricsReport, list[FailureCase]]:
    """Run metrics and collect the worst failure cases."""
    loader = build_dataloader(
        dataset,
        strategy="fixed_samples",
        batch_size=batch_size,
        shuffle=False,
        max_seq_len=max_seq_len,
        collect_slices=True,
    )
    accumulator = MetricsAccumulator(
        num_classes=num_classes, strength_scale=strength_scale
    )
    failures: list[FailureCase] = []

    model.eval()
    for batch in loader:
        moved = batch.to(device)
        output = model(moved.feature_ids, moved.attention_mask)
        accumulator.update_from_batch(output, moved)

        mask = moved.loss_mask.bool().cpu()
        predicted_visemes = output.viseme_logits.argmax(dim=-1).cpu()
        predicted_strength = (output.strength_norm * strength_scale).float().cpu()
        target_visemes = batch.viseme_targets
        target_strength = batch.strength_targets * strength_scale
        assert target_visemes is not None and target_strength is not None

        flat_index = 0
        for row in range(mask.shape[0]):
            for position in range(mask.shape[1]):
                if not bool(mask[row, position]):
                    continue
                phoneme = (
                    batch.phonemes[flat_index] if flat_index < len(batch.phonemes) else ""
                )
                language = ""
                if batch.slices and "language" in batch.slices:
                    language = batch.slices["language"][flat_index]
                flat_index += 1

                target_viseme = int(target_visemes[row, position])
                predicted_viseme = int(predicted_visemes[row, position])
                target_value = float(target_strength[row, position])
                predicted_value = float(predicted_strength[row, position])
                wrong_viseme = target_viseme != predicted_viseme
                large_strength_error = abs(predicted_value - target_value) > 15.0
                if wrong_viseme or large_strength_error:
                    failures.append(
                        FailureCase(
                            sample_id=batch.sample_ids[row],
                            position=position,
                            phoneme=phoneme,
                            language=language,
                            target_viseme=target_viseme,
                            predicted_viseme=predicted_viseme,
                            target_strength=target_value,
                            predicted_strength=predicted_value,
                        )
                    )

    failures.sort(
        key=lambda case: (case.target_viseme == case.predicted_viseme, -case.strength_error)
    )
    return accumulator.compute(), failures[:max_failures]


def write_artifacts(result: EvaluationResult, out_dir: str | Path) -> dict[str, Path]:
    """Write metrics.json plus the CSV / JSONL artifacts."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = result.report
    written: dict[str, Path] = {}

    metrics_path = directory / "metrics.json"
    payload = {
        "label_set": result.label_set,
        "dataset": result.dataset_path,
        "checkpoint": result.checkpoint_path,
        "note": (
            "Synthetic and Human Gold results must be reported separately; "
            "this file covers exactly one label_set."
        ),
        **report.as_dict(include_details=True),
    }
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    written["metrics"] = metrics_path

    per_class_path = directory / "per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["viseme_id", "support", "precision", "recall", "f1", "strength_mae"]
        )
        for viseme_id in range(report.viseme.num_classes):
            writer.writerow(
                [
                    viseme_id,
                    report.viseme.support[viseme_id],
                    f"{report.viseme.per_class_precision[viseme_id]:.6f}",
                    f"{report.viseme.per_class_recall[viseme_id]:.6f}",
                    f"{report.viseme.per_class_f1[viseme_id]:.6f}",
                    f"{report.strength.per_viseme_mae.get(viseme_id, float('nan')):.6f}",
                ]
            )
    written["per_class"] = per_class_path

    confusion_path = directory / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["target"] + [f"pred_{i}" for i in range(report.viseme.num_classes)])
        for target, row in enumerate(report.viseme.confusion_matrix):
            writer.writerow([target, *row])
    written["confusion_matrix"] = confusion_path

    strength_path = directory / "strength_report.csv"
    with strength_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scope", "key", "num_tokens", "mae", "rmse"])
        writer.writerow(
            [
                "overall",
                "all",
                report.strength.count,
                f"{report.strength.mae:.6f}",
                f"{report.strength.rmse:.6f}",
            ]
        )
        for viseme_id, mae in sorted(report.strength.per_viseme_mae.items()):
            writer.writerow(
                ["viseme", viseme_id, report.viseme.support[viseme_id], f"{mae:.6f}", ""]
            )
        for slice_name, values in report.slices.items():
            for key, stats in values.items():
                writer.writerow(
                    [
                        slice_name,
                        key,
                        stats["num_tokens"],
                        f"{float(stats['strength_mae']):.6f}",
                        f"{float(stats['strength_rmse']):.6f}",
                    ]
                )
    written["strength_report"] = strength_path

    failures_path = directory / "failure_cases.jsonl"
    with failures_path.open("w", encoding="utf-8") as fh:
        for case in result.failures:
            fh.write(json.dumps(case.as_dict(), ensure_ascii=False) + "\n")
    written["failure_cases"] = failures_path

    return written


def render_summary(result: EvaluationResult) -> str:
    report = result.report
    lines = [
        "=" * 72,
        f"ArticuLM-V1 evaluation — label_set={result.label_set}",
        "=" * 72,
        f"Checkpoint:        {result.checkpoint_path}",
        f"Dataset:           {result.dataset_path}",
        f"Sequences:         {result.num_sequences:,}",
        f"Tokens (unpadded): {report.num_tokens:,}",
        "",
        "Viseme:",
        f"  accuracy:        {report.viseme.accuracy:.4f}",
        f"  macro F1:        {report.viseme.macro_f1:.4f}",
        f"  weighted F1:     {report.viseme.weighted_f1:.4f}",
        "",
        "Strength (0..100 units):",
        f"  MAE:             {report.strength.mae:.4f}",
        f"  RMSE:            {report.strength.rmse:.4f}",
        f"  median abs err:  {report.strength.median_absolute_error:.4f}",
        "",
        f"Failure cases collected: {len(result.failures):,}",
    ]
    for slice_name, values in report.slices.items():
        lines.append("")
        lines.append(f"Slice — {slice_name}:")
        for key, stats in values.items():
            lines.append(
                f"  {key:<16} n={stats['num_tokens']:>8,}  "
                f"acc={float(stats['viseme_accuracy']):.4f}  "
                f"macroF1={float(stats['viseme_macro_f1']):.4f}  "
                f"MAE={float(stats['strength_mae']):.3f}"
            )
    lines.append("=" * 72)
    return "\n".join(lines)


def _resolve_dataset_path(
    explicit: str | None, split: str, data_cfg: DataConfig
) -> str:
    if explicit:
        return explicit
    mapping = {
        "train": data_cfg.train_path,
        "validation": data_cfg.validation_path,
        "test": data_cfg.test_path,
    }
    path = mapping.get(split)
    if not path:
        raise SystemExit(f"no dataset path configured for split={split!r}; pass --data")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.evaluate", description="Evaluate an ArticuLM-V1 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default=None, help="JSONL to evaluate (overrides --split)")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument(
        "--data-config",
        default=None,
        help="data config YAML; defaults to the one stored in the checkpoint",
    )
    parser.add_argument(
        "--label-set",
        default="unspecified",
        choices=list(LABEL_SETS),
        help="which label population this dataset represents (reported verbatim)",
    )
    parser.add_argument("--out-dir", default=None, help="directory for metrics artifacts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-failures", type=int, default=500)
    args = parser.parse_args(argv)

    loaded = load_checkpoint(args.checkpoint)
    data_cfg = load_data_config(args.data_config) if args.data_config else loaded.data_config
    vocab: FeatureVocabulary = loaded.vocab

    dataset_path = _resolve_dataset_path(args.data, args.split, data_cfg)
    samples = load_samples(dataset_path, data_cfg, require_labels=True)
    dataset = PhonemeSequenceDataset(
        samples, vocab, strength_scale=data_cfg.labels.strength_max
    )

    device = resolve_device(args.device)
    hardware = describe_hardware(device)
    model = ArticuLMV1.from_vocabulary(loaded.model_config, vocab).to(device)
    model.load_state_dict(loaded.payload["model_state_dict"], strict=True)

    data_report = build_report(samples, data_cfg, path=dataset_path, vocab=vocab)
    print(f"device: {device} ({hardware.device_name})")
    print(f"unknown phoneme rate: {(data_report.unknown_phoneme_rate or 0.0) * 100:.4f}%")

    report, failures = evaluate_dataset(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=data_cfg.max_seq_len,
        num_classes=loaded.model_config.viseme_head.num_classes,
        strength_scale=loaded.model_config.strength_head.output_scale,
        max_failures=args.max_failures,
    )
    result = EvaluationResult(
        report=report,
        failures=failures,
        label_set=args.label_set,
        dataset_path=dataset_path,
        checkpoint_path=args.checkpoint,
        num_sequences=len(samples),
    )
    print(render_summary(result))

    if args.out_dir:
        written = write_artifacts(result, args.out_dir)
        print("\nartifacts:")
        for name, path in written.items():
            print(f"  {name:<18} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
