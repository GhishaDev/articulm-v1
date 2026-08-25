"""Training CLI.

```bash
python -m articulm.train --config config/train_tiny_overfit.yaml
python -m articulm.train --config config/train_v1_50m.yaml
python -m articulm.train --config config/train_v1_50m.yaml --resume runs/.../checkpoints/last.pt
python -m articulm.train --config config/train_v1_50m.yaml --dry-run
```

``--dry-run`` prints the full pre-flight report (dataset stats, GPU, precision,
estimated batch memory, checkpoint directory) and exits without training.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainRunConfig, dump_yaml, load_train_config
from .data.collator import build_dataloader
from .data.dataset import PhonemeSequenceDataset
from .data.schema import load_samples
from .data.validate import DatasetReport, build_report
from .data.vocab import FeatureVocabulary, build_vocabulary
from .model.articulm_v1 import ArticuLMV1
from .runtime import (
    HardwareInfo,
    PrecisionPlan,
    StructuredLogger,
    describe_hardware,
    make_run_id,
    resolve_device,
    resolve_precision,
    set_seed,
)
from .training.checkpoint import load_checkpoint
from .training.trainer import Trainer


@dataclass
class PreparedRun:
    """Everything assembled before the first optimizer step."""

    config: TrainRunConfig
    run_dir: Path
    logger: StructuredLogger
    vocab: FeatureVocabulary
    model: ArticuLMV1
    train_dataset: PhonemeSequenceDataset
    validation_dataset: PhonemeSequenceDataset | None
    train_report: DatasetReport
    validation_report: DatasetReport | None
    hardware: HardwareInfo
    precision: PrecisionPlan
    device: torch.device


def estimate_activation_memory_bytes(
    *,
    batch_tokens: int,
    hidden_size: int,
    num_layers: int,
    ffn_size: int,
    bytes_per_element: int,
) -> int:
    """Rough forward-activation estimate for the batch-size sanity print.

    Counts the dominant per-token tensors kept for backward in each block
    (residual stream, attention projections, FFN intermediate). This is an
    order-of-magnitude guide, not a substitute for measuring real usage.
    """
    per_token_per_layer = 6 * hidden_size + 2 * ffn_size
    return batch_tokens * num_layers * per_token_per_layer * bytes_per_element


def _encoded_cache_path(data_cfg, source_path: str) -> Path | None:
    """Cache location for a data split, or ``None`` when caching is disabled."""
    if not getattr(data_cfg, "encoded_cache", False) or not source_path:
        return None
    from .data.cache import default_cache_path

    return default_cache_path(source_path)


def _load_data_split(path: str, data_cfg, limit: int | None):
    """Load a data split: lossless parquet (``*.tokens.parquet``) or JSONL."""
    if str(path).endswith(".tokens.parquet"):
        from .data.parquet import load_samples_parquet

        return load_samples_parquet(path, data_cfg, limit=limit)
    return load_samples(path, data_cfg, limit=limit)


def prepare_run(
    config_path: str,
    *,
    device_override: str | None = None,
    limit: int | None = None,
    run_dir_override: str | None = None,
    vocab_path: str | None = None,
) -> PreparedRun:
    """Load configs and data, build the vocab and model, report everything."""
    config = load_train_config(config_path)
    set_seed(config.experiment.seed)

    run_id = make_run_id(config.experiment.name)
    run_dir = Path(run_dir_override) if run_dir_override else Path(config.experiment.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(run_id, log_dir=run_dir / "logs")

    logger.event(
        "config",
        "configs loaded",
        train_config=config_path,
        model_config=config.model_config_path,
        data_config=config.data_config_path,
        training_stage=config.training.stage,
        seed=config.experiment.seed,
        run_dir=str(run_dir),
    )

    data_cfg = config.data
    tiny = config.training.tiny_subset.num_samples
    effective_limit = min(x for x in (limit, tiny) if x is not None) if (limit or tiny) else None

    train_samples = _load_data_split(data_cfg.train_path, data_cfg, limit=effective_limit)
    logger.event(
        "data_loaded",
        "train split loaded",
        path=data_cfg.train_path,
        sentences=len(train_samples),
        tokens=sum(len(s) for s in train_samples),
    )

    if vocab_path:
        vocab = FeatureVocabulary.load(vocab_path)
        logger.event("vocab", "vocabulary loaded from file", path=vocab_path)
    else:
        vocab = build_vocabulary(train_samples, data_cfg)
        logger.event("vocab", "vocabulary built from train split", sizes=vocab.sizes())

    train_report = build_report(
        train_samples, data_cfg, path=data_cfg.train_path, vocab=vocab
    )

    validation_samples = None
    validation_report = None
    if data_cfg.validation_path:
        validation_samples = _load_data_split(
            data_cfg.validation_path, data_cfg, limit=effective_limit
        )
        validation_report = build_report(
            validation_samples, data_cfg, path=data_cfg.validation_path, vocab=vocab
        )

    source_weights = dict(config.training.loss.strength.source_weights)
    train_dataset = PhonemeSequenceDataset(
        train_samples, vocab, source_weights=source_weights,
        strength_scale=data_cfg.labels.strength_max,
        cache_path=_encoded_cache_path(data_cfg, data_cfg.train_path),
        source_path=data_cfg.train_path,
    )
    logger.event(
        "dataset_cache",
        "train dataset encode cache",
        state=train_dataset.cache_state,
    )
    validation_dataset = (
        PhonemeSequenceDataset(
            validation_samples, vocab, source_weights=source_weights,
            strength_scale=data_cfg.labels.strength_max,
            cache_path=_encoded_cache_path(data_cfg, data_cfg.validation_path),
            source_path=data_cfg.validation_path,
        )
        if validation_samples
        else None
    )

    device = resolve_device(device_override)
    hardware = describe_hardware(device)
    precision = resolve_precision(config.training.precision, hardware)

    model = ArticuLMV1.from_vocabulary(config.model, vocab)

    # Persist the exact run inputs next to the checkpoints.
    dump_yaml(config.model, run_dir / "model_config.yaml")
    dump_yaml(config.data, run_dir / "data_config.yaml")
    dump_yaml(config.training, run_dir / "train_config.yaml")
    vocab.save(run_dir / "vocab" / "feature_vocab.json")

    return PreparedRun(
        config=config,
        run_dir=run_dir,
        logger=logger,
        vocab=vocab,
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_report=train_report,
        validation_report=validation_report,
        hardware=hardware,
        precision=precision,
        device=device,
    )


def render_preflight(prepared: PreparedRun) -> str:
    """The report that must be read before launching a long run."""
    config = prepared.config
    batching = config.training.batching
    transformer = config.model.transformer
    bytes_per_element = 2 if prepared.precision.name in {"fp16", "bf16"} else 4

    if batching.strategy == "dynamic_phoneme_tokens":
        batch_tokens = batching.max_phoneme_tokens_per_batch
        batch_description = (
            f"dynamic, <= {batch_tokens} padded phoneme tokens per micro-batch"
        )
    else:
        batch_tokens = batching.batch_size * prepared.train_report.seq_len_max
        batch_description = (
            f"fixed, {batching.batch_size} sentences "
            f"(worst case {batch_tokens} padded tokens)"
        )

    activation_bytes = estimate_activation_memory_bytes(
        batch_tokens=batch_tokens,
        hidden_size=transformer.hidden_size,
        num_layers=transformer.num_layers,
        ffn_size=transformer.ffn_size,
        bytes_per_element=bytes_per_element,
    )
    parameters = prepared.model.num_parameters()
    # AdamW: fp32 params + grads + 2 moments.
    optimizer_bytes = parameters * 4 * 4

    breakdown = prepared.model.parameter_breakdown().as_dict()

    lines = [
        "=" * 72,
        "ArticuLM-V1 pre-flight report",
        "=" * 72,
        f"Experiment:               {config.experiment.name}",
        f"Stage:                    {config.training.stage}",
        f"Run dir:                  {prepared.run_dir}",
        f"Checkpoint dir:           {prepared.run_dir / 'checkpoints'}",
        f"Seed:                     {config.experiment.seed}",
        "",
        "--- data ---",
        prepared.train_report.render(),
    ]
    if prepared.validation_report is not None:
        lines += [
            "",
            "--- validation split ---",
            f"Sentences:                {prepared.validation_report.num_sentences:,}",
            f"Phoneme Tokens:           {prepared.validation_report.num_phoneme_tokens:,}",
            (
                "Unknown Phoneme Rate:     "
                f"{(prepared.validation_report.unknown_phoneme_rate or 0.0) * 100:.4f}%"
            ),
        ]

    lines += [
        "",
        "--- model ---",
        f"Name:                     {config.model.name}",
        f"Layers x hidden:          {transformer.num_layers} x {transformer.hidden_size}",
        f"Heads x head_dim:         {transformer.num_heads} x {transformer.head_dim}",
        f"FFN:                      {transformer.ffn_size}",
        f"Position encoding:        {config.model.position.type}",
        (
            "Local Conv:               "
            f"{'ON' if config.model.local_conv.enabled else 'OFF'} "
            f"(kernel {config.model.local_conv.kernel_size}, "
            f"every {config.model.local_conv.every_n_layers} layers)"
        ),
        f"Total parameters:         {parameters:,}",
        "Parameter breakdown:",
    ]
    lines += [f"  {name:<24} {count:>12,}" for name, count in breakdown.items()]

    lines += [
        "",
        "--- hardware / precision ---",
        f"Device:                   {prepared.hardware.device} ({prepared.hardware.device_name})",
        f"Compute capability:       {prepared.hardware.as_dict()['compute_capability']}",
        f"Device memory:            {prepared.hardware.as_dict()['total_memory_gb']} GB",
        f"Torch / CUDA:             {prepared.hardware.torch_version} / {prepared.hardware.cuda_version}",
        f"BF16 supported:           {prepared.hardware.bf16_supported}",
        f"Precision:                {prepared.precision.name} ({prepared.precision.reason})",
        f"Grad scaler:              {prepared.precision.use_grad_scaler}",
        "",
        "--- batching / memory estimate ---",
        f"Strategy:                 {batch_description}",
        f"Grad accumulation:        {batching.gradient_accumulation_steps}",
        (
            f"Est. activations:         {activation_bytes / 1024**3:.2f} GB "
            "(dominant terms only, order-of-magnitude)"
        ),
        f"Est. weights+optimizer:   {optimizer_bytes / 1024**3:.2f} GB (fp32 AdamW)",
        "",
        "--- loss ---",
        (
            f"Viseme:                   {config.training.loss.viseme.type}, "
            f"weight={config.training.loss.viseme.weight}, "
            f"label_smoothing={config.training.loss.viseme.label_smoothing}"
        ),
        (
            f"Strength:                 {config.training.loss.strength.type}, "
            f"weight={config.training.loss.strength.weight}, "
            f"target=strength/{config.data.labels.strength_max:g}"
        ),
        f"Strength source weights:  {config.training.loss.strength.source_weights or '{} (all 1.0)'}",
        "=" * 72,
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.train", description="Train ArticuLM-V1."
    )
    parser.add_argument("--config", required=True, help="training config YAML")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "checkpoint to warm-start model weights from (fine-tuning); "
            "optimizer/scheduler/step start fresh, unlike --resume"
        ),
    )
    parser.add_argument("--device", default=None, help="override device (cuda/cpu/mps)")
    parser.add_argument("--limit", type=int, default=None, help="cap sentences per split")
    parser.add_argument("--run-dir", default=None, help="override the run directory")
    parser.add_argument("--vocab", default=None, help="load a frozen vocabulary instead of building one")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pre-flight report and exit without training",
    )
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    prepared = prepare_run(
        args.config,
        device_override=args.device,
        limit=args.limit,
        run_dir_override=args.run_dir,
        vocab_path=args.vocab,
    )
    print(render_preflight(prepared))

    if args.dry_run:
        print("\n--dry-run: stopping before training. Re-run without --dry-run to train.")
        return 0

    config = prepared.config
    batching = config.training.batching
    # Pinned host memory is what makes the trainer's non_blocking copies async.
    pin_memory = prepared.device.type == "cuda"
    train_loader = build_dataloader(
        prepared.train_dataset,
        strategy=batching.strategy,
        batch_size=batching.batch_size,
        max_phoneme_tokens_per_batch=batching.max_phoneme_tokens_per_batch,
        shuffle=batching.shuffle,
        seed=config.experiment.seed,
        num_workers=batching.num_workers,
        max_seq_len=config.data.max_seq_len,
        collect_slices=False,
        pin_memory=pin_memory,
    )
    validation_loader = None
    if prepared.validation_dataset is not None:
        validation_loader = build_dataloader(
            prepared.validation_dataset,
            strategy=batching.strategy,
            batch_size=batching.batch_size,
            max_phoneme_tokens_per_batch=batching.max_phoneme_tokens_per_batch,
            shuffle=False,
            seed=config.experiment.seed,
            num_workers=batching.num_workers,
            max_seq_len=config.data.max_seq_len,
            collect_slices=True,
            pin_memory=pin_memory,
        )

    trainer = Trainer(
        config=config,
        model=prepared.model,
        vocab=prepared.vocab,
        train_loader=train_loader,
        validation_loader=validation_loader,
        run_dir=prepared.run_dir,
        logger=prepared.logger,
        device=prepared.device,
        precision=prepared.precision,
    )

    if args.resume:
        loaded = load_checkpoint(args.resume)
        trainer.resume(loaded)

    if args.init_from:
        loaded = load_checkpoint(args.init_from)
        trainer.warm_start(loaded)

    summary = trainer.train()
    summary_path = prepared.run_dir / "metrics" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary.as_dict(), fh, ensure_ascii=False, indent=2)

    print("\n=== training summary ===")
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
