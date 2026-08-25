"""Inference CLI.

```bash
python -m articulm.infer --checkpoint runs/.../checkpoints/best.pt \
    --input examples/sample.jsonl --output predictions.jsonl
```

Input is the same non-label feature schema used in training; a ``labels``
block, if present, is ignored. Output per phoneme:

```json
{"phoneme": "n", "viseme_id": 14, "strength": 62.0}
```

Baseline decoding only:

```text
viseme_id = argmax(viseme_logits)
strength  = sigmoid(raw) * 100
```

No smoothing, no post-hoc heuristics, no extra rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, load_data_config
from .data.collator import PhonemeCollator
from .data.dataset import encode_sample
from .data.schema import Sample, iter_jsonl, parse_sample
from .data.vocab import FeatureVocabulary
from .model.articulm_v1 import ArticuLMV1
from .runtime import resolve_device
from .training.checkpoint import load_checkpoint


def load_inference_samples(
    path: str | Path, cfg: DataConfig
) -> list[Sample]:
    """Parse an input file without requiring labels."""
    samples: list[Sample] = []
    for line_no, record in iter_jsonl(path):
        samples.append(
            parse_sample(
                record, cfg, require_labels=False, location=f"{path}:{line_no}"
            )
        )
    if not samples:
        raise SystemExit(f"{path}: no samples found")
    return samples


@torch.no_grad()
def predict_samples(
    model: ArticuLMV1,
    samples: Sequence[Sample],
    vocab: FeatureVocabulary,
    *,
    device: torch.device,
    batch_size: int = 16,
    max_seq_len: int | None = None,
    strength_scale: float = 100.0,
    round_strength: int | None = 1,
) -> list[dict[str, Any]]:
    """Run baseline decoding over samples and return per-sentence outputs."""
    model.eval()
    collator = PhonemeCollator(
        max_seq_len=max_seq_len, collect_slices=False, collect_phonemes=False
    )
    results: list[dict[str, Any]] = []

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        encoded = [encode_sample(sample, vocab) for sample in chunk]
        batch = collator(encoded).to(device)

        output = model(batch.feature_ids, batch.attention_mask)
        viseme_ids = output.predicted_viseme_ids().cpu()
        strength = (output.strength_norm * strength_scale).float().cpu()

        for row, sample in enumerate(chunk):
            outputs = []
            for position, token in enumerate(sample.tokens):
                value = float(strength[row, position])
                outputs.append(
                    {
                        "phoneme": token.phoneme,
                        "viseme_id": int(viseme_ids[row, position]),
                        "strength": round(value, round_strength)
                        if round_strength is not None
                        else value,
                    }
                )
            results.append(
                {
                    "sample_id": sample.sample_id,
                    "text": sample.text,
                    "outputs": outputs,
                }
            )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.infer", description="Run ArticuLM-V1 inference."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="JSONL of feature-only samples")
    parser.add_argument("--output", default=None, help="JSONL output path (default: stdout)")
    parser.add_argument(
        "--data-config",
        default=None,
        help="data config YAML; defaults to the one stored in the checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    loaded = load_checkpoint(args.checkpoint)
    data_cfg = load_data_config(args.data_config) if args.data_config else loaded.data_config
    vocab = loaded.vocab

    samples = load_inference_samples(args.input, data_cfg)
    device = resolve_device(args.device)
    model = ArticuLMV1.from_vocabulary(loaded.model_config, vocab).to(device)
    model.load_state_dict(loaded.payload["model_state_dict"], strict=True)

    predictions = predict_samples(
        model,
        samples,
        vocab,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=data_cfg.max_seq_len,
        strength_scale=loaded.model_config.strength_head.output_scale,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for prediction in predictions:
                fh.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        print(f"wrote {out_path} ({len(predictions)} sentences)")
    else:
        for prediction in predictions:
            json.dump(prediction, sys.stdout, ensure_ascii=False)
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
