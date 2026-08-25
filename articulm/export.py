"""Export CLI.

Priority order from docs/07_inference_export.md:

```text
1. PyTorch checkpoint  (always available; this is the reference artifact)
2. TorchScript         (--format torchscript)
3. ONNX                (--format onnx — only ships after a parity test passes)
```

ONNX export always runs a numerical parity check against PyTorch and exports
with dynamic batch and dynamic sequence axes. A failed parity check is a
non-zero exit; the file is left in place for debugging but reported as FAILED.

Every export writes a sidecar ``<name>.metadata.json`` carrying the model
version, config hash, vocab version and checkpoint step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig, to_plain_dict
from .data.vocab import VOCAB_FORMAT_VERSION, FeatureVocabulary
from .model.articulm_v1 import ArticuLMV1
from .training.checkpoint import LoadedCheckpoint, load_checkpoint


class ExportError(RuntimeError):
    pass


class ArticuLMExportWrapper(nn.Module):
    """Tensor-in / tensor-out wrapper for tracing and ONNX.

    Returns ``(viseme_logits, strength_scaled)``. ``argmax`` stays outside the
    graph so downstream consumers see the same baseline decoding as
    ``articulm.infer``.
    """

    def __init__(self, model: ArticuLMV1) -> None:
        super().__init__()
        self.model = model
        self.strength_scale = model.strength_scale

    def forward(
        self, feature_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(feature_ids, attention_mask)
        return output.viseme_logits, output.strength_norm * self.strength_scale


def config_hash(model_config: ModelConfig) -> str:
    payload = json.dumps(to_plain_dict(model_config), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def vocab_hash(vocab: FeatureVocabulary) -> str:
    payload = json.dumps(vocab.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_metadata(
    loaded: LoadedCheckpoint,
    *,
    checkpoint_path: str,
    export_format: str,
    parity: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_name": loaded.model_config.name,
        "export_format": export_format,
        "checkpoint": checkpoint_path,
        "checkpoint_step": loaded.state.global_step,
        "checkpoint_epoch": loaded.state.epoch,
        "config_hash": config_hash(loaded.model_config),
        "phoneme_vocab_size": len(loaded.vocab.fields["phoneme"]),
        "feature_vocab_version": VOCAB_FORMAT_VERSION,
        "feature_vocab_hash": vocab_hash(loaded.vocab),
        "data_schema_version": loaded.data_config.schema_version,
        "training_data_train_path": loaded.data_config.train_path,
        "seed": loaded.seed,
        "torch_version": torch.__version__,
        "parity_test": parity,
    }


def example_inputs(
    vocab: FeatureVocabulary,
    *,
    batch_size: int = 2,
    seq_len: int = 8,
    num_fields: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small valid input pair for tracing and parity checks.

    Ids stay inside every field's vocabulary and the second row is padded so
    the mask path is exercised during tracing.
    """
    from .data.vocab import FEATURE_KEYS

    fields = num_fields if num_fields is not None else len(FEATURE_KEYS)
    sizes = vocab.sizes()
    feature_ids = torch.zeros((batch_size, seq_len, fields), dtype=torch.long)
    for index, key in enumerate(FEATURE_KEYS[:fields]):
        # Cycle through valid non-PAD ids for this field.
        upper = max(sizes[key] - 1, 1)
        feature_ids[..., index] = (
            torch.arange(batch_size * seq_len).reshape(batch_size, seq_len) % upper
        ) + 1

    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    if batch_size > 1 and seq_len > 2:
        attention_mask[-1, -2:] = False
        feature_ids[-1, -2:, :] = 0
    return feature_ids, attention_mask


@dataclass
class ParityResult:
    passed: bool
    max_logit_difference: float
    max_strength_difference: float
    logit_tolerance: float
    strength_tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "max_logit_difference": self.max_logit_difference,
            "max_strength_difference": self.max_strength_difference,
            "logit_tolerance": self.logit_tolerance,
            "strength_tolerance": self.strength_tolerance,
        }


def export_torchscript(
    wrapper: ArticuLMExportWrapper, path: Path, inputs: tuple[torch.Tensor, torch.Tensor]
) -> Path:
    wrapper.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, inputs, strict=False)
    traced.save(str(path))
    return path


def export_onnx(
    wrapper: ArticuLMExportWrapper,
    path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    opset: int = 17,
) -> Path:
    wrapper.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _export_onnx_graph(wrapper, path, inputs, opset=opset)
    except ModuleNotFoundError as exc:
        raise ExportError(
            f"ONNX export needs an optional dependency that is not installed: {exc.name}. "
            "Install the export extras (`pip install onnx onnxscript onnxruntime`) or "
            "use --format torchscript."
        ) from exc
    return path


def _export_onnx_graph(
    wrapper: ArticuLMExportWrapper,
    path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    opset: int,
) -> None:
    torch.onnx.export(
        wrapper,
        inputs,
        str(path),
        input_names=["feature_ids", "attention_mask"],
        output_names=["viseme_logits", "strength"],
        dynamic_axes={
            "feature_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "viseme_logits": {0: "batch", 1: "sequence"},
            "strength": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return path


def check_onnx_parity(
    wrapper: ArticuLMExportWrapper,
    onnx_path: Path,
    inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    logit_tolerance: float = 1e-3,
    strength_tolerance: float = 1e-2,
) -> ParityResult:
    """Compare ONNX Runtime outputs against PyTorch on the same inputs."""
    try:
        import onnxruntime  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ExportError(
            "onnxruntime is required for the ONNX parity test. "
            "Install it (`pip install onnxruntime`) or export TorchScript instead."
        ) from exc

    wrapper.eval()
    with torch.no_grad():
        reference_logits, reference_strength = wrapper(*inputs)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_logits, onnx_strength = session.run(
        None,
        {
            "feature_ids": inputs[0].numpy(),
            "attention_mask": inputs[1].numpy(),
        },
    )

    logit_difference = float(
        (reference_logits - torch.from_numpy(onnx_logits)).abs().max()
    )
    strength_difference = float(
        (reference_strength - torch.from_numpy(onnx_strength)).abs().max()
    )
    return ParityResult(
        passed=logit_difference <= logit_tolerance
        and strength_difference <= strength_tolerance,
        max_logit_difference=logit_difference,
        max_strength_difference=strength_difference,
        logit_tolerance=logit_tolerance,
        strength_tolerance=strength_tolerance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.export", description="Export an ArticuLM-V1 checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output file path")
    parser.add_argument(
        "--format",
        default="torchscript",
        choices=["torchscript", "onnx"],
        help="PyTorch checkpoints are already the reference artifact",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seq-len", type=int, default=16, help="tracing sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="tracing batch size")
    args = parser.parse_args(argv)

    loaded = load_checkpoint(args.checkpoint)
    model = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
    model.load_state_dict(loaded.payload["model_state_dict"], strict=True)
    wrapper = ArticuLMExportWrapper(model)
    wrapper.eval()

    inputs = example_inputs(
        loaded.vocab, batch_size=args.batch_size, seq_len=args.seq_len
    )
    out_path = Path(args.out)
    parity: dict[str, Any] | None = None
    exit_code = 0

    if args.format == "torchscript":
        export_torchscript(wrapper, out_path, inputs)
        print(f"exported TorchScript: {out_path}")
    else:
        export_onnx(wrapper, out_path, inputs, opset=args.opset)
        print(f"exported ONNX: {out_path}")
        result = check_onnx_parity(wrapper, out_path, inputs)
        parity = result.as_dict()
        status = "PASSED" if result.passed else "FAILED"
        print(
            f"ONNX parity {status}: max|logit diff|={result.max_logit_difference:.3e} "
            f"(tol {result.logit_tolerance:.1e}), "
            f"max|strength diff|={result.max_strength_difference:.3e} "
            f"(tol {result.strength_tolerance:.1e})"
        )
        if not result.passed:
            print("Do not ship this ONNX artifact until parity passes.")
            exit_code = 1

    metadata = build_metadata(
        loaded,
        checkpoint_path=args.checkpoint,
        export_format=args.format,
        parity=parity,
    )
    metadata_path = out_path.with_suffix(out_path.suffix + ".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    print(f"wrote {metadata_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
