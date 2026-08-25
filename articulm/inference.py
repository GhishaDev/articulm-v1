"""High-level programmatic inference API.

Load a trained checkpoint once and predict over feature-only samples:

```python
from articulm.inference import ModelPredictor

predictor = ModelPredictor.load("archive/strength_v2_fast_20260824/model/best.pt")
results = predictor.predict_samples(samples)          # [{"sample_id", "text", "outputs": [...]}]
results = predictor.predict_jsonl("examples/sample.jsonl")
```

Each per-token output is ``{"phoneme", "viseme_id", "strength"}`` using baseline
decoding (``argmax`` over viseme logits, ``sigmoid * scale`` for strength) — the
same decoding as the ``python -m articulm.infer`` CLI, with no post-hoc rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ModelConfig, load_data_config
from .data.schema import Sample, load_samples
from .data.vocab import FeatureVocabulary
from .infer import predict_samples
from .model.articulm_v1 import ArticuLMV1
from .runtime import resolve_device
from .training.checkpoint import load_checkpoint
from .visemes import viseme_name


class ModelPredictor:
    """A checkpoint loaded for inference: model + vocab + configs + device."""

    def __init__(
        self,
        *,
        model: ArticuLMV1,
        vocab: FeatureVocabulary,
        model_config: ModelConfig,
        data_config: DataConfig,
        device: torch.device,
        strength_scale: float = 100.0,
    ) -> None:
        self.model = model.eval()
        self.vocab = vocab
        self.model_config = model_config
        self.data_config = data_config
        self.device = device
        self.strength_scale = strength_scale

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str | None = None) -> "ModelPredictor":
        """Load a checkpoint (model weights + vocabulary + configs) for inference.

        ``device`` is optional; when omitted, resolution follows the same
        ``--device`` rules as training (cuda when available, else cpu).
        """
        resolved = resolve_device(device)
        loaded = load_checkpoint(checkpoint_path, map_location=str(resolved))

        model = ArticuLMV1.from_vocabulary(loaded.model_config, loaded.vocab)
        model.load_state_dict(loaded.payload["model_state_dict"])
        model.to(resolved).eval()

        return cls(
            model=model,
            vocab=loaded.vocab,
            model_config=loaded.model_config,
            data_config=loaded.data_config,
            device=resolved,
            strength_scale=loaded.model_config.strength_head.output_scale,
        )

    def predict_samples(
        self,
        samples: Sequence[Sample],
        *,
        batch_size: int = 16,
        round_strength: int | None = 1,
        label_names: bool = True,
    ) -> list[dict[str, Any]]:
        """Baseline-decode a list of :class:`Sample` objects.

        Each per-token output is ``{"phoneme", "viseme", "strength"}`` where
        ``viseme`` is the shapeV2 name (e.g. ``"304_Out"``). Set
        ``label_names=False`` to get the numeric id under ``"viseme_id"``
        instead (the raw classifier argmax).
        """
        raw = predict_samples(
            self.model,
            samples,
            self.vocab,
            device=self.device,
            batch_size=batch_size,
            max_seq_len=self.data_config.max_seq_len,
            strength_scale=self.strength_scale,
            round_strength=round_strength,
        )
        if not label_names:
            return raw
        for result in raw:
            for item in result["outputs"]:
                item["viseme"] = viseme_name(item.pop("viseme_id"))
        return raw

    def predict_jsonl(
        self,
        path: str | Path,
        *,
        batch_size: int = 16,
        round_strength: int | None = 1,
        label_names: bool = True,
        data_config: DataConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Parse a JSONL of feature-only samples and decode them.

        ``data_config`` defaults to the checkpoint's data config; pass one to
        override the label/feature schema expectations (e.g. different paths).
        """
        cfg = data_config or self.data_config
        samples = load_samples(path, cfg, require_labels=False, limit=None)
        return self.predict_samples(
            samples,
            batch_size=batch_size,
            round_strength=round_strength,
            label_names=label_names,
        )

    @property
    def num_viseme_classes(self) -> int:
        return self.model_config.viseme_head.num_classes
