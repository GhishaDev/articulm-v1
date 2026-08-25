"""Per-field feature embeddings concatenated to the fused input width.

Only the documented encoder features are embedded:

```text
phoneme / language / surface_tone / stress / syllable_role
articulatory (8 sub-fields) / boundary (5 sub-fields)
```

There is no code path here that can read ``viseme_id``, ``strength``,
``shapeV2``, ``Talk``, ``raw_value``, ``duration`` or ``timing``: the module
consumes an id tensor whose columns are fixed by
:data:`articulm.data.vocab.FEATURE_KEYS`.

Every table uses ``padding_idx=PAD_ID``, so a PAD token embeds to an exact
zero vector.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import ModelConfig
from ..data.schema import ARTICULATORY_FIELDS, BOUNDARY_FIELDS
from ..data.vocab import FEATURE_KEYS, PAD_ID


class FeatureEmbeddingError(ValueError):
    """Raised on a vocab/dimension mismatch in the embedding stack."""


def split_dim_evenly(total: int, parts: int, field_name: str) -> tuple[int, ...]:
    """Split a composite embedding width into equal per-sub-field widths."""
    if total <= 0:
        raise FeatureEmbeddingError(f"{field_name}: embedding dim must be positive, got {total}")
    if total % parts != 0:
        raise FeatureEmbeddingError(
            f"{field_name}: embedding dim {total} is not divisible by its "
            f"{parts} sub-fields; choose a multiple of {parts}"
        )
    return tuple([total // parts] * parts)


class PhonemeFeatureEmbedding(nn.Module):
    """Embed all encoder feature fields and concatenate them.

    Input:  ``feature_ids`` ``[B, T, F]`` long, columns in ``FEATURE_KEYS`` order.
    Output: ``[B, T, fused_input_dim]``.
    """

    def __init__(self, cfg: ModelConfig, vocab_sizes: dict[str, int]) -> None:
        super().__init__()
        dims = cfg.input.embedding_dims

        missing = [key for key in FEATURE_KEYS if key not in vocab_sizes]
        if missing:
            raise FeatureEmbeddingError(f"missing vocab sizes for fields {missing}")

        # Articulatory sub-field widths are configured explicitly. Only fall
        # back to an even split if the config left them inconsistent.
        configured = dims.articulatory.as_dict()
        if sum(configured.values()) == dims.articulatory.total:
            articulatory_dims = tuple(configured[name] for name in ARTICULATORY_FIELDS)
        else:
            articulatory_dims = split_dim_evenly(
                dims.articulatory.total, len(ARTICULATORY_FIELDS), "articulatory"
            )

        # `boundary` is a single composite width in the config; split it evenly
        # across its 5 sub-fields.
        boundary_dims = split_dim_evenly(dims.boundary, len(BOUNDARY_FIELDS), "boundary")

        self.field_dims: dict[str, int] = {
            "phoneme": dims.phoneme,
            "language": dims.language,
            "surface_tone": dims.surface_tone,
            "stress": dims.stress,
            "syllable_role": dims.syllable_role,
        }
        for name, dim in zip(ARTICULATORY_FIELDS, articulatory_dims, strict=True):
            self.field_dims[f"articulatory.{name}"] = dim
        for name, dim in zip(BOUNDARY_FIELDS, boundary_dims, strict=True):
            self.field_dims[f"boundary.{name}"] = dim

        total = sum(self.field_dims[key] for key in FEATURE_KEYS)
        if total != cfg.input.fused_input_dim:
            raise FeatureEmbeddingError(
                f"concatenated embedding width {total} != "
                f"model.input.fused_input_dim {cfg.input.fused_input_dim}"
            )

        # ModuleDict keys cannot contain '.', so store a sanitised alias.
        self._module_keys = {key: key.replace(".", "__") for key in FEATURE_KEYS}
        self.tables = nn.ModuleDict(
            {
                self._module_keys[key]: nn.Embedding(
                    num_embeddings=vocab_sizes[key],
                    embedding_dim=self.field_dims[key],
                    padding_idx=PAD_ID,
                )
                for key in FEATURE_KEYS
            }
        )
        self.output_dim = total
        self.vocab_sizes = dict(vocab_sizes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in self.tables.values():
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                table.weight[PAD_ID].zero_()

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        if feature_ids.dim() != 3:
            raise FeatureEmbeddingError(
                f"feature_ids must be [B,T,F], got shape {tuple(feature_ids.shape)}"
            )
        if feature_ids.shape[-1] != len(FEATURE_KEYS):
            raise FeatureEmbeddingError(
                f"feature_ids last dim must be {len(FEATURE_KEYS)} "
                f"(len(FEATURE_KEYS)), got {feature_ids.shape[-1]}"
            )

        pieces = [
            self.tables[self._module_keys[key]](feature_ids[..., index])
            for index, key in enumerate(FEATURE_KEYS)
        ]
        return torch.cat(pieces, dim=-1)
