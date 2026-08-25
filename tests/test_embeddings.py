"""Feature embedding and fusion shapes."""

from __future__ import annotations

import pytest
import torch

from articulm.data.collator import PhonemeCollator
from articulm.data.vocab import FEATURE_KEYS, PAD_ID
from articulm.model.embeddings import (
    FeatureEmbeddingError,
    PhonemeFeatureEmbedding,
    split_dim_evenly,
)
from articulm.model.fusion import FeatureFusion


@pytest.fixture
def embedding(model_config, vocab) -> PhonemeFeatureEmbedding:
    return PhonemeFeatureEmbedding(model_config, vocab.sizes())


def test_concatenated_width_is_384(embedding):
    assert embedding.output_dim == 384
    assert sum(embedding.field_dims[key] for key in FEATURE_KEYS) == 384


def test_documented_per_field_dims(embedding):
    assert embedding.field_dims["phoneme"] == 256
    assert embedding.field_dims["language"] == 8
    assert embedding.field_dims["surface_tone"] == 16
    assert embedding.field_dims["stress"] == 8
    assert embedding.field_dims["syllable_role"] == 16
    articulatory = sum(
        dim for key, dim in embedding.field_dims.items() if key.startswith("articulatory.")
    )
    boundary = sum(
        dim for key, dim in embedding.field_dims.items() if key.startswith("boundary.")
    )
    assert articulatory == 60
    assert boundary == 20


def test_articulatory_sub_field_dims_match_config(embedding, model_config):
    configured = model_config.input.embedding_dims.articulatory.as_dict()
    for name, dim in configured.items():
        assert embedding.field_dims[f"articulatory.{name}"] == dim


def test_forward_shape(embedding, dataset):
    batch = PhonemeCollator()(list(dataset.encoded))
    out = embedding(batch.feature_ids)
    assert out.shape == (batch.batch_size, batch.max_length, 384)
    assert out.dtype == torch.float32


def test_pad_embeds_to_exactly_zero(embedding):
    ids = torch.zeros((1, 3, len(FEATURE_KEYS)), dtype=torch.long)
    out = embedding(ids)
    assert float(out.detach().abs().max()) == 0.0


def test_pad_row_stays_zero_after_training_step(embedding):
    """padding_idx must keep the PAD row at zero across updates."""
    optimizer = torch.optim.SGD(embedding.parameters(), lr=1.0)
    ids = torch.ones((2, 4, len(FEATURE_KEYS)), dtype=torch.long)
    embedding(ids).sum().backward()
    optimizer.step()
    for key in FEATURE_KEYS:
        table = embedding.tables[key.replace(".", "__")]
        assert float(table.weight[PAD_ID].detach().abs().max()) == 0.0


def test_wrong_field_count_is_rejected(embedding):
    ids = torch.ones((1, 2, len(FEATURE_KEYS) - 1), dtype=torch.long)
    with pytest.raises(FeatureEmbeddingError, match="last dim"):
        embedding(ids)


def test_non_3d_input_is_rejected(embedding):
    with pytest.raises(FeatureEmbeddingError, match=r"\[B,T,F\]"):
        embedding(torch.ones((2, len(FEATURE_KEYS)), dtype=torch.long))


def test_missing_vocab_size_is_rejected(model_config, vocab):
    sizes = vocab.sizes()
    sizes.pop("phoneme")
    with pytest.raises(FeatureEmbeddingError, match="missing vocab sizes"):
        PhonemeFeatureEmbedding(model_config, sizes)


def test_non_divisible_boundary_dim_is_rejected():
    with pytest.raises(FeatureEmbeddingError, match="not divisible"):
        split_dim_evenly(21, 5, "boundary")


def test_split_dim_evenly_is_exact():
    assert split_dim_evenly(20, 5, "boundary") == (4, 4, 4, 4, 4)
    assert sum(split_dim_evenly(20, 5, "boundary")) == 20


# -- fusion ----------------------------------------------------------------


def test_fusion_projects_384_to_640(model_config):
    fusion = FeatureFusion(model_config)
    out = fusion(torch.randn(3, 7, 384))
    assert out.shape == (3, 7, 640)


def test_fusion_layer_order_is_linear_norm_activation_dropout(model_config):
    fusion = FeatureFusion(model_config)
    assert isinstance(fusion.projection, torch.nn.Linear)
    assert fusion.projection.in_features == 384
    assert fusion.projection.out_features == 640
    assert isinstance(fusion.norm, torch.nn.LayerNorm)
    assert isinstance(fusion.activation, torch.nn.GELU)
    assert isinstance(fusion.dropout, torch.nn.Dropout)
    assert fusion.dropout.p == pytest.approx(0.1)


def test_fusion_rejects_wrong_input_width(model_config):
    fusion = FeatureFusion(model_config)
    with pytest.raises(ValueError, match="expected last dim 384"):
        fusion(torch.randn(2, 3, 128))


def test_embedding_to_fusion_pipeline_shape(embedding, model_config, dataset):
    fusion = FeatureFusion(model_config)
    batch = PhonemeCollator()(list(dataset.encoded))
    out = fusion(embedding(batch.feature_ids))
    assert out.shape == (batch.batch_size, batch.max_length, 640)
