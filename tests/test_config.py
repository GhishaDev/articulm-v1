"""Config loading and fail-fast dimension validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from articulm.config import (
    BASELINE_FFN_SIZE,
    BASELINE_FUSED_INPUT_DIM,
    BASELINE_HEAD_DIM,
    BASELINE_HIDDEN_SIZE,
    BASELINE_NUM_HEADS,
    BASELINE_NUM_LAYERS,
    SOFT_VISEME_EMBEDDING_DIM,
    VISEME_CLASSES,
    ConfigError,
    load_data_config,
    load_model_config,
    load_train_config,
    to_plain_dict,
)


def test_baseline_model_config_matches_specification(model_config):
    transformer = model_config.transformer
    assert transformer.hidden_size == BASELINE_HIDDEN_SIZE == 640
    assert transformer.num_layers == BASELINE_NUM_LAYERS == 10
    assert transformer.num_heads == BASELINE_NUM_HEADS == 10
    assert transformer.head_dim == BASELINE_HEAD_DIM == 64
    assert transformer.ffn_size == BASELINE_FFN_SIZE == 2560
    assert transformer.norm == "pre_layer_norm"
    assert transformer.activation == "gelu"
    assert transformer.dropout == pytest.approx(0.1)
    assert transformer.attention_dropout == pytest.approx(0.1)


def test_embedding_dims_sum_to_384(model_config):
    assert model_config.input.embedding_dims.total == BASELINE_FUSED_INPUT_DIM == 384
    assert model_config.input.fused_input_dim == 384
    assert model_config.input.embedding_dims.articulatory.total == 60


def test_head_and_soft_embedding_dims(model_config):
    assert model_config.viseme_head.num_classes == VISEME_CLASSES == 18
    assert model_config.viseme_embedding.dim == SOFT_VISEME_EMBEDDING_DIM == 32
    assert model_config.viseme_embedding.mode == "soft_probability_weighted"
    assert model_config.strength_head.input_dim == 640 + 32 == 672
    assert model_config.strength_head.hidden_dims == (256, 64)
    assert model_config.strength_head.output_activation == "sigmoid"
    assert model_config.strength_head.output_scale == pytest.approx(100.0)


def test_baseline_local_conv_is_off(model_config):
    assert model_config.local_conv.enabled is False
    assert model_config.local_conv.kernel_size == 5
    assert model_config.local_conv.every_n_layers == 2
    assert model_config.local_conv.depthwise is True


def test_default_position_encoding_is_rope(model_config):
    assert model_config.position.type == "rope"


# -- fail-fast cases -------------------------------------------------------


def _write_model_yaml(tmp_path: Path, model_config, **overrides) -> Path:
    raw = to_plain_dict(dataclasses.replace(model_config, **overrides))
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({"model": raw}), encoding="utf-8")
    return path


def test_embedding_dim_mismatch_fails_fast(tmp_path, model_config):
    broken = dataclasses.replace(
        model_config.input, fused_input_dim=400
    )
    path = _write_model_yaml(tmp_path, model_config, input=broken)
    with pytest.raises(ConfigError, match="fused_input_dim"):
        load_model_config(path)


def test_heads_times_head_dim_must_equal_hidden(tmp_path, model_config):
    broken = dataclasses.replace(model_config.transformer, num_heads=8)
    path = _write_model_yaml(tmp_path, model_config, transformer=broken)
    with pytest.raises(ConfigError, match=r"num_heads \* head_dim"):
        load_model_config(path)


def test_fusion_output_must_equal_hidden_size(tmp_path, model_config):
    broken = dataclasses.replace(model_config.fusion, output_dim=512)
    path = _write_model_yaml(tmp_path, model_config, fusion=broken)
    with pytest.raises(ConfigError, match=r"fusion\.output_dim"):
        load_model_config(path)


def test_strength_head_input_dim_must_be_hidden_plus_soft_viseme(tmp_path, model_config):
    broken = dataclasses.replace(model_config.strength_head, input_dim=640)
    path = _write_model_yaml(tmp_path, model_config, strength_head=broken)
    with pytest.raises(ConfigError, match=r"strength_head\.input_dim"):
        load_model_config(path)


def test_hard_argmax_viseme_embedding_mode_is_rejected(tmp_path, model_config):
    broken = dataclasses.replace(model_config.viseme_embedding, mode="hard_argmax")
    path = _write_model_yaml(tmp_path, model_config, viseme_embedding=broken)
    with pytest.raises(ConfigError, match="soft_probability_weighted"):
        load_model_config(path)


def test_post_layer_norm_is_rejected(tmp_path, model_config):
    broken = dataclasses.replace(model_config.transformer, norm="post_layer_norm")
    path = _write_model_yaml(tmp_path, model_config, transformer=broken)
    with pytest.raises(ConfigError, match="pre_layer_norm"):
        load_model_config(path)


def test_wrong_viseme_class_count_is_rejected(tmp_path, model_config):
    broken = dataclasses.replace(model_config.viseme_head, num_classes=20)
    path = _write_model_yaml(tmp_path, model_config, viseme_head=broken)
    with pytest.raises(ConfigError, match="num_classes"):
        load_model_config(path)


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"name": "x", "mystery_option": 1}}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="unknown config keys"):
        load_model_config(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "nope.yaml")


# -- data config ----------------------------------------------------------


def test_data_config_conventions(data_config):
    assert data_config.schema_version == "articulm_v1_sample_v1"
    assert data_config.labels.viseme_classes == 18
    assert data_config.labels.strength_min == 0.0
    assert data_config.labels.strength_max == 100.0
    assert data_config.chinese.surface_tone_values == (1, 2, 3, 4, 5)
    assert data_config.chinese.stress_default == 0
    assert data_config.english.surface_tone_default == 0
    assert data_config.english.stress_values == (0, 1, 2)


def test_split_ratios_must_sum_to_one(tmp_path, data_config):
    broken = dataclasses.replace(data_config.split, test_ratio=0.20)
    raw = to_plain_dict(dataclasses.replace(data_config, split=broken))
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"data": raw}), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        load_data_config(path)


# -- train config ---------------------------------------------------------


def test_baseline_train_config_loads(config_dir):
    config = load_train_config(config_dir / "train_v1_50m.yaml")
    assert config.training.stage == "synthetic_pretraining"
    assert config.training.optimizer.type == "adamw"
    assert config.training.optimizer.learning_rate == pytest.approx(3.0e-4)
    assert config.training.optimizer.weight_decay == pytest.approx(0.01)
    assert config.training.scheduler.warmup_ratio == pytest.approx(0.05)
    assert config.training.gradient_clip_norm == pytest.approx(1.0)
    assert config.training.batching.strategy == "dynamic_phoneme_tokens"
    assert config.training.batching.max_phoneme_tokens_per_batch == 6000
    assert config.training.batching.gradient_accumulation_steps == 2
    # Synthetic baseline loss weights from docs/05.
    assert config.training.loss.viseme.weight == pytest.approx(1.0)
    assert config.training.loss.viseme.label_smoothing == pytest.approx(0.05)
    assert config.training.loss.strength.weight == pytest.approx(0.3)
    assert config.training.loss.strength.normalize_target_to_0_1 is True


def test_tiny_overfit_config_loads(config_dir):
    config = load_train_config(config_dir / "train_tiny_overfit.yaml")
    assert config.training.stage == "tiny_overfit"
    assert config.training.tiny_subset.num_samples == 64
    assert config.training.max_steps == 2000
    assert config.training.batching.strategy == "fixed_samples"


def test_unnormalised_strength_target_is_rejected(tmp_path, config_dir):
    raw = yaml.safe_load((config_dir / "train_v1_50m.yaml").read_text(encoding="utf-8"))
    raw["training"]["loss"]["strength"]["normalize_target_to_0_1"] = False
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="normalize_target_to_0_1"):
        load_train_config(path)


def test_non_adamw_optimizer_is_rejected(tmp_path, config_dir):
    raw = yaml.safe_load((config_dir / "train_v1_50m.yaml").read_text(encoding="utf-8"))
    raw["training"]["optimizer"]["type"] = "sgd"
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="adamw"):
        load_train_config(path)


def test_run_without_step_or_epoch_budget_is_rejected(tmp_path, config_dir):
    raw = yaml.safe_load((config_dir / "train_v1_50m.yaml").read_text(encoding="utf-8"))
    raw["training"].pop("max_steps", None)
    raw["training"].pop("max_epochs", None)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_steps / max_epochs"):
        load_train_config(path)
