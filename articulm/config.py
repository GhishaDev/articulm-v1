"""YAML configuration loading and fail-fast validation for ArticuLM-V1.

Every tunable value comes from YAML. Dimension mismatches raise
:class:`ConfigError` immediately instead of being silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# Baseline constants fixed by docs/01_model_architecture.md. Changing these
# requires an explicit config override AND an experiment note; the loader
# refuses silent drift by validating internal consistency instead.
BASELINE_HIDDEN_SIZE = 640
BASELINE_NUM_LAYERS = 10
BASELINE_NUM_HEADS = 10
BASELINE_HEAD_DIM = 64
BASELINE_FFN_SIZE = 2560
BASELINE_FUSED_INPUT_DIM = 384
VISEME_CLASSES = 18
SOFT_VISEME_EMBEDDING_DIM = 32

# Fields that must never reach the encoder (docs/09_acceptance_criteria.md).
# The parser structurally reads only the documented encoder keys, so these
# sets exist to fail fast on malformed data rather than to filter it.
#
# LEAKAGE: target or teacher-signal fields. Their presence at token feature
# level is a data-generation bug and is rejected outright.
LEAKAGE_ENCODER_FIELDS = frozenset(
    {
        "viseme_id",
        "strength",
        "shapeV2",
        "shapev2",
        "Talk",
        "talk",
        "raw_value",
    }
)
# METADATA: documented as retainable teacher metadata (docs/02). Never an
# encoder input in V1, but tolerated in the file and simply not read.
METADATA_ONLY_FIELDS = frozenset({"duration", "timing"})

FORBIDDEN_ENCODER_FIELDS = LEAKAGE_ENCODER_FIELDS | METADATA_ONLY_FIELDS


class ConfigError(ValueError):
    """Raised when a configuration is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------
# Model config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArticulatoryEmbeddingDims:
    type: int = 8
    height: int = 8
    backness: int = 8
    rounded: int = 4
    place: int = 12
    manner: int = 12
    voiced: int = 4
    aspirated: int = 4

    @property
    def total(self) -> int:
        return sum(getattr(self, f.name) for f in fields(self))

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class EmbeddingDims:
    phoneme: int = 256
    language: int = 8
    surface_tone: int = 16
    stress: int = 8
    syllable_role: int = 16
    articulatory: ArticulatoryEmbeddingDims = field(default_factory=ArticulatoryEmbeddingDims)
    boundary: int = 20

    @property
    def total(self) -> int:
        return (
            self.phoneme
            + self.language
            + self.surface_tone
            + self.stress
            + self.syllable_role
            + self.articulatory.total
            + self.boundary
        )


@dataclass(frozen=True)
class InputConfig:
    max_seq_len: int = 256
    embedding_dims: EmbeddingDims = field(default_factory=EmbeddingDims)
    fused_input_dim: int = BASELINE_FUSED_INPUT_DIM


@dataclass(frozen=True)
class FusionConfig:
    output_dim: int = BASELINE_HIDDEN_SIZE
    layer_norm: bool = True
    activation: str = "gelu"
    dropout: float = 0.1


@dataclass(frozen=True)
class PositionConfig:
    type: str = "rope"


@dataclass(frozen=True)
class TransformerConfig:
    hidden_size: int = BASELINE_HIDDEN_SIZE
    num_layers: int = BASELINE_NUM_LAYERS
    num_heads: int = BASELINE_NUM_HEADS
    head_dim: int = BASELINE_HEAD_DIM
    ffn_size: int = BASELINE_FFN_SIZE
    activation: str = "gelu"
    norm: str = "pre_layer_norm"
    dropout: float = 0.1
    attention_dropout: float = 0.1


@dataclass(frozen=True)
class LocalConvConfig:
    enabled: bool = False
    every_n_layers: int = 2
    kernel_size: int = 5
    depthwise: bool = True


@dataclass(frozen=True)
class VisemeHeadConfig:
    hidden_size: int = 256
    num_classes: int = VISEME_CLASSES
    dropout: float = 0.1


@dataclass(frozen=True)
class VisemeEmbeddingConfig:
    num_embeddings: int = VISEME_CLASSES
    dim: int = SOFT_VISEME_EMBEDDING_DIM
    mode: str = "soft_probability_weighted"


@dataclass(frozen=True)
class StrengthHeadConfig:
    input_dim: int = BASELINE_HIDDEN_SIZE + SOFT_VISEME_EMBEDDING_DIM
    hidden_dims: tuple[int, ...] = (256, 64)
    dropout: float = 0.1
    activation: str = "gelu"
    output_activation: str = "sigmoid"
    output_scale: float = 100.0


@dataclass(frozen=True)
class ModelConfig:
    name: str = "ArticuLM-V1-50M"
    input: InputConfig = field(default_factory=InputConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    position: PositionConfig = field(default_factory=PositionConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    local_conv: LocalConvConfig = field(default_factory=LocalConvConfig)
    viseme_head: VisemeHeadConfig = field(default_factory=VisemeHeadConfig)
    viseme_embedding: VisemeEmbeddingConfig = field(default_factory=VisemeEmbeddingConfig)
    strength_head: StrengthHeadConfig = field(default_factory=StrengthHeadConfig)

    def validate(self) -> None:
        _validate_model_config(self)


# --------------------------------------------------------------------------
# Data config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PaddingConfig:
    phoneme_pad_token: str = "[PAD]"
    phoneme_unk_token: str = "[UNK]"


@dataclass(frozen=True)
class LanguageConfig:
    supported: tuple[str, ...] = ("zh", "en")


@dataclass(frozen=True)
class ChineseConfig:
    surface_tone_values: tuple[int, ...] = (1, 2, 3, 4, 5)
    stress_default: int = 0


@dataclass(frozen=True)
class EnglishConfig:
    surface_tone_default: int = 0
    stress_values: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class LabelConfig:
    viseme_classes: int = VISEME_CLASSES
    strength_min: float = 0.0
    strength_max: float = 100.0


@dataclass(frozen=True)
class DataValidationConfig:
    reject_nan_inf: bool = True
    reject_invalid_viseme: bool = True
    reject_invalid_strength: bool = True
    fail_on_schema_mismatch: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.90
    validation_ratio: float = 0.05
    test_ratio: float = 0.05
    seed: int = 42
    prevent_near_duplicate_leakage: bool = True
    # Near-duplicate detection (see articulm/data/split.py). Sentences whose
    # phoneme-shingle Jaccard reaches the threshold are grouped and always land
    # in the same split.
    # Deliberately below the usual "looks like a copy" bar. For leakage control
    # the error costs are asymmetric: a false grouping only nudges split ratios,
    # while a missed near-duplicate inflates held-out metrics and, per docs/09,
    # invalidates the run. Recall is worth more than precision here.
    near_duplicate_jaccard_threshold: float = 0.8
    near_duplicate_shingle_size: int = 4
    near_duplicate_sketch_size: int = 8
    # A sketch hash shared by very many sentences is uninformative and would
    # cost O(n^2) inside that bucket. Oversized buckets are skipped and
    # reported, never silently truncated. Raise it for small corpora where
    # exhaustive comparison is affordable.
    near_duplicate_max_bucket_size: int = 512


@dataclass(frozen=True)
class DataConfig:
    schema_version: str = "articulm_v1_sample_v1"
    train_path: str = "data/train.jsonl"
    validation_path: str | None = "data/validation.jsonl"
    test_path: str | None = "data/test.jsonl"
    max_seq_len: int = 256
    padding: PaddingConfig = field(default_factory=PaddingConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    chinese: ChineseConfig = field(default_factory=ChineseConfig)
    english: EnglishConfig = field(default_factory=EnglishConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    # Persist eagerly-encoded tensors next to each split (train.enc-cache.pt)
    # so restarts skip the ~20 min pure-Python encode. Unwritable locations
    # degrade silently to encoding in memory.
    encoded_cache: bool = True

    def validate(self) -> None:
        _validate_data_config(self)


# --------------------------------------------------------------------------
# Training config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "articulm_v1_50m_baseline"
    seed: int = 42
    output_dir: str = "runs/articulm_v1_50m_baseline"


@dataclass(frozen=True)
class OptimizerConfig:
    type: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    # Optional split LR for Human Gold fine-tuning (docs/04_training_plan.md).
    head_learning_rate: float | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    type: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.0


@dataclass(frozen=True)
class BatchingConfig:
    strategy: str = "fixed_samples"
    batch_size: int = 8
    max_phoneme_tokens_per_batch: int = 6000
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    shuffle: bool = True


@dataclass(frozen=True)
class VisemeLossConfig:
    type: str = "cross_entropy"
    weight: float = 1.0
    label_smoothing: float = 0.05


@dataclass(frozen=True)
class StrengthLossConfig:
    type: str = "smooth_l1"
    weight: float = 0.3
    normalize_target_to_0_1: bool = True
    beta: float = 0.1
    # Per-source weight multipliers. `pseudo_strength_v1` is a programmatic
    # prior, never Human Gold (docs/03_training_data_spec.md).
    source_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LossConfig:
    viseme: VisemeLossConfig = field(default_factory=VisemeLossConfig)
    strength: StrengthLossConfig = field(default_factory=StrengthLossConfig)


@dataclass(frozen=True)
class EvaluationConfig:
    every_steps: int = 1000


@dataclass(frozen=True)
class CheckpointConfig:
    every_steps: int = 1000
    save_last: bool = True
    save_best_by: str = "val_viseme_macro_f1"
    keep_last_n: int = 3
    higher_is_better: bool = True
    # Weight of the strength term in the "val_composite" selection metric:
    #   val_composite = val_viseme_macro_f1 - alpha * val_strength_mae / 100
    # macro F1 saturates near its noise ceiling on synthetic labels, where it
    # cannot separate candidates any more; the composite keeps viseme primary
    # while strength quality still breaks ties.
    best_composite_alpha: float = 1.0


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience_evaluations: int = 5


@dataclass(frozen=True)
class LoggingConfig:
    every_steps: int = 50


@dataclass(frozen=True)
class TinySubsetConfig:
    num_samples: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    stage: str = "synthetic_pretraining"
    precision: str = "auto"
    max_steps: int | None = None
    max_epochs: int | None = None
    tiny_subset: TinySubsetConfig = field(default_factory=TinySubsetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    gradient_clip_norm: float = 1.0
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass(frozen=True)
class TrainRunConfig:
    """Top-level training config: experiment + model + data + training."""

    experiment: ExperimentConfig
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    model_config_path: str | None = None
    data_config_path: str | None = None

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        _validate_training_config(self.training, self.model, self.data)


# --------------------------------------------------------------------------
# Generic YAML -> dataclass construction
# --------------------------------------------------------------------------


def _unwrap_optional(annotation: Any) -> Any:
    """Return the non-None type of `X | None`, else the annotation itself."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is None:
        return annotation
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _resolve_annotation(cls: type, name: str) -> Any:
    import typing

    hints = typing.get_type_hints(cls)
    return hints.get(name)


def _build(cls: type, raw: Any, path: str) -> Any:
    """Recursively build a frozen dataclass from a plain YAML mapping."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{path}: unknown config keys {sorted(unknown)}; known keys {sorted(known)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        annotation = _unwrap_optional(_resolve_annotation(cls, f.name))
        child_path = f"{path}.{f.name}"
        if is_dataclass(annotation):
            kwargs[f.name] = _build(annotation, value, child_path)
        elif annotation is not None and getattr(annotation, "__origin__", None) is tuple:
            if not isinstance(value, (list, tuple)):
                raise ConfigError(f"{child_path}: expected a list")
            kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"config file is empty: {p}")
    if not isinstance(raw, dict):
        raise ConfigError(f"config file must contain a mapping at top level: {p}")
    return raw


def load_model_config(path: str | Path) -> ModelConfig:
    raw = load_yaml(path)
    if "model" not in raw:
        raise ConfigError(f"{path}: missing top-level 'model' section")
    cfg = _build(ModelConfig, raw["model"], "model")
    cfg.validate()
    return cfg


def load_data_config(path: str | Path) -> DataConfig:
    raw = load_yaml(path)
    if "data" not in raw:
        raise ConfigError(f"{path}: missing top-level 'data' section")
    cfg = _build(DataConfig, raw["data"], "data")
    cfg.validate()
    return cfg


def load_train_config(path: str | Path) -> TrainRunConfig:
    """Load a training config and the model/data configs it references.

    Referenced paths are resolved relative to the current working directory
    first, then relative to the training config's directory.
    """
    raw = load_yaml(path)
    train_path = Path(path)

    for required in ("experiment", "training"):
        if required not in raw:
            raise ConfigError(f"{path}: missing top-level '{required}' section")

    model_ref = raw.get("model_config")
    data_ref = raw.get("data_config")
    if not model_ref:
        raise ConfigError(f"{path}: missing 'model_config' path")
    if not data_ref:
        raise ConfigError(f"{path}: missing 'data_config' path")

    model_path = _resolve_ref(model_ref, train_path)
    data_path = _resolve_ref(data_ref, train_path)

    cfg = TrainRunConfig(
        experiment=_build(ExperimentConfig, raw["experiment"], "experiment"),
        training=_build(TrainingConfig, raw["training"], "training"),
        model=load_model_config(model_path),
        data=load_data_config(data_path),
        model_config_path=str(model_path),
        data_config_path=str(data_path),
    )
    cfg.validate()
    return cfg


def _resolve_ref(ref: str, base_config: Path) -> Path:
    direct = Path(ref)
    if direct.is_file():
        return direct
    sibling = base_config.parent.parent / ref
    if sibling.is_file():
        return sibling
    sibling2 = base_config.parent / ref
    if sibling2.is_file():
        return sibling2
    raise ConfigError(f"referenced config not found: {ref}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_model_config(cfg: ModelConfig) -> None:
    dims = cfg.input.embedding_dims
    tf = cfg.transformer

    for name, value in [
        ("input.max_seq_len", cfg.input.max_seq_len),
        ("transformer.hidden_size", tf.hidden_size),
        ("transformer.num_layers", tf.num_layers),
        ("transformer.num_heads", tf.num_heads),
        ("transformer.head_dim", tf.head_dim),
        ("transformer.ffn_size", tf.ffn_size),
        ("viseme_head.hidden_size", cfg.viseme_head.hidden_size),
        ("viseme_embedding.dim", cfg.viseme_embedding.dim),
    ]:
        if value <= 0:
            raise ConfigError(f"model.{name} must be positive, got {value}")

    for name, value in [
        ("fusion.dropout", cfg.fusion.dropout),
        ("transformer.dropout", tf.dropout),
        ("transformer.attention_dropout", tf.attention_dropout),
        ("viseme_head.dropout", cfg.viseme_head.dropout),
        ("strength_head.dropout", cfg.strength_head.dropout),
    ]:
        if not 0.0 <= value < 1.0:
            raise ConfigError(f"model.{name} must be in [0,1), got {value}")

    if dims.total != cfg.input.fused_input_dim:
        raise ConfigError(
            "embedding dimensions must sum to model.input.fused_input_dim: "
            f"sum={dims.total} != fused_input_dim={cfg.input.fused_input_dim}. "
            f"components: phoneme={dims.phoneme} language={dims.language} "
            f"surface_tone={dims.surface_tone} stress={dims.stress} "
            f"syllable_role={dims.syllable_role} articulatory={dims.articulatory.total} "
            f"boundary={dims.boundary}"
        )

    if cfg.fusion.output_dim != tf.hidden_size:
        raise ConfigError(
            f"model.fusion.output_dim ({cfg.fusion.output_dim}) must equal "
            f"model.transformer.hidden_size ({tf.hidden_size})"
        )

    if tf.num_heads * tf.head_dim != tf.hidden_size:
        raise ConfigError(
            f"num_heads * head_dim must equal hidden_size: "
            f"{tf.num_heads} * {tf.head_dim} = {tf.num_heads * tf.head_dim} "
            f"!= {tf.hidden_size}"
        )

    if cfg.position.type not in {"rope", "learned", "sinusoidal", "none"}:
        raise ConfigError(
            f"model.position.type must be one of rope/learned/sinusoidal/none, "
            f"got {cfg.position.type!r}"
        )

    if cfg.position.type == "rope" and tf.head_dim % 2 != 0:
        raise ConfigError(f"RoPE requires an even head_dim, got {tf.head_dim}")

    if tf.norm != "pre_layer_norm":
        raise ConfigError(
            f"model.transformer.norm must be 'pre_layer_norm' for the V1 baseline, got {tf.norm!r}"
        )

    for name, value in [
        ("fusion.activation", cfg.fusion.activation),
        ("transformer.activation", tf.activation),
        ("strength_head.activation", cfg.strength_head.activation),
    ]:
        if value not in {"gelu", "relu", "silu"}:
            raise ConfigError(f"model.{name} must be gelu/relu/silu, got {value!r}")

    if cfg.viseme_head.num_classes != VISEME_CLASSES:
        raise ConfigError(
            f"model.viseme_head.num_classes must be {VISEME_CLASSES}, "
            f"got {cfg.viseme_head.num_classes}"
        )

    if cfg.viseme_embedding.num_embeddings != cfg.viseme_head.num_classes:
        raise ConfigError(
            "model.viseme_embedding.num_embeddings must equal viseme_head.num_classes: "
            f"{cfg.viseme_embedding.num_embeddings} != {cfg.viseme_head.num_classes}"
        )

    if cfg.viseme_embedding.mode != "soft_probability_weighted":
        raise ConfigError(
            "model.viseme_embedding.mode must be 'soft_probability_weighted'; hard argmax "
            f"before the Strength Head is forbidden during training, got "
            f"{cfg.viseme_embedding.mode!r}"
        )

    expected_strength_input = tf.hidden_size + cfg.viseme_embedding.dim
    if cfg.strength_head.input_dim != expected_strength_input:
        raise ConfigError(
            "model.strength_head.input_dim must equal transformer.hidden_size + "
            f"viseme_embedding.dim: {cfg.strength_head.input_dim} != "
            f"{tf.hidden_size} + {cfg.viseme_embedding.dim} = {expected_strength_input}"
        )

    if not cfg.strength_head.hidden_dims:
        raise ConfigError("model.strength_head.hidden_dims must be non-empty")
    if any(d <= 0 for d in cfg.strength_head.hidden_dims):
        raise ConfigError(
            f"model.strength_head.hidden_dims must be positive, got {cfg.strength_head.hidden_dims}"
        )

    if cfg.strength_head.output_activation != "sigmoid":
        raise ConfigError(
            "model.strength_head.output_activation must be 'sigmoid', got "
            f"{cfg.strength_head.output_activation!r}"
        )
    if cfg.strength_head.output_scale <= 0:
        raise ConfigError(
            f"model.strength_head.output_scale must be positive, got {cfg.strength_head.output_scale}"
        )

    lc = cfg.local_conv
    if lc.kernel_size < 1 or lc.kernel_size % 2 == 0:
        raise ConfigError(f"model.local_conv.kernel_size must be odd and >=1, got {lc.kernel_size}")
    if lc.every_n_layers < 1:
        raise ConfigError(
            f"model.local_conv.every_n_layers must be >=1, got {lc.every_n_layers}"
        )


def _validate_data_config(cfg: DataConfig) -> None:
    if cfg.max_seq_len <= 0:
        raise ConfigError(f"data.max_seq_len must be positive, got {cfg.max_seq_len}")
    if cfg.labels.viseme_classes != VISEME_CLASSES:
        raise ConfigError(
            f"data.labels.viseme_classes must be {VISEME_CLASSES}, got {cfg.labels.viseme_classes}"
        )
    if cfg.labels.strength_min >= cfg.labels.strength_max:
        raise ConfigError(
            "data.labels.strength_min must be < strength_max: "
            f"{cfg.labels.strength_min} >= {cfg.labels.strength_max}"
        )
    if not cfg.language.supported:
        raise ConfigError("data.language.supported must be non-empty")
    if cfg.padding.phoneme_pad_token == cfg.padding.phoneme_unk_token:
        raise ConfigError("data.padding pad and unk tokens must differ")
    if not cfg.chinese.surface_tone_values:
        raise ConfigError("data.chinese.surface_tone_values must be non-empty")
    if not cfg.english.stress_values:
        raise ConfigError("data.english.stress_values must be non-empty")

    ratios = (cfg.split.train_ratio, cfg.split.validation_ratio, cfg.split.test_ratio)
    if any(r < 0 for r in ratios):
        raise ConfigError(f"data.split ratios must be non-negative, got {ratios}")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"data.split ratios must sum to 1.0, got {total}")

    if not 0.0 < cfg.split.near_duplicate_jaccard_threshold <= 1.0:
        raise ConfigError(
            "data.split.near_duplicate_jaccard_threshold must be in (0,1], got "
            f"{cfg.split.near_duplicate_jaccard_threshold}"
        )
    if cfg.split.near_duplicate_shingle_size < 1:
        raise ConfigError("data.split.near_duplicate_shingle_size must be >= 1")
    if cfg.split.near_duplicate_sketch_size < 1:
        raise ConfigError("data.split.near_duplicate_sketch_size must be >= 1")
    if cfg.split.near_duplicate_max_bucket_size < 2:
        raise ConfigError(
            "data.split.near_duplicate_max_bucket_size must be >= 2 for any pair "
            f"to be comparable, got {cfg.split.near_duplicate_max_bucket_size}"
        )


def _validate_training_config(
    cfg: TrainingConfig, model: ModelConfig, data: DataConfig
) -> None:
    if cfg.optimizer.type.lower() != "adamw":
        raise ConfigError(
            f"training.optimizer.type must be 'adamw' for V1, got {cfg.optimizer.type!r}"
        )
    if cfg.optimizer.learning_rate <= 0:
        raise ConfigError(
            f"training.optimizer.learning_rate must be positive, got {cfg.optimizer.learning_rate}"
        )
    if cfg.optimizer.weight_decay < 0:
        raise ConfigError("training.optimizer.weight_decay must be non-negative")
    if len(cfg.optimizer.betas) != 2:
        raise ConfigError(f"training.optimizer.betas must have 2 values, got {cfg.optimizer.betas}")

    if cfg.scheduler.type not in {"cosine", "linear", "constant"}:
        raise ConfigError(
            f"training.scheduler.type must be cosine/linear/constant, got {cfg.scheduler.type!r}"
        )
    if not 0.0 <= cfg.scheduler.warmup_ratio < 1.0:
        raise ConfigError(
            f"training.scheduler.warmup_ratio must be in [0,1), got {cfg.scheduler.warmup_ratio}"
        )

    if cfg.batching.strategy not in {"fixed_samples", "dynamic_phoneme_tokens"}:
        raise ConfigError(
            "training.batching.strategy must be fixed_samples or dynamic_phoneme_tokens, "
            f"got {cfg.batching.strategy!r}"
        )
    if cfg.batching.strategy == "fixed_samples" and cfg.batching.batch_size <= 0:
        raise ConfigError("training.batching.batch_size must be positive for fixed_samples")
    if (
        cfg.batching.strategy == "dynamic_phoneme_tokens"
        and cfg.batching.max_phoneme_tokens_per_batch < data.max_seq_len
    ):
        raise ConfigError(
            "training.batching.max_phoneme_tokens_per_batch "
            f"({cfg.batching.max_phoneme_tokens_per_batch}) must be at least "
            f"data.max_seq_len ({data.max_seq_len}) so a longest sequence still fits"
        )
    if cfg.batching.gradient_accumulation_steps < 1:
        raise ConfigError("training.batching.gradient_accumulation_steps must be >=1")

    if cfg.gradient_clip_norm < 0:
        raise ConfigError("training.gradient_clip_norm must be non-negative")

    if cfg.precision not in {"auto", "fp32", "fp16", "bf16"}:
        raise ConfigError(
            f"training.precision must be auto/fp32/fp16/bf16, got {cfg.precision!r}"
        )

    if cfg.loss.viseme.type != "cross_entropy":
        raise ConfigError(
            f"training.loss.viseme.type must be cross_entropy, got {cfg.loss.viseme.type!r}"
        )
    if not 0.0 <= cfg.loss.viseme.label_smoothing < 1.0:
        raise ConfigError(
            f"training.loss.viseme.label_smoothing must be in [0,1), "
            f"got {cfg.loss.viseme.label_smoothing}"
        )
    if cfg.loss.strength.type not in {"smooth_l1", "huber", "l1", "mse"}:
        raise ConfigError(
            "training.loss.strength.type must be smooth_l1/huber/l1/mse, "
            f"got {cfg.loss.strength.type!r}"
        )
    if cfg.loss.viseme.weight < 0 or cfg.loss.strength.weight < 0:
        raise ConfigError("loss weights must be non-negative")
    if cfg.loss.viseme.weight == 0 and cfg.loss.strength.weight == 0:
        raise ConfigError("at least one loss weight must be non-zero")
    if not cfg.loss.strength.normalize_target_to_0_1:
        raise ConfigError(
            "training.loss.strength.normalize_target_to_0_1 must be true: the Strength Head "
            "emits sigmoid in [0,1] and the target must be strength/100"
        )
    for source, weight in cfg.loss.strength.source_weights.items():
        if weight < 0:
            raise ConfigError(
                f"training.loss.strength.source_weights[{source!r}] must be non-negative"
            )

    if cfg.max_steps is not None and cfg.max_steps <= 0:
        raise ConfigError(f"training.max_steps must be positive, got {cfg.max_steps}")
    if cfg.max_epochs is not None and cfg.max_epochs <= 0:
        raise ConfigError(f"training.max_epochs must be positive, got {cfg.max_epochs}")
    if cfg.max_steps is None and cfg.max_epochs is None:
        raise ConfigError("training must set at least one of max_steps / max_epochs")
    if cfg.tiny_subset.num_samples is not None and cfg.tiny_subset.num_samples <= 0:
        raise ConfigError("training.tiny_subset.num_samples must be positive when set")

    if cfg.checkpoint.keep_last_n < 1:
        raise ConfigError("training.checkpoint.keep_last_n must be >=1")
    if cfg.evaluation.every_steps <= 0:
        raise ConfigError("training.evaluation.every_steps must be positive")
    if cfg.logging.every_steps <= 0:
        raise ConfigError("training.logging.every_steps must be positive")

    if data.max_seq_len > model.input.max_seq_len:
        raise ConfigError(
            f"data.max_seq_len ({data.max_seq_len}) exceeds "
            f"model.input.max_seq_len ({model.input.max_seq_len})"
        )


# --------------------------------------------------------------------------
# Serialisation for checkpoints / run directories
# --------------------------------------------------------------------------


def to_plain_dict(obj: Any) -> Any:
    """Convert nested dataclasses/tuples into YAML/JSON-safe plain data."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [to_plain_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    return obj


def dump_yaml(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_plain_dict(obj), fh, sort_keys=False, allow_unicode=True)


def model_config_from_dict(raw: dict[str, Any]) -> ModelConfig:
    """Rebuild a ModelConfig from a plain dict (checkpoint reload path)."""
    cfg = _build(ModelConfig, raw, "model")
    cfg.validate()
    return cfg


def data_config_from_dict(raw: dict[str, Any]) -> DataConfig:
    cfg = _build(DataConfig, raw, "data")
    cfg.validate()
    return cfg
