"""Training-sample schema, parsing and fail-fast validation.

The training unit is one sentence / utterance = one phoneme sequence.
Splitting a sentence into independent per-phoneme rows is rejected upstream
by construction: a sample always carries a `tokens` list.

Teacher metadata (``shapeV2`` / ``Talk`` / ``raw_value`` / ``timing`` /
``duration``) may be present in the file but is never parsed into encoder
inputs, and must never appear as a token-level feature.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import LEAKAGE_ENCODER_FIELDS, DataConfig

NA = "[NA]"

SYLLABLE_ROLES = ("onset", "nucleus", "coda", "other", "silence")
BOUNDARY_TYPES = ("none", "minor", "major")
ARTICULATORY_FIELDS = (
    "type",
    "height",
    "backness",
    "rounded",
    "place",
    "manner",
    "voiced",
    "aspirated",
)
BOUNDARY_FIELDS = (
    "word_start",
    "word_end",
    "phrase_start",
    "phrase_end",
    "boundary_type",
)
# Token keys the encoder is allowed to read.
ENCODER_TOKEN_KEYS = frozenset(
    {
        "phoneme",
        "language",
        "surface_tone",
        "stress",
        "syllable_role",
        "articulatory",
        "boundary",
    }
)

PSEUDO_STRENGTH_SOURCES = frozenset({"pseudo_strength_v1", "pseudo", "rule", "programmatic"})
HUMAN_GOLD_STRENGTH_SOURCES = frozenset({"human", "human_gold"})


class SchemaError(ValueError):
    """Raised when a sample violates the ArticuLM-V1 training schema."""


@dataclass(frozen=True)
class ArticulatoryFeatures:
    """Categorical articulatory description. ``None`` becomes ``[NA]``."""

    type: str = NA
    height: str = NA
    backness: str = NA
    rounded: str = NA
    place: str = NA
    manner: str = NA
    voiced: str = NA
    aspirated: str = NA

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in ARTICULATORY_FIELDS}


@dataclass(frozen=True)
class BoundaryFeatures:
    word_start: str = "false"
    word_end: str = "false"
    phrase_start: str = "false"
    phrase_end: str = "false"
    boundary_type: str = "none"

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in BOUNDARY_FIELDS}


@dataclass(frozen=True)
class TokenLabels:
    viseme_id: int
    strength: float
    viseme_source: str = "unknown"
    strength_source: str = "unknown"

    @property
    def is_human_gold_strength(self) -> bool:
        return self.strength_source.lower() in HUMAN_GOLD_STRENGTH_SOURCES


@dataclass(frozen=True)
class PhonemeToken:
    """One phoneme-aligned encoder input plus its optional supervision."""

    phoneme: str
    language: str
    surface_tone: int
    stress: int
    syllable_role: str
    articulatory: ArticulatoryFeatures
    boundary: BoundaryFeatures
    labels: TokenLabels | None = None


@dataclass(frozen=True)
class Sample:
    sample_id: str
    tokens: tuple[PhonemeToken, ...]
    text: str = ""
    schema_version: str = ""
    normalized_text: str | None = None
    # Kept for traceability only; never fed to the encoder.
    teacher_metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def has_labels(self) -> bool:
        return all(t.labels is not None for t in self.tokens)


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


def _categorical(value: Any, where: str) -> str:
    """Normalise a nullable categorical value into a vocabulary token."""
    if value is None:
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise SchemaError(f"{where}: non-finite categorical value {value!r}")
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return NA
        return text.lower()
    raise SchemaError(f"{where}: unsupported categorical type {type(value).__name__}")


def _boolean_token(value: Any, where: str) -> str:
    if value is None:
        return "false"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower()
    raise SchemaError(f"{where}: expected a boolean, got {value!r}")


def _require_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise SchemaError(f"{where}: expected an int, got a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaError(f"{where}: non-finite value {value!r}")
        if float(value).is_integer():
            return int(value)
        raise SchemaError(f"{where}: expected an integer, got {value!r}")
    raise SchemaError(f"{where}: expected an int, got {type(value).__name__}")


def _require_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{where}: expected a float, got a bool")
    if isinstance(value, (int, float)):
        out = float(value)
        if not math.isfinite(out):
            raise SchemaError(f"{where}: NaN/Inf is not allowed, got {value!r}")
        return out
    raise SchemaError(f"{where}: expected a float, got {type(value).__name__}")


def _reject_leaked_keys(mapping: dict[str, Any], where: str) -> None:
    """Reject target/teacher-signal fields sitting at encoder-feature level."""
    leaked = sorted(set(mapping) & LEAKAGE_ENCODER_FIELDS)
    if leaked:
        raise SchemaError(
            f"{where}: target/teacher fields {leaked} must not appear as encoder features"
        )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def parse_articulatory(raw: Any, where: str) -> ArticulatoryFeatures:
    if raw is None:
        return ArticulatoryFeatures()
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: 'articulatory' must be a mapping")
    _reject_leaked_keys(raw, f"{where}.articulatory")
    unknown = sorted(set(raw) - set(ARTICULATORY_FIELDS))
    if unknown:
        raise SchemaError(f"{where}.articulatory: unknown fields {unknown}")
    return ArticulatoryFeatures(
        **{name: _categorical(raw.get(name), f"{where}.articulatory.{name}") for name in ARTICULATORY_FIELDS}
    )


def parse_boundary(raw: Any, where: str) -> BoundaryFeatures:
    if raw is None:
        return BoundaryFeatures()
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: 'boundary' must be a mapping")
    _reject_leaked_keys(raw, f"{where}.boundary")
    unknown = sorted(set(raw) - set(BOUNDARY_FIELDS))
    if unknown:
        raise SchemaError(f"{where}.boundary: unknown fields {unknown}")
    boundary_type = _categorical(raw.get("boundary_type", "none"), f"{where}.boundary.boundary_type")
    return BoundaryFeatures(
        word_start=_boolean_token(raw.get("word_start"), f"{where}.boundary.word_start"),
        word_end=_boolean_token(raw.get("word_end"), f"{where}.boundary.word_end"),
        phrase_start=_boolean_token(raw.get("phrase_start"), f"{where}.boundary.phrase_start"),
        phrase_end=_boolean_token(raw.get("phrase_end"), f"{where}.boundary.phrase_end"),
        boundary_type=boundary_type if boundary_type != NA else "none",
    )


def parse_labels(raw: Any, where: str, cfg: DataConfig) -> TokenLabels:
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: 'labels' must be a mapping")

    if "viseme_id" not in raw:
        raise SchemaError(f"{where}.labels: missing 'viseme_id'")
    if "strength" not in raw:
        raise SchemaError(f"{where}.labels: missing 'strength'")

    viseme_id = _require_int(raw["viseme_id"], f"{where}.labels.viseme_id")
    if cfg.validation.reject_invalid_viseme and not 0 <= viseme_id < cfg.labels.viseme_classes:
        raise SchemaError(
            f"{where}.labels.viseme_id must be in [0,{cfg.labels.viseme_classes - 1}], got {viseme_id}"
        )

    strength = _require_float(raw["strength"], f"{where}.labels.strength")
    if cfg.validation.reject_invalid_strength and not (
        cfg.labels.strength_min <= strength <= cfg.labels.strength_max
    ):
        raise SchemaError(
            f"{where}.labels.strength must be in "
            f"[{cfg.labels.strength_min},{cfg.labels.strength_max}], got {strength}"
        )

    strength_source = str(raw.get("strength_source", "unknown"))
    raw_value = raw.get("raw_value")
    if strength_source.lower() in HUMAN_GOLD_STRENGTH_SOURCES and raw_value is not None:
        raise SchemaError(
            f"{where}.labels: strength_source='{strength_source}' carries a teacher 'raw_value'; "
            "a programmatic raw value must not be relabelled as Human Gold"
        )

    return TokenLabels(
        viseme_id=viseme_id,
        strength=strength,
        viseme_source=str(raw.get("viseme_source", "unknown")),
        strength_source=strength_source,
    )


def parse_token(
    raw: Any, where: str, cfg: DataConfig, *, require_labels: bool
) -> PhonemeToken:
    if not isinstance(raw, dict):
        raise SchemaError(f"{where}: token must be a mapping")

    # Encoder-feature leakage check: forbidden keys are only tolerated inside
    # `labels` / `teacher_metadata`, never at token feature level.
    feature_keys = {k for k in raw if k not in {"labels", "teacher_metadata"}}
    _reject_leaked_keys({k: raw[k] for k in feature_keys}, where)
    if "features" in raw and isinstance(raw["features"], dict):
        _reject_leaked_keys(raw["features"], f"{where}.features")

    phoneme = raw.get("phoneme")
    if not isinstance(phoneme, str) or not phoneme.strip():
        raise SchemaError(f"{where}: 'phoneme' must be a non-empty string, got {phoneme!r}")
    phoneme = phoneme.strip()

    language = _categorical(raw.get("language"), f"{where}.language")
    if language == NA:
        raise SchemaError(f"{where}: 'language' is required")
    if language not in cfg.language.supported:
        raise SchemaError(
            f"{where}.language={language!r} is not in data.language.supported "
            f"{list(cfg.language.supported)}"
        )

    surface_tone = _require_int(raw.get("surface_tone", 0), f"{where}.surface_tone")
    stress = _require_int(raw.get("stress", 0), f"{where}.stress")

    if language == "zh":
        if surface_tone not in cfg.chinese.surface_tone_values:
            raise SchemaError(
                f"{where}: Chinese surface_tone must be one of "
                f"{list(cfg.chinese.surface_tone_values)}, got {surface_tone}"
            )
        if stress != cfg.chinese.stress_default:
            raise SchemaError(
                f"{where}: Chinese stress must be {cfg.chinese.stress_default}, got {stress}"
            )
    elif language == "en":
        if surface_tone != cfg.english.surface_tone_default:
            raise SchemaError(
                f"{where}: English surface_tone must be "
                f"{cfg.english.surface_tone_default}, got {surface_tone}"
            )
        if stress not in cfg.english.stress_values:
            raise SchemaError(
                f"{where}: English stress must be one of "
                f"{list(cfg.english.stress_values)}, got {stress}"
            )

    syllable_role = _categorical(raw.get("syllable_role", "other"), f"{where}.syllable_role")
    if syllable_role == NA:
        syllable_role = "other"

    labels_raw = raw.get("labels")
    if labels_raw is None:
        if require_labels:
            raise SchemaError(f"{where}: missing 'labels'")
        labels = None
    else:
        labels = parse_labels(labels_raw, where, cfg)

    return PhonemeToken(
        phoneme=phoneme,
        language=language,
        surface_tone=surface_tone,
        stress=stress,
        syllable_role=syllable_role,
        articulatory=parse_articulatory(raw.get("articulatory"), where),
        boundary=parse_boundary(raw.get("boundary"), where),
        labels=labels,
    )


def parse_sample(
    raw: Any,
    cfg: DataConfig,
    *,
    require_labels: bool = True,
    location: str = "<memory>",
) -> Sample:
    """Parse and validate one sequence-level record.

    Raises :class:`SchemaError` on any violation; nothing is auto-repaired.
    """
    if not isinstance(raw, dict):
        raise SchemaError(f"{location}: sample must be a JSON object")

    sample_id = str(raw.get("sample_id") or "")
    where_sample = f"{location}[{sample_id or '?'}]"

    schema_version = str(raw.get("schema_version") or "")
    if (
        cfg.validation.fail_on_schema_mismatch
        and schema_version
        and schema_version != cfg.schema_version
    ):
        raise SchemaError(
            f"{where_sample}: schema_version {schema_version!r} != expected "
            f"{cfg.schema_version!r}"
        )

    tokens_raw = raw.get("tokens")
    if tokens_raw is None:
        raise SchemaError(
            f"{where_sample}: missing 'tokens'; the training unit is a full phoneme "
            "sequence, not an isolated phoneme row"
        )
    if not isinstance(tokens_raw, list):
        raise SchemaError(f"{where_sample}: 'tokens' must be a list")
    if not tokens_raw:
        raise SchemaError(f"{where_sample}: 'tokens' must not be empty")
    if len(tokens_raw) > cfg.max_seq_len:
        raise SchemaError(
            f"{where_sample}: sequence length {len(tokens_raw)} exceeds "
            f"data.max_seq_len {cfg.max_seq_len}"
        )

    tokens = tuple(
        parse_token(tok, f"{where_sample}.tokens[{i}]", cfg, require_labels=require_labels)
        for i, tok in enumerate(tokens_raw)
    )

    # Token/label count parity: never silently pad or shift.
    if require_labels:
        labelled = sum(1 for t in tokens if t.labels is not None)
        if labelled != len(tokens):
            raise SchemaError(
                f"{where_sample}: phoneme count {len(tokens)} != label count {labelled}"
            )

    teacher_metadata = raw.get("teacher_metadata") or {}
    if not isinstance(teacher_metadata, dict):
        raise SchemaError(f"{where_sample}: 'teacher_metadata' must be a mapping when present")

    return Sample(
        sample_id=sample_id or f"{location}#auto",
        tokens=tokens,
        text=str(raw.get("text") or ""),
        schema_version=schema_version,
        normalized_text=raw.get("normalized_text"),
        teacher_metadata=teacher_metadata,
    )


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` pairs from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise SchemaError(f"dataset file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{p}:{line_no}: invalid JSON ({exc})") from exc
            yield line_no, record


def load_samples(
    path: str | Path,
    cfg: DataConfig,
    *,
    require_labels: bool = True,
    limit: int | None = None,
) -> list[Sample]:
    """Load and validate every sample in a JSONL file (fail fast)."""
    samples: list[Sample] = []
    for line_no, record in iter_jsonl(path):
        samples.append(
            parse_sample(
                record, cfg, require_labels=require_labels, location=f"{path}:{line_no}"
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise SchemaError(f"{path}: no samples found")
    return samples
