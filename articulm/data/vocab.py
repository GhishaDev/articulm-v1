"""Stable categorical vocabularies for ArticuLM-V1 encoder features.

Every feature field gets its own vocabulary with a fixed reserved prefix:

```text
0 -> [PAD]   padding, embedded as a zero vector via padding_idx
1 -> [UNK]   unseen category at inference time
```

Nullable fields additionally reserve ``[NA]`` for an explicit JSON ``null``.

Closed-set fields (language, tone, stress, syllable role, boundary) are
seeded from the documented value sets so their ids never depend on which
corpus was scanned first. ``phoneme`` is an open set built from the training
corpus and then frozen into the checkpoint.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..config import DataConfig
from .schema import (
    ARTICULATORY_FIELDS,
    BOUNDARY_FIELDS,
    BOUNDARY_TYPES,
    NA,
    SYLLABLE_ROLES,
    PhonemeToken,
    Sample,
)

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

PAD_ID = 0
UNK_ID = 1

VOCAB_FORMAT_VERSION = "articulm_v1_vocab_v1"

# Canonical, ordered list of encoder feature fields. Embedding tables and the
# collator both key off this order, so it must stay stable.
ARTICULATORY_KEYS = tuple(f"articulatory.{name}" for name in ARTICULATORY_FIELDS)
BOUNDARY_KEYS = tuple(f"boundary.{name}" for name in BOUNDARY_FIELDS)
FEATURE_KEYS = (
    "phoneme",
    "language",
    "surface_tone",
    "stress",
    "syllable_role",
    *ARTICULATORY_KEYS,
    *BOUNDARY_KEYS,
)

_BOOL_TOKENS = ("false", "true")

# Seed values for the closed-set fields. Fields that can legitimately be
# ``null`` in the source JSON also reserve ``[NA]``.
_ARTICULATORY_SEEDS: dict[str, tuple[str, ...]] = {
    "type": (NA, "consonant", "vowel", "silence"),
    "height": (NA, "high", "mid", "low", "close", "open", "near-close", "near-open"),
    "backness": (NA, "front", "central", "back", "near-front", "near-back"),
    "rounded": (NA, *_BOOL_TOKENS),
    "place": (
        NA,
        "bilabial",
        "labiodental",
        "dental",
        "alveolar",
        "postalveolar",
        "retroflex",
        "alveolo-palatal",
        "palatal",
        "velar",
        "uvular",
        "glottal",
    ),
    "manner": (
        NA,
        "plosive",
        "stop",
        "nasal",
        "fricative",
        "affricate",
        "approximant",
        "lateral",
        "trill",
        "tap",
        "glide",
    ),
    "voiced": (NA, *_BOOL_TOKENS),
    "aspirated": (NA, *_BOOL_TOKENS),
}

_BOUNDARY_SEEDS: dict[str, tuple[str, ...]] = {
    "word_start": _BOOL_TOKENS,
    "word_end": _BOOL_TOKENS,
    "phrase_start": _BOOL_TOKENS,
    "phrase_end": _BOOL_TOKENS,
    "boundary_type": BOUNDARY_TYPES,
}


class VocabError(ValueError):
    """Raised on vocabulary construction / loading problems."""


@dataclass
class CategoricalVocabulary:
    """Ordered token table for one feature field."""

    name: str
    tokens: list[str]

    def __post_init__(self) -> None:
        if len(self.tokens) < 2:
            raise VocabError(f"{self.name}: vocabulary needs at least [PAD] and [UNK]")
        if self.tokens[PAD_ID] != PAD_TOKEN:
            raise VocabError(f"{self.name}: index {PAD_ID} must be {PAD_TOKEN}")
        if self.tokens[UNK_ID] != UNK_TOKEN:
            raise VocabError(f"{self.name}: index {UNK_ID} must be {UNK_TOKEN}")
        if len(set(self.tokens)) != len(self.tokens):
            raise VocabError(f"{self.name}: duplicate tokens in vocabulary")
        self._index = {token: i for i, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._index

    def encode(self, token: str) -> int:
        """Map a token to its id, falling back to ``[UNK]``."""
        return self._index.get(token, UNK_ID)

    def decode(self, index: int) -> str:
        if not 0 <= index < len(self.tokens):
            raise VocabError(f"{self.name}: id {index} out of range")
        return self.tokens[index]

    def add(self, token: str) -> int:
        existing = self._index.get(token)
        if existing is not None:
            return existing
        self.tokens.append(token)
        self._index[token] = len(self.tokens) - 1
        return len(self.tokens) - 1

    @classmethod
    def build(cls, name: str, values: Iterable[str]) -> CategoricalVocabulary:
        """Build a vocabulary with reserved prefix plus sorted unique values."""
        reserved = [PAD_TOKEN, UNK_TOKEN]
        extra = sorted({v for v in values if v not in reserved})
        return cls(name=name, tokens=reserved + extra)


@dataclass
class FeatureVocabulary:
    """All per-field vocabularies plus the label space."""

    fields: dict[str, CategoricalVocabulary]
    viseme_classes: int
    strength_min: float
    strength_max: float
    format_version: str = VOCAB_FORMAT_VERSION

    def __post_init__(self) -> None:
        missing = [key for key in FEATURE_KEYS if key not in self.fields]
        if missing:
            raise VocabError(f"missing vocabularies for fields {missing}")
        unexpected = sorted(set(self.fields) - set(FEATURE_KEYS))
        if unexpected:
            raise VocabError(f"unexpected vocabulary fields {unexpected}")

    def sizes(self) -> dict[str, int]:
        """Vocabulary size per field, in canonical field order."""
        return {key: len(self.fields[key]) for key in FEATURE_KEYS}

    def encode_token(self, token: PhonemeToken) -> dict[str, int]:
        """Encode one phoneme token into per-field ids."""
        articulatory = token.articulatory.as_dict()
        boundary = token.boundary.as_dict()
        ids = {
            "phoneme": self.fields["phoneme"].encode(token.phoneme),
            "language": self.fields["language"].encode(token.language),
            "surface_tone": self.fields["surface_tone"].encode(str(token.surface_tone)),
            "stress": self.fields["stress"].encode(str(token.stress)),
            "syllable_role": self.fields["syllable_role"].encode(token.syllable_role),
        }
        for name in ARTICULATORY_FIELDS:
            key = f"articulatory.{name}"
            ids[key] = self.fields[key].encode(articulatory[name])
        for name in BOUNDARY_FIELDS:
            key = f"boundary.{name}"
            ids[key] = self.fields[key].encode(boundary[name])
        return ids

    def unknown_phoneme(self, phoneme: str) -> bool:
        return phoneme not in self.fields["phoneme"]

    # ---------------------------------------------------------------- I/O

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "viseme_classes": self.viseme_classes,
            "strength_min": self.strength_min,
            "strength_max": self.strength_max,
            "fields": {key: self.fields[key].tokens for key in FEATURE_KEYS},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> FeatureVocabulary:
        version = str(raw.get("format_version", ""))
        if version != VOCAB_FORMAT_VERSION:
            raise VocabError(
                f"vocab format_version {version!r} != expected {VOCAB_FORMAT_VERSION!r}"
            )
        fields_raw = raw.get("fields")
        if not isinstance(fields_raw, dict):
            raise VocabError("vocab 'fields' must be a mapping")
        fields = {
            key: CategoricalVocabulary(name=key, tokens=list(tokens))
            for key, tokens in fields_raw.items()
        }
        return cls(
            fields=fields,
            viseme_classes=int(raw["viseme_classes"]),  # type: ignore[arg-type]
            strength_min=float(raw["strength_min"]),  # type: ignore[arg-type]
            strength_max=float(raw["strength_max"]),  # type: ignore[arg-type]
            format_version=version,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=False)
            fh.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> FeatureVocabulary:
        p = Path(path)
        if not p.is_file():
            raise VocabError(f"vocab file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _closed_set_vocabularies(cfg: DataConfig) -> dict[str, CategoricalVocabulary]:
    """Seed the fields whose value sets are fixed by the data specification."""
    tone_values = sorted(
        {str(cfg.english.surface_tone_default)} | {str(v) for v in cfg.chinese.surface_tone_values}
    )
    stress_values = sorted(
        {str(cfg.chinese.stress_default)} | {str(v) for v in cfg.english.stress_values}
    )

    vocabs: dict[str, CategoricalVocabulary] = {
        "language": CategoricalVocabulary.build("language", cfg.language.supported),
        "surface_tone": CategoricalVocabulary.build("surface_tone", tone_values),
        "stress": CategoricalVocabulary.build("stress", stress_values),
        "syllable_role": CategoricalVocabulary.build("syllable_role", SYLLABLE_ROLES),
    }
    for name, seeds in _ARTICULATORY_SEEDS.items():
        key = f"articulatory.{name}"
        vocabs[key] = CategoricalVocabulary.build(key, seeds)
    for name, seeds in _BOUNDARY_SEEDS.items():
        key = f"boundary.{name}"
        vocabs[key] = CategoricalVocabulary.build(key, seeds)
    return vocabs


def build_vocabulary(
    samples: Iterable[Sample],
    cfg: DataConfig,
    *,
    extend_closed_sets: bool = True,
) -> FeatureVocabulary:
    """Build a vocabulary from training samples.

    Closed-set fields are seeded from the spec; values seen in the corpus but
    absent from the seeds are appended when ``extend_closed_sets`` is true so
    a real corpus never silently collapses into ``[UNK]``.
    """
    vocabs = _closed_set_vocabularies(cfg)
    phonemes: set[str] = set()

    for sample in samples:
        for token in sample.tokens:
            phonemes.add(token.phoneme)
            if not extend_closed_sets:
                continue
            encoded = {
                "language": token.language,
                "surface_tone": str(token.surface_tone),
                "stress": str(token.stress),
                "syllable_role": token.syllable_role,
            }
            for name, value in token.articulatory.as_dict().items():
                encoded[f"articulatory.{name}"] = value
            for name, value in token.boundary.as_dict().items():
                encoded[f"boundary.{name}"] = value
            for key, value in encoded.items():
                if value not in vocabs[key]:
                    vocabs[key].add(value)

    if not phonemes:
        raise VocabError("no phonemes found while building the vocabulary")

    vocabs["phoneme"] = CategoricalVocabulary.build("phoneme", phonemes)

    return FeatureVocabulary(
        fields=vocabs,
        viseme_classes=cfg.labels.viseme_classes,
        strength_min=cfg.labels.strength_min,
        strength_max=cfg.labels.strength_max,
    )
