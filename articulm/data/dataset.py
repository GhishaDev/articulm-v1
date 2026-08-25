"""JSONL dataset over full phoneme sequences.

One dataset item is one sentence / utterance. Encoder features are encoded to
per-field ids up front; labels stay separate so an inference dataset can omit
them entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .schema import (
    HUMAN_GOLD_STRENGTH_SOURCES,
    Sample,
    load_samples,
)
from .vocab import FEATURE_KEYS, FeatureVocabulary


@dataclass
class EncodedSample:
    """One sentence encoded into tensors. No padding applied yet."""

    sample_id: str
    length: int
    # [T, num_feature_fields] in canonical FEATURE_KEYS order.
    feature_ids: torch.Tensor
    # [T] int64, or None for inference inputs.
    viseme_ids: torch.Tensor | None
    # [T] float32 in [0,100], or None.
    strength: torch.Tensor | None
    # [T] float32 per-token strength loss multiplier from strength_source.
    strength_weight: torch.Tensor | None
    # [T] bool, True where the label came from a human annotator.
    human_gold_strength: torch.Tensor | None
    text: str = ""
    phonemes: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    surface_tones: tuple[int, ...] = ()
    stresses: tuple[int, ...] = ()
    syllable_roles: tuple[str, ...] = ()
    phrase_positions: tuple[str, ...] = ()

    def __len__(self) -> int:
        return self.length


def _phrase_position(word_start: str, word_end: str, phrase_start: str, phrase_end: str) -> str:
    """Coarse position label used for slice metrics."""
    if phrase_start == "true":
        return "phrase_start"
    if phrase_end == "true":
        return "phrase_end"
    if word_start == "true":
        return "word_start"
    if word_end == "true":
        return "word_end"
    return "word_internal"


def encode_sample(
    sample: Sample,
    vocab: FeatureVocabulary,
    *,
    source_weights: dict[str, float] | None = None,
    strength_scale: float = 100.0,
) -> EncodedSample:
    """Encode one parsed :class:`Sample` into tensors."""
    length = len(sample.tokens)
    feature_ids = torch.empty((length, len(FEATURE_KEYS)), dtype=torch.long)

    has_labels = sample.has_labels
    viseme_ids = torch.empty(length, dtype=torch.long) if has_labels else None
    strength = torch.empty(length, dtype=torch.float32) if has_labels else None
    strength_weight = torch.ones(length, dtype=torch.float32) if has_labels else None
    human_gold = torch.zeros(length, dtype=torch.bool) if has_labels else None

    phonemes: list[str] = []
    languages: list[str] = []
    surface_tones: list[int] = []
    stresses: list[int] = []
    syllable_roles: list[str] = []
    phrase_positions: list[str] = []

    for index, token in enumerate(sample.tokens):
        ids = vocab.encode_token(token)
        for field_index, key in enumerate(FEATURE_KEYS):
            feature_ids[index, field_index] = ids[key]

        phonemes.append(token.phoneme)
        languages.append(token.language)
        surface_tones.append(token.surface_tone)
        stresses.append(token.stress)
        syllable_roles.append(token.syllable_role)
        boundary = token.boundary
        phrase_positions.append(
            _phrase_position(
                boundary.word_start, boundary.word_end, boundary.phrase_start, boundary.phrase_end
            )
        )

        if token.labels is not None and viseme_ids is not None:
            assert strength is not None and strength_weight is not None and human_gold is not None
            viseme_ids[index] = token.labels.viseme_id
            strength[index] = token.labels.strength / strength_scale
            source = token.labels.strength_source.lower()
            if source_weights:
                strength_weight[index] = source_weights.get(source, 1.0)
            human_gold[index] = source in HUMAN_GOLD_STRENGTH_SOURCES

    return EncodedSample(
        sample_id=sample.sample_id,
        length=length,
        feature_ids=feature_ids,
        viseme_ids=viseme_ids,
        strength=strength,
        strength_weight=strength_weight,
        human_gold_strength=human_gold,
        text=sample.text,
        phonemes=tuple(phonemes),
        languages=tuple(languages),
        surface_tones=tuple(surface_tones),
        stresses=tuple(stresses),
        syllable_roles=tuple(syllable_roles),
        phrase_positions=tuple(phrase_positions),
    )


class PhonemeSequenceDataset(Dataset[EncodedSample]):
    """In-memory dataset of encoded phoneme sequences.

    Samples are validated and encoded eagerly so schema violations surface
    before a training run starts rather than mid-epoch.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        vocab: FeatureVocabulary,
        *,
        source_weights: dict[str, float] | None = None,
        strength_scale: float = 100.0,
        retain_parsed_samples: bool = False,
        cache_path: str | Path | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        """Encode samples into tensors.

        The parsed :class:`Sample` objects cost roughly 14x what the encoded
        tensors do (~910 B/token vs ~67 B/token), and training never reads them
        again. They are therefore dropped by default; pass
        ``retain_parsed_samples=True`` if a caller needs the originals.

        ``cache_path`` persists the encoded tensors on disk so later runs with
        the same corpus/vocabulary skip the pure-Python encode loop (~20 min
        for the 6M-token corpus). ``source_path`` is the JSONL the samples
        were loaded from; its size/mtime feed the cache key, so a changed
        corpus is always a miss, never a stale read. The key also covers the
        vocabulary fingerprint, source weights and strength scale.
        ``cache_state`` reports what happened: ``hit`` / ``saved`` /
        ``disabled`` / ``write_failed``.
        """
        if not samples:
            raise ValueError("PhonemeSequenceDataset requires at least one sample")
        self.vocab = vocab
        self.strength_scale = strength_scale
        self.samples: tuple[Sample, ...] = tuple(samples) if retain_parsed_samples else ()
        self.cache_state = "disabled"

        encoded = None
        if cache_path is not None and not retain_parsed_samples:
            from .cache import (
                CacheKeyInput,
                compute_cache_key,
                load_encoded_cache,
                save_encoded_cache,
                vocabulary_fingerprint,
            )

            weights = tuple(sorted((source_weights or {}).items()))
            key = compute_cache_key(
                CacheKeyInput(
                    source_path=Path(source_path or cache_path).resolve(),
                    num_samples=len(samples),
                    vocab_fingerprint=vocabulary_fingerprint(vocab),
                    source_weights=weights,
                    strength_scale=strength_scale,
                )
            )
            encoded = load_encoded_cache(cache_path, key)
            if encoded is not None:
                self.cache_state = "hit"
            self._pending_cache = (cache_path, key)
        else:
            self._pending_cache = None

        if encoded is None:
            encoded = tuple(
                encode_sample(
                    sample,
                    vocab,
                    source_weights=source_weights,
                    strength_scale=strength_scale,
                )
                for sample in samples
            )
            if self._pending_cache is not None:
                cache_path, key = self._pending_cache
                if save_encoded_cache(cache_path, key, encoded):
                    self.cache_state = "saved"
                else:
                    self.cache_state = "write_failed"
        self.encoded: tuple[EncodedSample, ...] = encoded
        self._pending_cache = None

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> EncodedSample:
        return self.encoded[index]

    # ------------------------------------------------------------ helpers

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(item.length for item in self.encoded)

    @property
    def num_tokens(self) -> int:
        return sum(self.lengths)

    @property
    def has_labels(self) -> bool:
        return all(item.viseme_ids is not None for item in self.encoded)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        cfg: DataConfig,
        vocab: FeatureVocabulary,
        *,
        require_labels: bool = True,
        limit: int | None = None,
        source_weights: dict[str, float] | None = None,
    ) -> PhonemeSequenceDataset:
        samples = load_samples(path, cfg, require_labels=require_labels, limit=limit)
        return cls(
            samples,
            vocab,
            source_weights=source_weights,
            strength_scale=cfg.labels.strength_max,
        )

    def subset(self, num_samples: int) -> PhonemeSequenceDataset:
        """First ``num_samples`` sentences, for smoke / tiny-overfit runs."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        keep = min(num_samples, len(self.encoded))
        out = PhonemeSequenceDataset.__new__(PhonemeSequenceDataset)
        out.vocab = self.vocab
        out.strength_scale = self.strength_scale
        out.samples = self.samples[:keep]
        out.encoded = self.encoded[:keep]
        return out
