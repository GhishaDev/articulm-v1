"""Pre-training dataset validation and reporting.

Run before every training job (docs/03_training_data_spec.md):

```bash
python -m articulm.data.validate --config config/data_v1.yaml
```

Reports sentence/token counts, sequence-length stats, viseme distribution,
strength histogram, language / tone / stress / boundary distributions and the
unknown-phoneme rate. Schema violations fail fast rather than being repaired.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import DataConfig, load_data_config
from .schema import (
    HUMAN_GOLD_STRENGTH_SOURCES,
    Sample,
    SchemaError,
    load_samples,
)
from .vocab import FeatureVocabulary


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


@dataclass
class DatasetReport:
    """Everything the pre-run gate needs to print and check."""

    path: str
    num_sentences: int
    num_phoneme_tokens: int
    seq_len_min: int
    seq_len_mean: float
    seq_len_p50: float
    seq_len_p95: float
    seq_len_max: int
    viseme_distribution: dict[int, int]
    strength_stats: dict[str, float]
    strength_by_viseme_mean: dict[int, float]
    languages: dict[str, int]
    surface_tones: dict[int, int]
    stresses: dict[int, int]
    syllable_roles: dict[str, int]
    boundary_types: dict[str, int]
    phrase_end_tokens: int
    word_end_tokens: int
    viseme_sources: dict[str, int]
    strength_sources: dict[str, int]
    num_human_gold_strength_tokens: int
    unique_phonemes: int
    unknown_phoneme_rate: float | None = None
    unknown_phoneme_examples: list[str] = field(default_factory=list)
    missing_viseme_classes: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "num_sentences": self.num_sentences,
            "num_phoneme_tokens": self.num_phoneme_tokens,
            "sequence_length": {
                "min": self.seq_len_min,
                "mean": self.seq_len_mean,
                "p50": self.seq_len_p50,
                "p95": self.seq_len_p95,
                "max": self.seq_len_max,
            },
            "viseme_distribution": {str(k): v for k, v in sorted(self.viseme_distribution.items())},
            "missing_viseme_classes": self.missing_viseme_classes,
            "strength": self.strength_stats,
            "strength_by_viseme_mean": {
                str(k): v for k, v in sorted(self.strength_by_viseme_mean.items())
            },
            "languages": self.languages,
            "surface_tones": {str(k): v for k, v in sorted(self.surface_tones.items())},
            "stresses": {str(k): v for k, v in sorted(self.stresses.items())},
            "syllable_roles": self.syllable_roles,
            "boundary_types": self.boundary_types,
            "phrase_end_tokens": self.phrase_end_tokens,
            "word_end_tokens": self.word_end_tokens,
            "viseme_sources": self.viseme_sources,
            "strength_sources": self.strength_sources,
            "num_human_gold_strength_tokens": self.num_human_gold_strength_tokens,
            "unique_phonemes": self.unique_phonemes,
            "unknown_phoneme_rate": self.unknown_phoneme_rate,
            "unknown_phoneme_examples": self.unknown_phoneme_examples,
        }

    def render(self) -> str:
        lines = [
            f"Dataset:                  {self.path}",
            f"Sentences:                {self.num_sentences:,}",
            f"Phoneme Tokens:           {self.num_phoneme_tokens:,}",
            f"Seq Length min/mean:      {self.seq_len_min} / {self.seq_len_mean:.1f}",
            f"Seq Length p50/p95/max:   {self.seq_len_p50:.0f} / {self.seq_len_p95:.0f} / {self.seq_len_max}",
        ]
        if self.unknown_phoneme_rate is not None:
            lines.append(f"Unknown Phoneme Rate:     {self.unknown_phoneme_rate * 100:.4f}%")
            if self.unknown_phoneme_examples:
                lines.append(
                    f"  examples:               {', '.join(self.unknown_phoneme_examples)}"
                )
        lines.append(f"Unique Phonemes:          {self.unique_phonemes}")
        lines.append("")
        lines.append("Viseme Classes:")
        total = max(self.num_phoneme_tokens, 1)
        for viseme_id in sorted(self.viseme_distribution):
            count = self.viseme_distribution[viseme_id]
            mean_strength = self.strength_by_viseme_mean.get(viseme_id, float("nan"))
            lines.append(
                f"  {viseme_id:2d}: {count:>10,}  ({count / total * 100:5.2f}%)  "
                f"mean strength {mean_strength:6.2f}"
            )
        if self.missing_viseme_classes:
            lines.append(f"  MISSING CLASSES: {self.missing_viseme_classes}")
        lines.append("")
        lines.append("Strength:")
        for key in ("mean", "std", "min", "p05", "p50", "p95", "max"):
            lines.append(f"  {key:<4}: {self.strength_stats[key]:.2f}")
        lines.append("")
        lines.append(f"Languages:                {_pct(self.languages, total)}")
        lines.append(f"Surface tones:            {_pct(self.surface_tones, total)}")
        lines.append(f"Stress:                   {_pct(self.stresses, total)}")
        lines.append(f"Syllable roles:           {_pct(self.syllable_roles, total)}")
        lines.append(f"Boundary types:           {_pct(self.boundary_types, total)}")
        lines.append(f"Word-end tokens:          {self.word_end_tokens:,}")
        lines.append(f"Phrase-end tokens:        {self.phrase_end_tokens:,}")
        lines.append(f"Viseme label sources:     {dict(sorted(self.viseme_sources.items()))}")
        lines.append(f"Strength label sources:   {dict(sorted(self.strength_sources.items()))}")
        lines.append(
            f"Human Gold strength:      {self.num_human_gold_strength_tokens:,} tokens "
            "(everything else is a programmatic prior, not Human Gold)"
        )
        return "\n".join(lines)


def _pct(counter: dict[Any, int], total: int) -> str:
    if not counter:
        return "{}"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ", ".join(f"{key}: {value / total * 100:.2f}%" for key, value in items)


def build_report(
    samples: Sequence[Sample],
    cfg: DataConfig,
    *,
    path: str = "<memory>",
    vocab: FeatureVocabulary | None = None,
) -> DatasetReport:
    """Compute a dataset report over already-validated samples."""
    if not samples:
        raise SchemaError(f"{path}: no samples to report on")

    lengths = sorted(len(sample) for sample in samples)
    num_tokens = sum(lengths)

    visemes: Counter[int] = Counter()
    languages: Counter[str] = Counter()
    tones: Counter[int] = Counter()
    stresses: Counter[int] = Counter()
    roles: Counter[str] = Counter()
    boundary_types: Counter[str] = Counter()
    viseme_sources: Counter[str] = Counter()
    strength_sources: Counter[str] = Counter()
    phonemes: Counter[str] = Counter()

    strengths: list[float] = []
    strength_sum_by_viseme: dict[int, float] = {}
    strength_count_by_viseme: dict[int, int] = {}
    phrase_end = 0
    word_end = 0
    human_gold = 0
    unknown_phonemes: Counter[str] = Counter()

    for sample in samples:
        for token in sample.tokens:
            languages[token.language] += 1
            tones[token.surface_tone] += 1
            stresses[token.stress] += 1
            roles[token.syllable_role] += 1
            boundary_types[token.boundary.boundary_type] += 1
            phonemes[token.phoneme] += 1
            if token.boundary.phrase_end == "true":
                phrase_end += 1
            if token.boundary.word_end == "true":
                word_end += 1
            if vocab is not None and vocab.unknown_phoneme(token.phoneme):
                unknown_phonemes[token.phoneme] += 1

            if token.labels is None:
                continue
            visemes[token.labels.viseme_id] += 1
            strengths.append(token.labels.strength)
            strength_sum_by_viseme[token.labels.viseme_id] = (
                strength_sum_by_viseme.get(token.labels.viseme_id, 0.0)
                + token.labels.strength
            )
            strength_count_by_viseme[token.labels.viseme_id] = (
                strength_count_by_viseme.get(token.labels.viseme_id, 0) + 1
            )
            viseme_sources[token.labels.viseme_source] += 1
            strength_sources[token.labels.strength_source] += 1
            if token.labels.strength_source.lower() in HUMAN_GOLD_STRENGTH_SOURCES:
                human_gold += 1

    sorted_strengths = sorted(strengths)
    if sorted_strengths:
        mean = sum(sorted_strengths) / len(sorted_strengths)
        variance = sum((v - mean) ** 2 for v in sorted_strengths) / len(sorted_strengths)
        strength_stats = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": sorted_strengths[0],
            "p05": _percentile(sorted_strengths, 0.05),
            "p50": _percentile(sorted_strengths, 0.50),
            "p95": _percentile(sorted_strengths, 0.95),
            "max": sorted_strengths[-1],
        }
    else:
        strength_stats = {k: float("nan") for k in ("mean", "std", "min", "p05", "p50", "p95", "max")}

    unknown_rate = (
        sum(unknown_phonemes.values()) / num_tokens if vocab is not None and num_tokens else None
    )

    return DatasetReport(
        path=str(path),
        num_sentences=len(samples),
        num_phoneme_tokens=num_tokens,
        seq_len_min=lengths[0],
        seq_len_mean=num_tokens / len(samples),
        seq_len_p50=_percentile(lengths, 0.50),
        seq_len_p95=_percentile(lengths, 0.95),
        seq_len_max=lengths[-1],
        viseme_distribution=dict(visemes),
        strength_stats=strength_stats,
        strength_by_viseme_mean={
            viseme_id: strength_sum_by_viseme[viseme_id] / strength_count_by_viseme[viseme_id]
            for viseme_id in sorted(strength_count_by_viseme)
        },
        languages=dict(languages),
        surface_tones=dict(tones),
        stresses=dict(stresses),
        syllable_roles=dict(roles),
        boundary_types=dict(boundary_types),
        phrase_end_tokens=phrase_end,
        word_end_tokens=word_end,
        viseme_sources=dict(viseme_sources),
        strength_sources=dict(strength_sources),
        num_human_gold_strength_tokens=human_gold,
        unique_phonemes=len(phonemes),
        unknown_phoneme_rate=unknown_rate,
        unknown_phoneme_examples=[p for p, _ in unknown_phonemes.most_common(10)],
        missing_viseme_classes=[
            viseme_id for viseme_id in range(cfg.labels.viseme_classes) if viseme_id not in visemes
        ],
    )


def validate_file(
    path: str | Path,
    cfg: DataConfig,
    *,
    vocab: FeatureVocabulary | None = None,
    limit: int | None = None,
) -> DatasetReport:
    samples = load_samples(path, cfg, require_labels=True, limit=limit)
    return build_report(samples, cfg, path=str(path), vocab=vocab)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.data.validate",
        description="Validate ArticuLM training data and print a pre-run report.",
    )
    parser.add_argument("--config", required=True, help="path to a data config YAML")
    parser.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=None, help="only read the first N sentences")
    parser.add_argument("--json-out", type=Path, default=None, help="write the report as JSON")
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    candidates: list[tuple[str, str | None]] = [
        ("train", cfg.train_path),
        ("validation", cfg.validation_path),
        ("test", cfg.test_path),
    ]
    selected = [
        (name, path)
        for name, path in candidates
        if path and (args.split == "all" or args.split == name)
    ]
    if not selected:
        print(f"no dataset path configured for split={args.split}")
        return 2

    reports: dict[str, Any] = {}
    failures = 0
    for name, path in selected:
        print(f"\n=== {name} ===")
        try:
            report = validate_file(path, cfg, limit=args.limit)
        except SchemaError as exc:
            failures += 1
            print(f"FAILED: {exc}")
            continue
        print(report.render())
        reports[name] = report.as_dict()

    if args.json_out is not None and reports:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json_out}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
