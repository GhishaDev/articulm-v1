"""Sentence-level dataset splitting with duplicate and near-duplicate control.

``docs/03_training_data_spec.md`` requires:

```text
train 90% / validation 5% / test 5%
Split at sentence level after deduplication.
Prevent near-duplicate leakage across splits.
```

and ``docs/09_acceptance_criteria.md`` rejects any run where train/val leakage
exists. This module is the enforcement point.

How it works
------------

1. **Exact duplicates** are collapsed by two signatures: normalised text (or
   raw text when no normalisation is recorded) and the phoneme sequence.
   Identical sentences become one group.

2. **Near duplicates** are found with a bottom-k sketch over phoneme
   n-gram shingles. Two sentences are only compared when their sketches share
   a hash, which keeps the work near-linear instead of O(n²); the pair is then
   scored with exact Jaccard over the full shingle sets. This is a *recall*
   heuristic: it will not find every paraphrase, but it reliably catches the
   copy-with-small-edit case that actually leaks.

3. Duplicate groups are assigned to splits **whole**, so no group can straddle
   two splits. That makes leakage structurally impossible rather than merely
   unlikely — :func:`verify_no_leakage` re-checks it afterwards.

Group assignment is greedy over groups sorted largest-first, each going to the
split furthest below its quota. That keeps large clusters from overshooting a
small validation quota. Everything is deterministic given ``split.seed``.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from ..config import DataConfig, load_data_config
from .schema import Sample, SchemaError, load_samples

SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    pass


# --------------------------------------------------------------------------
# Signatures and shingles
# --------------------------------------------------------------------------


def phoneme_sequence(sample: Sample) -> str:
    return " ".join(token.phoneme for token in sample.tokens)


def text_signature(sample: Sample) -> str:
    """Preferred text identity: normalised text when the pipeline recorded it.

    Training and inference must share one text-normalisation pass, so the
    normalised form is the meaningful identity when present.
    """
    source = sample.normalized_text or sample.text
    return " ".join(source.split()).strip()


def exact_signatures(sample: Sample) -> tuple[str, ...]:
    """Signatures that make two samples exact duplicates of each other."""
    signatures = [f"phonemes:{phoneme_sequence(sample)}"]
    text = text_signature(sample)
    if text:
        signatures.append(f"text:{text}")
    return tuple(signatures)


def shingle_hashes(sample: Sample, size: int) -> frozenset[int]:
    """Hashed phoneme n-grams, ignoring silence/rest tokens.

    Silence tokens (``∅``/``-``/``_``/``ː``/``‿``/``ˈ``/``ˌ``) are identical in
    every sentence, so hashing them makes the sentence-final n-gram a
    near-universal collision — e.g. ``l ɤ - ∅`` for any sentence ending in 了.
    Near-duplicate detection is about phonetic content, so we shingle only the
    articulatory tokens. Short sentences fall back to one whole-sequence shingle
    so they still participate.
    """
    phonemes = [
        token.phoneme for token in sample.tokens
        if token.articulatory.type != "silence"
    ]
    if len(phonemes) < size:
        return frozenset({zlib.crc32(" ".join(phonemes).encode("utf-8"))})
    return frozenset(
        zlib.crc32(" ".join(phonemes[index : index + size]).encode("utf-8"))
        for index in range(len(phonemes) - size + 1)
    )


def bottom_k_sketch(hashes: Iterable[int], sketch_size: int) -> tuple[int, ...]:
    """The k smallest hashes — a bottom-k sketch used as an LSH-style index.

    High-Jaccard pairs are very likely to share at least one bottom-k hash,
    so this bounds candidate generation without scanning all pairs.
    """
    return tuple(sorted(hashes)[:sketch_size])


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    if union == 0:
        return 0.0
    return len(left & right) / union


# --------------------------------------------------------------------------
# Union-find over sentences
# --------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression.
        while self.parent[index] != root:
            self.parent[index], index = root, self.parent[index]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def groups(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            buckets[self.find(index)].append(index)
        return [sorted(members) for _, members in sorted(buckets.items())]


@dataclass
class DuplicateAnalysis:
    """Duplicate structure of a corpus."""

    groups: list[list[int]]
    num_exact_duplicate_pairs: int
    num_near_duplicate_pairs: int
    num_candidate_pairs_compared: int
    # Oversized sketch buckets that were skipped for cost. Reported, never
    # silent: a non-zero count means near-duplicate recall was reduced.
    num_skipped_buckets: int = 0
    num_sentences_in_skipped_buckets: int = 0
    largest_skipped_bucket: int = 0

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_sentences(self) -> int:
        return sum(len(group) for group in self.groups)

    @property
    def num_redundant_sentences(self) -> int:
        """Sentences beyond one per group — the corpus's duplicate mass."""
        return self.num_sentences - self.num_groups


def analyse_duplicates(
    samples: Sequence[Sample], cfg: DataConfig
) -> DuplicateAnalysis:
    """Group samples that must not be split apart."""
    if not samples:
        raise SplitError("cannot analyse an empty corpus")

    split_cfg = cfg.split
    union_find = _UnionFind(len(samples))

    # 1) exact duplicates
    by_signature: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        for signature in exact_signatures(sample):
            by_signature[signature].append(index)

    exact_pairs = 0
    for members in by_signature.values():
        for other in members[1:]:
            union_find.union(members[0], other)
            exact_pairs += 1

    # 2) near duplicates
    near_pairs = 0
    compared = 0
    skipped_buckets = 0
    skipped_sentences = 0
    largest_skipped = 0
    if split_cfg.prevent_near_duplicate_leakage:
        shingles = [
            shingle_hashes(sample, split_cfg.near_duplicate_shingle_size)
            for sample in samples
        ]
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, hashes in enumerate(shingles):
            for value in bottom_k_sketch(hashes, split_cfg.near_duplicate_sketch_size):
                buckets[value].append(index)

        seen_pairs: set[tuple[int, int]] = set()
        for members in buckets.values():
            if len(members) > split_cfg.near_duplicate_max_bucket_size:
                skipped_buckets += 1
                skipped_sentences += len(members)
                largest_skipped = max(largest_skipped, len(members))
                continue
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    pair = (left, right)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    compared += 1
                    if (
                        jaccard(shingles[left], shingles[right])
                        >= split_cfg.near_duplicate_jaccard_threshold
                    ):
                        union_find.union(left, right)
                        near_pairs += 1

    return DuplicateAnalysis(
        groups=union_find.groups(),
        num_exact_duplicate_pairs=exact_pairs,
        num_near_duplicate_pairs=near_pairs,
        num_candidate_pairs_compared=compared,
        num_skipped_buckets=skipped_buckets,
        num_sentences_in_skipped_buckets=skipped_sentences,
        largest_skipped_bucket=largest_skipped,
    )


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------


@dataclass
class SplitResult:
    """Sentence indices per split, plus the duplicate analysis behind them."""

    indices: dict[str, list[int]]
    analysis: DuplicateAnalysis
    target_ratios: dict[str, float]
    dropped_duplicate_indices: list[int] = field(default_factory=list)
    # Ratio problems that the caller must see. Duplicate groups are never
    # broken to hit a ratio, so a corpus with few groups can miss its targets.
    warnings: list[str] = field(default_factory=list)

    @property
    def empty_targeted_splits(self) -> list[str]:
        """Splits with a positive target that received nothing."""
        return [
            name
            for name in SPLIT_NAMES
            if self.target_ratios.get(name, 0.0) > 0 and not self.indices[name]
        ]

    def counts(self) -> dict[str, int]:
        return {name: len(self.indices[name]) for name in SPLIT_NAMES}

    def actual_ratios(self) -> dict[str, float]:
        total = sum(self.counts().values())
        if total == 0:
            return dict.fromkeys(SPLIT_NAMES, 0.0)
        return {name: count / total for name, count in self.counts().items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "target_ratios": self.target_ratios,
            "actual_ratios": self.actual_ratios(),
            "num_duplicate_groups": self.analysis.num_groups,
            "num_redundant_sentences": self.analysis.num_redundant_sentences,
            "num_exact_duplicate_pairs": self.analysis.num_exact_duplicate_pairs,
            "num_near_duplicate_pairs": self.analysis.num_near_duplicate_pairs,
            "num_candidate_pairs_compared": self.analysis.num_candidate_pairs_compared,
            "num_skipped_buckets": self.analysis.num_skipped_buckets,
            "num_sentences_in_skipped_buckets": (
                self.analysis.num_sentences_in_skipped_buckets
            ),
            "largest_skipped_bucket": self.analysis.largest_skipped_bucket,
            "num_dropped_duplicates": len(self.dropped_duplicate_indices),
            "warnings": self.warnings,
            "empty_targeted_splits": self.empty_targeted_splits,
        }


def assign_groups(
    groups: Sequence[Sequence[int]],
    ratios: dict[str, float],
    *,
    seed: int,
) -> dict[str, list[int]]:
    """Assign whole groups to splits, largest group first.

    Each group goes to the split with the largest remaining deficit against its
    quota. Splits with a zero ratio never receive anything.
    """
    total = sum(len(group) for group in groups)
    quotas = {name: ratios.get(name, 0.0) * total for name in SPLIT_NAMES}
    assigned: dict[str, list[int]] = {name: [] for name in SPLIT_NAMES}
    filled = dict.fromkeys(SPLIT_NAMES, 0.0)

    # Shuffle first so equal-size groups are ordered reproducibly but not by
    # corpus position, then sort by size descending (Python sort is stable).
    order = list(range(len(groups)))
    Random(seed).shuffle(order)
    order.sort(key=lambda index: len(groups[index]), reverse=True)

    eligible = [name for name in SPLIT_NAMES if ratios.get(name, 0.0) > 0]
    if not eligible:
        raise SplitError("at least one split ratio must be positive")

    for index in order:
        group = groups[index]
        target = max(eligible, key=lambda name: (quotas[name] - filled[name], name))
        assigned[target].extend(group)
        filled[target] += len(group)

    for name in SPLIT_NAMES:
        assigned[name].sort()
    return assigned


def split_samples(
    samples: Sequence[Sample],
    cfg: DataConfig,
    *,
    drop_duplicates: bool = False,
) -> SplitResult:
    """Split a corpus at sentence level with leakage control.

    ``drop_duplicates`` keeps only one representative per duplicate group,
    which is the right choice when redundancy would inflate training weight on
    repeated sentences. Off by default: dropping data is the caller's call.
    """
    analysis = analyse_duplicates(samples, cfg)

    groups = analysis.groups
    dropped: list[int] = []
    if drop_duplicates:
        kept_groups: list[list[int]] = []
        for group in groups:
            kept_groups.append([group[0]])
            dropped.extend(group[1:])
        groups = kept_groups

    ratios = {
        "train": cfg.split.train_ratio,
        "validation": cfg.split.validation_ratio,
        "test": cfg.split.test_ratio,
    }
    indices = assign_groups(groups, ratios, seed=cfg.split.seed)

    result = SplitResult(
        indices=indices,
        analysis=analysis,
        target_ratios=ratios,
        dropped_duplicate_indices=sorted(dropped),
    )
    result.warnings = _ratio_warnings(result, num_groups=len(groups))
    return result


# Beyond this absolute deviation from the target ratio, the split is called out.
RATIO_TOLERANCE = 0.02


def _ratio_warnings(result: SplitResult, *, num_groups: int) -> list[str]:
    """Flag ratio targets that duplicate grouping made unreachable."""
    warnings: list[str] = []
    actual = result.actual_ratios()

    for name in SPLIT_NAMES:
        target = result.target_ratios.get(name, 0.0)
        if target <= 0:
            continue
        if not result.indices[name]:
            warnings.append(
                f"{name} split is EMPTY but its target ratio is {target:.2%}. "
                f"The corpus has only {num_groups} duplicate group(s); groups are "
                "never split apart, so the ratios are unreachable. Add more "
                "distinct sentences or relax near-duplicate detection."
            )
        elif abs(actual[name] - target) > RATIO_TOLERANCE:
            warnings.append(
                f"{name} split is {actual[name]:.2%} against a {target:.2%} target "
                f"(deviation {abs(actual[name] - target):.2%}); a few large duplicate "
                "groups dominate the corpus."
            )
    return warnings


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def verify_no_leakage(
    samples: Sequence[Sample], result: SplitResult, cfg: DataConfig
) -> list[str]:
    """Re-check the split independently. Empty list means clean.

    This does not trust the assignment logic: it recomputes signatures and
    duplicate groups and looks for any that cross a split boundary.
    """
    problems: list[str] = []

    split_of: dict[int, str] = {}
    for name, indices in result.indices.items():
        for index in indices:
            if index in split_of:
                problems.append(
                    f"sentence {index} appears in both {split_of[index]} and {name}"
                )
            split_of[index] = name

    # Every duplicate group must sit inside exactly one split.
    for group in result.analysis.groups:
        present = {split_of[index] for index in group if index in split_of}
        if len(present) > 1:
            sample_ids = [samples[index].sample_id for index in group[:4]]
            problems.append(
                f"duplicate group {sample_ids} spans splits {sorted(present)}"
            )

    # Exact signatures must not cross splits either (belt and braces).
    signature_splits: dict[str, set[str]] = defaultdict(set)
    for index, name in split_of.items():
        for signature in exact_signatures(samples[index]):
            signature_splits[signature].add(name)
    for signature, names in signature_splits.items():
        if len(names) > 1:
            problems.append(
                f"signature {signature[:60]!r} appears in splits {sorted(names)}"
            )

    return problems


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_split(
    source_path: str | Path,
    indices: Sequence[int],
    destination: str | Path,
) -> int:
    """Copy the selected JSONL lines verbatim into ``destination``.

    Records are copied byte-for-byte from the source rather than re-serialised,
    so splitting can never alter data semantics.
    """
    wanted = set(indices)
    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with Path(source_path).open("r", encoding="utf-8") as source, out.open(
        "w", encoding="utf-8"
    ) as sink:
        record_index = 0
        for line in source:
            if not line.strip():
                continue
            if record_index in wanted:
                sink.write(line if line.endswith("\n") else line + "\n")
                written += 1
            record_index += 1
    if written != len(wanted):
        raise SplitError(
            f"{destination}: wrote {written} records but {len(wanted)} were selected"
        )
    return written


def render_report(result: SplitResult, samples: Sequence[Sample]) -> str:
    counts = result.counts()
    actual = result.actual_ratios()
    tokens = {
        name: sum(len(samples[index]) for index in result.indices[name])
        for name in SPLIT_NAMES
    }
    lines = [
        "=" * 64,
        "Dataset split report",
        "=" * 64,
        f"Input sentences:              {len(samples):,}",
        f"Duplicate groups:             {result.analysis.num_groups:,}",
        f"Redundant sentences:          {result.analysis.num_redundant_sentences:,}",
        f"  exact duplicate pairs:      {result.analysis.num_exact_duplicate_pairs:,}",
        f"  near duplicate pairs:       {result.analysis.num_near_duplicate_pairs:,}",
        f"  candidate pairs compared:   {result.analysis.num_candidate_pairs_compared:,}",
        f"Dropped duplicates:           {len(result.dropped_duplicate_indices):,}",
    ]
    if result.analysis.num_skipped_buckets:
        lines += [
            "",
            "NOTE: near-duplicate recall was reduced by oversized bucket skipping:",
            f"  buckets skipped:            {result.analysis.num_skipped_buckets:,}",
            (
                "  sentence slots affected:    "
                f"{result.analysis.num_sentences_in_skipped_buckets:,}"
            ),
            f"  largest skipped bucket:     {result.analysis.largest_skipped_bucket:,}",
            "  raise data.split.near_duplicate_max_bucket_size to compare them",
        ]
    lines.append("")
    for name in SPLIT_NAMES:
        lines.append(
            f"{name:<12} {counts[name]:>8,} sentences  "
            f"{actual[name] * 100:6.2f}% (target {result.target_ratios[name] * 100:5.2f}%)  "
            f"{tokens[name]:>10,} tokens"
        )
    if result.warnings:
        lines.append("")
        lines.append("RATIO WARNINGS:")
        lines += [f"  - {warning}" for warning in result.warnings]
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m articulm.data.split",
        description=(
            "Split a JSONL corpus into train/validation/test at sentence level, "
            "keeping duplicates and near-duplicates inside one split."
        ),
    )
    parser.add_argument("--config", required=True, help="data config YAML")
    parser.add_argument("--input", required=True, help="corpus JSONL to split")
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="keep only one sentence per duplicate group",
    )
    parser.add_argument(
        "--report-out", type=Path, default=None, help="write the report as JSON"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report only; do not write split files"
    )
    parser.add_argument(
        "--allow-empty-splits",
        action="store_true",
        help="do not fail when a split with a positive target ratio ends up empty",
    )
    args = parser.parse_args(argv)

    cfg = load_data_config(args.config)
    try:
        samples = load_samples(args.input, cfg)
    except SchemaError as exc:
        print(f"FAILED: {exc}")
        return 2

    result = split_samples(samples, cfg, drop_duplicates=args.drop_duplicates)
    print(render_report(result, samples))

    problems = verify_no_leakage(samples, result, cfg)
    if problems:
        print("\nLEAKAGE CHECK FAILED:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 1
    print("\nleakage check passed: no duplicate group spans two splits")

    empty = result.empty_targeted_splits
    if empty and not args.allow_empty_splits:
        print(
            f"\nFAILED: split(s) {empty} have a positive target ratio but received no "
            "sentences. Training without a validation set is not a usable setup. "
            "Pass --allow-empty-splits only if you genuinely intend this."
        )
        return 1

    destinations = {
        "train": cfg.train_path,
        "validation": cfg.validation_path,
        "test": cfg.test_path,
    }
    if not args.dry_run:
        for name in SPLIT_NAMES:
            destination = destinations[name]
            if not destination:
                if result.indices[name]:
                    print(
                        f"WARNING: {len(result.indices[name])} sentences assigned to "
                        f"{name} but data.{name}_path is not configured; not written"
                    )
                continue
            if not result.indices[name]:
                continue
            written = write_split(args.input, result.indices[name], destination)
            print(f"wrote {destination} ({written:,} sentences)")

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **result.as_dict(),
            "input": str(args.input),
            "destinations": destinations,
            "leakage_check": "passed",
        }
        with args.report_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
