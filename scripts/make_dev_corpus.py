#!/usr/bin/env python3
"""Generate a larger synthetic corpus for pipeline exercise.

Development tooling. Labels are illustrative pseudo labels
(``strength_source="pseudo_strength_v1"``), NOT Human Gold, and this corpus
says nothing about real-world accuracy. Its purpose is to exercise the
production code path end to end: split -> validate -> dynamic-token batching ->
evaluation -> checkpointing.

Real training data must come from ``articulm_data_pipeline``.

A configurable fraction of sentences is emitted as deliberate duplicates and
near-duplicates so the splitter's leakage control has something to catch.

Usage:
    python scripts/make_dev_corpus.py --num-samples 2000 --out data/dev/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fixture_corpus import Word, build_sample
from word_bank import EN_WORDS, ZH_WORDS

SEED = 20260821


def generate_sentences(
    count: int, rng: random.Random
) -> list[tuple[str, list[list[Word]]]]:
    zh_pool = list(ZH_WORDS)
    en_pool = list(EN_WORDS)
    out: list[tuple[str, list[list[Word]]]] = []

    for index in range(count):
        num_phrases = rng.choice([1, 1, 2, 2, 2, 3, 3, 4])
        phrases: list[list[Word]] = []
        for _ in range(num_phrases):
            words = [rng.choice(zh_pool) for _ in range(rng.choice([2, 3, 3, 4, 5]))]
            if index % 5 == 0:
                words.insert(rng.randrange(len(words) + 1), rng.choice(en_pool))
            phrases.append(words)
        text = "，".join("".join(w.text for w in phrase) for phrase in phrases) + "。"
        out.append((text, phrases))
    return out


def inject_duplicates(
    sentences: list[tuple[str, list[list[Word]]]],
    rng: random.Random,
    *,
    exact_fraction: float,
    near_fraction: float,
) -> tuple[list[tuple[str, list[list[Word]]]], int, int]:
    """Append exact copies and near-copies so leakage control is testable."""
    num_exact = int(len(sentences) * exact_fraction)
    num_near = int(len(sentences) * near_fraction)
    extra: list[tuple[str, list[list[Word]]]] = []

    for _ in range(num_exact):
        extra.append(sentences[rng.randrange(len(sentences))])

    for _ in range(num_near):
        _, phrases = sentences[rng.randrange(len(sentences))]
        # Near-duplicate: swap one word in one phrase, keeping the rest.
        mutated = [list(phrase) for phrase in phrases]
        phrase_index = rng.randrange(len(mutated))
        if len(mutated[phrase_index]) > 1:
            word_index = rng.randrange(len(mutated[phrase_index]))
            mutated[phrase_index][word_index] = rng.choice(list(ZH_WORDS))
            new_text = (
                "，".join("".join(w.text for w in phrase) for phrase in mutated) + "。"
            )
            extra.append((new_text, mutated))

    return sentences + extra, num_exact, len(extra) - num_exact


def report(samples: list[dict]) -> None:
    lengths = [len(s["tokens"]) for s in samples]
    visemes = Counter(t["labels"]["viseme_id"] for s in samples for t in s["tokens"])
    languages = Counter(t["language"] for s in samples for t in s["tokens"])
    texts = Counter(s["text"] for s in samples)
    print(f"sentences:        {len(samples):,}")
    print(f"phoneme tokens:   {sum(lengths):,}")
    print(
        f"seq len:          min={min(lengths)} "
        f"mean={sum(lengths) / len(lengths):.1f} max={max(lengths)}"
    )
    print(f"languages:        {dict(sorted(languages.items()))}")
    print(f"exact text dups:  {sum(c - 1 for c in texts.values() if c > 1):,}")
    missing = [v for v in range(18) if visemes.get(v, 0) == 0]
    if missing:
        print(f"WARNING: viseme classes with zero examples: {missing}")
    else:
        print(f"all 18 viseme classes covered (min count {min(visemes.values()):,})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--exact-duplicate-fraction", type=float, default=0.02)
    parser.add_argument("--near-duplicate-fraction", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "dev" / "corpus.jsonl")
    args = parser.parse_args()

    rng = random.Random(SEED)
    sentences = generate_sentences(args.num_samples, rng)
    sentences, num_exact, num_near = inject_duplicates(
        sentences,
        rng,
        exact_fraction=args.exact_duplicate_fraction,
        near_fraction=args.near_duplicate_fraction,
    )
    print(f"injected {num_exact} exact and {num_near} near duplicates")

    samples = [
        build_sample(f"dev_{index:06d}", text, phrases)
        for index, (text, phrases) in enumerate(sentences)
    ]
    report(samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
