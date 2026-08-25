#!/usr/bin/env python3
"""Generate the deterministic tiny-overfit dataset (docs/12 section 9).

Development tooling. Every label is illustrative
(``strength_source="pseudo_strength_v1"``) and is NOT Human Gold. This set
exists only to prove that
``Dataset -> Collator -> Model -> Loss -> Backward -> Optimizer`` is wired
correctly. It says nothing about generalisation.

Validation mirrors train on purpose: the tiny-overfit gate measures fitting
capacity, not held-out quality.

Usage:
    python scripts/make_tiny_overfit_data.py [--num-samples 64] [--out data/tiny]
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
from word_bank import EN_WORDS, ZH_WORDS, w

SEED = 20260820

# Sentences written out explicitly so the set covers the cases docs/12 asks
# for: numbers, mixed language, polyphone context, phrase boundaries.
def _anchor_sentences() -> list[tuple[str, list[list[Word]], str | None]]:
    return [
        ("你好。", [[w("你好")]], None),
        ("请稍等一下。", [[w("请"), w("稍等"), w("一下")]], None),
        (
            "预计2026年销售额增长12.5%。",
            [
                [w("预计"), w("二零二六"), w("年")],
                [w("销售额"), w("增长"), w("百分之"), w("十二"), w("点五")],
            ],
            "预计二零二六年销售额增长百分之十二点五。",
        ),
        ("We can move.", [[w("we"), w("can"), w("move")]], None),
        (
            "新的AI模型将在GPU服务器上运行。",
            [
                [w("新"), w("的"), w("AI"), w("模型")],
                [w("将"), w("在"), w("GPU"), w("服务器"), w("上"), w("运行")],
            ],
            None,
        ),
        # Polyphone context: 行 as hang2 (银行) vs xing2 (行走).
        ("这家银行正常营业。", [[w("银行"), w("正常"), w("营业")]], None),
        ("他们继续向前行走。", [[w("向前"), w("行走")]], None),
        # ü coverage.
        ("女士需求。", [[w("女士"), w("需求")]], None),
        ("绿色旅行。", [[w("绿色"), w("旅行")]], None),
        # English-heavy, gives ey / jh coverage.
        ("new AI key view", [[w("new"), w("AI")], [w("key"), w("view")]], None),
        ("move me up", [[w("move"), w("me"), w("up")]], None),
        (
            "GPU可以运行AI模型。",
            [[w("GPU")], [w("运行"), w("AI"), w("模型")]],
            None,
        ),
    ]


def _generated_sentences(count: int, rng: random.Random) -> list[tuple[str, list[list[Word]], None]]:
    """Combine bank words into phrase-structured sentences."""
    zh_pool = list(ZH_WORDS)
    en_pool = list(EN_WORDS)
    out: list[tuple[str, list[list[Word]], None]] = []

    for index in range(count):
        num_phrases = rng.choice([1, 1, 2, 2, 3])
        phrases: list[list[Word]] = []
        for _ in range(num_phrases):
            num_words = rng.choice([2, 3, 3, 4])
            words = [rng.choice(zh_pool) for _ in range(num_words)]
            # Every 4th sentence mixes in an English word so language
            # switching inside one sentence is covered.
            if index % 4 == 0:
                words.insert(rng.randrange(len(words) + 1), rng.choice(en_pool))
            phrases.append(words)
        text = "，".join("".join(word.text for word in phrase) for phrase in phrases) + "。"
        out.append((text, phrases, None))
    return out


def build_dataset(num_samples: int) -> list[dict]:
    rng = random.Random(SEED)
    anchors = _anchor_sentences()
    if num_samples < len(anchors):
        raise SystemExit(
            f"--num-samples must be >= {len(anchors)} so all anchor cases are kept"
        )
    generated = _generated_sentences(num_samples - len(anchors), rng)

    samples: list[dict] = []
    for index, (text, phrases, normalized) in enumerate([*anchors, *generated]):
        samples.append(
            build_sample(
                f"tiny_{index:03d}",
                text,
                phrases,
                normalized_text=normalized,
            )
        )
    return samples


def report(samples: list[dict]) -> None:
    lengths = [len(s["tokens"]) for s in samples]
    visemes = Counter(t["labels"]["viseme_id"] for s in samples for t in s["tokens"])
    languages = Counter(t["language"] for s in samples for t in s["tokens"])
    tones = Counter(t["surface_tone"] for s in samples for t in s["tokens"])
    roles = Counter(t["syllable_role"] for s in samples for t in s["tokens"])
    strengths = [t["labels"]["strength"] for s in samples for t in s["tokens"]]

    print(f"samples:        {len(samples)}")
    print(f"phoneme tokens: {sum(lengths)}")
    print(f"seq len:        min={min(lengths)} mean={sum(lengths)/len(lengths):.1f} max={max(lengths)}")
    print(f"languages:      {dict(sorted(languages.items()))}")
    print(f"surface_tone:   {dict(sorted(tones.items()))}")
    print(f"syllable_role:  {dict(sorted(roles.items()))}")
    print(f"strength:       min={min(strengths):.1f} mean={sum(strengths)/len(strengths):.1f} max={max(strengths):.1f}")
    print("viseme distribution:")
    for viseme_id in range(18):
        print(f"  {viseme_id:2d}: {visemes.get(viseme_id, 0)}")
    missing = [v for v in range(18) if visemes.get(v, 0) == 0]
    if missing:
        print(f"WARNING: viseme classes with zero examples: {missing}")
    else:
        print("all 18 viseme classes covered")


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "tiny")
    args = parser.parse_args()

    samples = build_dataset(args.num_samples)
    report(samples)
    write_jsonl(args.out / "train.jsonl", samples)
    # Deliberate mirror: the gate measures fitting capacity.
    write_jsonl(args.out / "validation.jsonl", samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
