#!/usr/bin/env python3
"""Regenerate tests/fixtures/*.jsonl.

Development tooling. Labels are illustrative
(``strength_source="pseudo_strength_v1"``), never Human Gold.

Usage:
    python scripts/build_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fixture_corpus import build_sample, doc_canonical_zh_nihao
from word_bank import w

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def zh_samples() -> list[dict]:
    return [
        # Verbatim from docs/12_training_sample_examples.md.
        doc_canonical_zh_nihao(),
        build_sample(
            "zh_qing_shaodeng_001",
            "请稍等一下。",
            [[w("请"), w("稍等"), w("一下")]],
        ),
        build_sample(
            "zh_forecast_2026_001",
            "预计2026年销售额增长12.5%。",
            [
                [w("预计"), w("二零二六"), w("年")],
                [w("销售额"), w("增长"), w("百分之"), w("十二"), w("点五")],
            ],
            normalized_text="预计二零二六年销售额增长百分之十二点五。",
        ),
    ]


def en_samples() -> list[dict]:
    return [
        build_sample("en_we_can_move_001", "We can move.", [[w("we"), w("can"), w("move")]]),
    ]


def mixed_samples() -> list[dict]:
    return [
        build_sample(
            "mixed_ai_gpu_001",
            "新的AI模型将在GPU服务器上运行。",
            [
                [w("新"), w("的"), w("AI"), w("模型")],
                [w("将"), w("在"), w("GPU"), w("服务器"), w("上"), w("运行")],
            ],
        ),
    ]


def write_jsonl(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tokens = sum(len(s["tokens"]) for s in samples)
    print(f"wrote {path.relative_to(REPO_ROOT)}: {len(samples)} samples, {tokens} tokens")


def main() -> int:
    write_jsonl(FIXTURE_DIR / "sample_zh.jsonl", zh_samples())
    write_jsonl(FIXTURE_DIR / "sample_en.jsonl", en_samples())
    write_jsonl(FIXTURE_DIR / "sample_mixed.jsonl", mixed_samples())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
