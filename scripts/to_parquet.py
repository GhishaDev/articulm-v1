"""Flatten the ArticuLM JSONL training data into columnar Parquet.

The training corpus is 5GB of JSONL; flattened to one row per phoneme token it
collapses to ~150MB (dictionary encoding + zstd), which is far friendlier for
sharing / GitHub. Two tables are emitted per split:

  <out>/<split>.tokens.parquet   - one row per phoneme token (the encoder input)
  <out>/<split>.samples.parquet  - one row per sentence (sample_id + text + meta)

Usage:

    python scripts/to_parquet.py \
        --train data/v2/train.jsonl \
        --validation data/v2/validation.jsonl \
        --test data/v2/test.jsonl \
        --out data/parquet

Requires pyarrow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Per-token columns in a stable order. All but `strength` are low-cardinality
# categoricals (see feature_vocab.json), which Parquet dictionary-encodes to a
# one-byte index each.
TOKEN_COLUMNS = [
    "sample_id",
    "token_index",
    "phoneme",
    "language",
    "surface_tone",
    "stress",
    "syllable_role",
    "articulatory.type",
    "articulatory.height",
    "articulatory.backness",
    "articulatory.rounded",
    "articulatory.place",
    "articulatory.manner",
    "articulatory.voiced",
    "articulatory.aspirated",
    "boundary.word_start",
    "boundary.word_end",
    "boundary.phrase_start",
    "boundary.phrase_end",
    "boundary.boundary_type",
    "viseme_id",
    "strength",
    "viseme_source",
    "strength_source",
]

SAMPLE_COLUMNS = [
    "sample_id",
    "text",
    "original_text",
    "batch_id",
    "num_tokens",
]


def _token_rows(sample: dict) -> list[tuple]:
    sample_id = sample["sample_id"]
    rows = []
    for i, t in enumerate(sample["tokens"]):
        a = t["articulatory"]
        b = t["boundary"]
        lab = t["labels"]
        rows.append(
            (
                sample_id,
                i,
                t["phoneme"],
                t["language"],
                t["surface_tone"],
                t["stress"],
                t["syllable_role"],
                a["type"], a["height"], a["backness"], a["rounded"],
                a["place"], a["manner"], a["voiced"], a["aspirated"],
                b["word_start"], b["word_end"], b["phrase_start"],
                b["phrase_end"], b["boundary_type"],
                lab["viseme_id"],
                float(lab["strength"]),
                lab["viseme_source"],
                lab["strength_source"],
            )
        )
    return rows


def _write_table(path: Path, columns: list[str], columnar: dict[str, list]) -> None:
    schema = pa.schema([(c, pa.string()) for c in columns if c != "strength"]
                       + [("strength", pa.float32())])
    arrays = []
    for c in columns:
        if c == "strength":
            arrays.append(pa.array(columnar[c], type=pa.float32()))
        else:
            arrays.append(pa.array(columnar[c], type=pa.string()))
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path, compression="zstd", row_group_size=1_000_000)


def convert(source: Path, out_dir: Path) -> None:
    token_cols = {c: [] for c in TOKEN_COLUMNS}
    sample_cols = {c: [] for c in SAMPLE_COLUMNS}

    with source.open(encoding="utf-8") as fh:
        for line in fh:
            sample = json.loads(line)
            for row in _token_rows(sample):
                for col, val in zip(TOKEN_COLUMNS, row, strict=True):
                    token_cols[col].append(val)
            sample_cols["sample_id"].append(sample["sample_id"])
            sample_cols["text"].append(sample.get("text", ""))
            sample_cols["original_text"].append(sample.get("original_text") or sample.get("text", ""))
            sample_cols["batch_id"].append(sample.get("batch_id") or "")
            sample_cols["num_tokens"].append(str(sample.get("num_tokens", len(sample["tokens"]))))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem  # e.g. "train" from train.jsonl
    _write_table(out_dir / f"{stem}.tokens.parquet", TOKEN_COLUMNS, token_cols)
    _write_table(out_dir / f"{stem}.samples.parquet", SAMPLE_COLUMNS, sample_cols)
    print(f"{source.name}: {len(sample_cols['sample_id'])} 句 / "
          f"{len(token_cols['sample_id'])} tokens -> {out_dir / f'{stem}.tokens.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="train.jsonl")
    parser.add_argument("--validation", type=Path, help="validation.jsonl")
    parser.add_argument("--test", type=Path, help="test.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/parquet"))
    args = parser.parse_args()

    for path in (args.train, args.validation, args.test):
        if path:
            convert(path, args.out)


if __name__ == "__main__":
    main()
