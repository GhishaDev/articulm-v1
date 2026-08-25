"""Verify lossless Parquet round-trip: reconstruct the JSONL and deep-compare.

Reads the two tables written by scripts/to_parquet_lossless.sh via
clickhouse-local (TSV), rebuilds the original sample dicts, and compares them
against the source JSONL with exact JSON semantics (nulls preserved, float
equality on the parsed doubles, JSON-valued fields re-parsed).

Usage:
    # verify round-trip against the source JSONL
    python scripts/verify_parquet_lossless.py data/v2/test.jsonl data/parquet_lossless

    # or reconstruct a JSONL from the parquet tables (the "decompress" path)
    python scripts/verify_parquet_lossless.py data/v2/test.jsonl data/parquet_lossless \
        --write-jsonl /tmp/test.rebuilt.jsonl
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

TOKEN_COLUMNS = [
    "sample_id", "token_index", "phoneme", "language", "surface_tone", "stress",
    "syllable_role",
    "articulatory_type", "articulatory_height", "articulatory_backness",
    "articulatory_rounded", "articulatory_place", "articulatory_manner",
    "articulatory_voiced", "articulatory_aspirated",
    "boundary_word_start", "boundary_word_end", "boundary_phrase_start",
    "boundary_phrase_end", "boundary_boundary_type",
    "viseme_id", "strength", "viseme_source", "strength_source",
    "teacher_shapeV2", "teacher_shape", "teacher_raw_value",
    "teacher_raw_phoneme", "teacher_word_index", "teacher_char_index",
    "timing_start_percent", "timing_end_percent", "timing_duration_raw",
    "timing_duration_ms",
]

SAMPLE_COLUMNS = [
    "sample_id", "batch_id", "schema_version", "text", "original_text",
    "text_normalized", "num_tokens", "normalization", "generation", "warnings",
]

# String-or-null articulatory fields.
ARTICULATORY_STR = [
    ("type", "articulatory_type"), ("height", "articulatory_height"),
    ("backness", "articulatory_backness"),
    ("place", "articulatory_place"), ("manner", "articulatory_manner"),
]
# Bool-or-null articulatory fields (JSON true/false, null for N/A).
ARTICULATORY_BOOL = [
    ("rounded", "articulatory_rounded"),
    ("voiced", "articulatory_voiced"), ("aspirated", "articulatory_aspirated"),
]
BOUNDARY_BOOL = [
    ("word_start", "boundary_word_start"), ("word_end", "boundary_word_end"),
    ("phrase_start", "boundary_phrase_start"), ("phrase_end", "boundary_phrase_end"),
]


_TSV_ESCAPES = {
    "\\\\": "\\",
    "\\t": "\t",
    "\\n": "\n",
    "\\r": "\r",
    "\\b": "\b",
    "\\f": "\f",
    "\\0": "\0",
    "\\'": "'",
    '\\"': '"',
    "\\`": "`",
    "\\=": "=",
}


def _tsv_unescape(s: str) -> str:
    if "\\" not in s:
        return s
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i : i + 2] in _TSV_ESCAPES:
            out.append(_TSV_ESCAPES[s[i : i + 2]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _tsv(path: Path, columns: list[str]):
    query = f"SELECT {', '.join(columns)} FROM file('{path}', 'Parquet') FORMAT TSV"
    proc = subprocess.run(
        ["clickhouse-local", "--query", query],
        capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        # NULL is a bare \N; escaped literals appear as \\N and survive.
        yield [(_tsv_unescape(f) if f != "\\N" else f) for f in line.split("\t")]


def _nullable_str(v: str):
    return None if v == "\\N" else v


def _nullable_int(v: str):
    return None if v == "\\N" else int(v)


def _nullable_float(v: str):
    return None if v == "\\N" else float(v)


def _json_bool(v: str):
    """"true"/"false" (String column) or "1"/"0" (Bool column) -> bool."""
    return v in ("true", "1")


def _reconstruct_token(f: list[str]) -> dict:
    def acol_str(name):
        v = f[TOKEN_COLUMNS.index(name)]
        return None if v == "[NA]" else v

    def acol_bool(name):
        v = f[TOKEN_COLUMNS.index(name)]
        if v == "[NA]":
            return None
        return v in ("true", "1")

    def bcol(name):
        return f[TOKEN_COLUMNS.index(name)] in ("true", "1")

    return {
        "phoneme": f[TOKEN_COLUMNS.index("phoneme")],
        "language": f[TOKEN_COLUMNS.index("language")],
        "surface_tone": int(f[TOKEN_COLUMNS.index("surface_tone")]),
        "stress": int(f[TOKEN_COLUMNS.index("stress")]),
        "syllable_role": _nullable_str(f[TOKEN_COLUMNS.index("syllable_role")]),
        "articulatory": {**{k: acol_str(col) for k, col in ARTICULATORY_STR},
                         **{k: acol_bool(col) for k, col in ARTICULATORY_BOOL}},
        "boundary": {**{k: bcol(col) for k, col in BOUNDARY_BOOL},
                     "boundary_type": f[TOKEN_COLUMNS.index("boundary_boundary_type")]},
        "labels": {
            "viseme_id": int(f[TOKEN_COLUMNS.index("viseme_id")]),
            "strength": float(f[TOKEN_COLUMNS.index("strength")]),
            "viseme_source": f[TOKEN_COLUMNS.index("viseme_source")],
            "strength_source": f[TOKEN_COLUMNS.index("strength_source")],
        },
        "teacher_metadata": {
            "shapeV2": f[TOKEN_COLUMNS.index("teacher_shapeV2")],
            "shape": f[TOKEN_COLUMNS.index("teacher_shape")],
            "raw_value": _nullable_int(f[TOKEN_COLUMNS.index("teacher_raw_value")]),
            "raw_phoneme": f[TOKEN_COLUMNS.index("teacher_raw_phoneme")],
            "word_index": int(f[TOKEN_COLUMNS.index("teacher_word_index")]),
            "char_index": int(f[TOKEN_COLUMNS.index("teacher_char_index")]),
        },
        "timing_metadata": {
            "start_percent": float(f[TOKEN_COLUMNS.index("timing_start_percent")]),
            "end_percent": float(f[TOKEN_COLUMNS.index("timing_end_percent")]),
            "duration_raw": _nullable_float(f[TOKEN_COLUMNS.index("timing_duration_raw")]),
            "duration_ms": float(f[TOKEN_COLUMNS.index("timing_duration_ms")]),
        },
    }


def reconstruct(tokens_parquet: Path, samples_parquet: Path) -> list[dict]:
    tokens_by_sample: dict[str, list[tuple[int, dict]]] = {}
    for f in _tsv(tokens_parquet, TOKEN_COLUMNS):
        sid = f[0]
        idx = int(f[1])
        tokens_by_sample.setdefault(sid, []).append((idx, _reconstruct_token(f)))

    samples = []
    for f in _tsv(samples_parquet, SAMPLE_COLUMNS):
        sid = f[SAMPLE_COLUMNS.index("sample_id")]
        tokens = [t for _, t in sorted(tokens_by_sample.get(sid, []))]
        samples.append({
            "schema_version": f[SAMPLE_COLUMNS.index("schema_version")],
            "sample_id": sid,
            "batch_id": f[SAMPLE_COLUMNS.index("batch_id")],
            "text": f[SAMPLE_COLUMNS.index("text")],
            "original_text": f[SAMPLE_COLUMNS.index("original_text")],
            "text_normalized": _json_bool(f[SAMPLE_COLUMNS.index("text_normalized")]),
            "normalization": json.loads(f[SAMPLE_COLUMNS.index("normalization")]),
            "num_tokens": int(f[SAMPLE_COLUMNS.index("num_tokens")]),
            "generation": json.loads(f[SAMPLE_COLUMNS.index("generation")]),
            "warnings": json.loads(f[SAMPLE_COLUMNS.index("warnings")]),
            "tokens": tokens,
        })
    return samples


def deep_equal(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (a != a and b != b)  # NaN-safe
    return a == b


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write_jsonl = None
    for a in sys.argv[1:]:
        if a.startswith("--write-jsonl="):
            write_jsonl = a.split("=", 1)[1]
        elif a == "--write-jsonl":
            write_jsonl = None  # value form handled below
    if "--write-jsonl" in sys.argv:
        write_jsonl = sys.argv[sys.argv.index("--write-jsonl") + 1]

    src = Path(args[0])
    out_dir = Path(args[1])
    stem = src.stem
    tokens_parquet = out_dir / f"{stem}.tokens.parquet"
    samples_parquet = out_dir / f"{stem}.samples.parquet"

    original = [json.loads(line) for line in src.open(encoding="utf-8")]
    rebuilt = reconstruct(tokens_parquet, samples_parquet)

    if write_jsonl:
        with open(write_jsonl, "w", encoding="utf-8") as fh:
            for sample in rebuilt:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"重建 JSONL -> {write_jsonl} ({len(rebuilt)} 句)")

    if len(original) != len(rebuilt):
        print(f"FAIL: 句数不一致 {len(original)} vs {len(rebuilt)}")
        return 1

    mismatch = 0
    for i, (a, b) in enumerate(zip(original, rebuilt)):
        if not deep_equal(a, b):
            mismatch += 1
            if mismatch <= 3:
                for k in a:
                    if k not in b or not deep_equal(a[k], b[k]):
                        print(f"  样本 {a.get('sample_id')} 字段 {k!r} 不一致")
    print(f"{src.name}: {len(original)} 句, 不一致 {mismatch}")
    print("✅ 无损往返一致" if mismatch == 0 else "❌ 存在差异")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
