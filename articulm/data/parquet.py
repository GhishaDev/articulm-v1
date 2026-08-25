"""Load ArticuLM training samples from lossless Parquet tables.

The tables are written by ``scripts/to_parquet_lossless.sh`` - one row per
phoneme token (``<split>.tokens.parquet``, every JSONL field preserved) plus
one row per sentence (``<split>.samples.parquet``). This module reconstructs
the original JSON record for each sentence and pipes it through the regular
:meth:`articulm.data.schema.parse_sample`, so validation and normalisation
semantics are byte-identical to loading the JSONL directly.

Reading goes through ``clickhouse-local`` (TSV on stdout) which keeps the
model repo free of a Python Parquet dependency. Pass the *tokens* parquet
path; the samples table is looked up next to it::

    load_samples_parquet("data/parquet_lossless/train.tokens.parquet", cfg)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import DataConfig
from .schema import Sample, SchemaError, parse_sample

# Only the columns that feed Sample construction; teacher/timing metadata are
# preserved in the parquet but ignored here (parse_sample never reads them).
TOKEN_COLUMNS = [
    "sample_id", "token_index", "phoneme", "language", "surface_tone", "stress",
    "syllable_role",
    "articulatory_type", "articulatory_height", "articulatory_backness",
    "articulatory_rounded", "articulatory_place", "articulatory_manner",
    "articulatory_voiced", "articulatory_aspirated",
    "boundary_word_start", "boundary_word_end", "boundary_phrase_start",
    "boundary_phrase_end", "boundary_boundary_type",
    "viseme_id", "strength", "viseme_source", "strength_source",
]

SAMPLE_COLUMNS = [
    "sample_id", "batch_id", "schema_version", "text", "original_text",
    "text_normalized", "num_tokens", "normalization", "generation", "warnings",
]

_COL = {c: i for i, c in enumerate(TOKEN_COLUMNS)}
_SCOL = {c: i for i, c in enumerate(SAMPLE_COLUMNS)}

# String-or-null articulatory fields (parquet stores "[NA]" for JSON null).
_ARTICULATORY_STR = [
    ("type", "articulatory_type"), ("height", "articulatory_height"),
    ("backness", "articulatory_backness"),
    ("place", "articulatory_place"), ("manner", "articulatory_manner"),
]
# Bool-or-null articulatory fields.
_ARTICULATORY_BOOL = [
    ("rounded", "articulatory_rounded"),
    ("voiced", "articulatory_voiced"), ("aspirated", "articulatory_aspirated"),
]
_BOUNDARY_BOOL = [
    ("word_start", "boundary_word_start"), ("word_end", "boundary_word_end"),
    ("phrase_start", "boundary_phrase_start"), ("phrase_end", "boundary_phrase_end"),
]

_TSV_ESCAPES = {
    "\\\\": "\\", "\\t": "\t", "\\n": "\n", "\\r": "\r",
    "\\b": "\b", "\\f": "\f", "\\0": "\0",
    "\\'": "'", '\\"': '"', "\\`": "`", "\\=": "=",
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


def _iter_tsv(path: Path, columns: list[str]):
    """Stream a parquet file as unescaped TSV rows via clickhouse-local."""
    query = (
        f"SELECT {', '.join(columns)} FROM file('{path}', 'Parquet') "
        f"FORMAT TSV SETTINGS max_threads=1"
    )
    proc = subprocess.Popen(
        ["clickhouse-local", "--query", query],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(columns):
                continue
            # A bare \N is NULL; escaped literals appear as \\N and survive.
            yield ["\\N" if f == "\\N" else _tsv_unescape(f) for f in fields]
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()


def _nullable_int(v: str):
    return None if v == "\\N" else int(v)


def _nullable_float(v: str):
    return None if v == "\\N" else float(v)


def _raw_token(f: list[str]) -> dict:
    """Rebuild the original per-token JSON dict (teacher/timing included)."""
    def s(col):
        return f[_COL[col]]

    def na_str(col):
        v = f[_COL[col]]
        return None if v == "[NA]" else v

    def na_bool(col):
        v = f[_COL[col]]
        if v == "[NA]":
            return None
        return v in ("true", "1")

    def b(col):
        return f[_COL[col]] in ("true", "1")

    return {
        "phoneme": s("phoneme"),
        "language": s("language"),
        "surface_tone": int(s("surface_tone")),
        "stress": int(s("stress")),
        "syllable_role": _nullable_int_tolerant(s("syllable_role")),
        "articulatory": {**{k: na_str(col) for k, col in _ARTICULATORY_STR},
                         **{k: na_bool(col) for k, col in _ARTICULATORY_BOOL}},
        "boundary": {**{k: b(col) for k, col in _BOUNDARY_BOOL},
                     "boundary_type": s("boundary_boundary_type")},
        "labels": {
            "viseme_id": int(s("viseme_id")),
            "strength": float(s("strength")),
            "viseme_source": s("viseme_source"),
            "strength_source": s("strength_source"),
        },
    }


def _nullable_int_tolerant(v: str):
    """syllable_role round-trips as a string column; NULL stays NULL."""
    return None if v == "\\N" else v


def _raw_json(col_value: str, default):
    """JSON-valued column; absent fields extract as '' and use the default."""
    return json.loads(col_value) if col_value.strip() else default


def _raw_sample(f: list[str]) -> dict:
    """Rebuild the sentence-level JSON dict (only what parse_sample reads)."""
    def s(col):
        return f[_SCOL[col]]

    return {
        "schema_version": s("schema_version"),
        "sample_id": s("sample_id"),
        "text": s("text"),
        "normalized_text": None,
        "normalization": _raw_json(s("normalization"), []),
        "generation": _raw_json(s("generation"), {}),
        "warnings": _raw_json(s("warnings"), []),
    }


def _samples_parquet_path(tokens_path: Path) -> Path:
    """``train.tokens.parquet`` -> ``train.samples.parquet``."""
    name = tokens_path.name
    if not name.endswith(".tokens.parquet"):
        raise SchemaError(
            f"expected a '<split>.tokens.parquet' path, got {tokens_path}"
        )
    return tokens_path.with_name(name[: -len(".tokens.parquet")] + ".samples.parquet")


def load_samples_parquet(
    tokens_path: str | Path,
    cfg: DataConfig,
    *,
    require_labels: bool = True,
    limit: int | None = None,
) -> list[Sample]:
    """Load samples from lossless parquet tables, matching ``load_samples``.

    ``limit`` caps sentences, mirroring the JSONL loader. Every reconstructed
    record goes through :func:`parse_sample`, so schema validation and
    normalisation are identical to the JSONL path.
    """
    tokens_path = Path(tokens_path)
    if not tokens_path.is_file():
        raise SchemaError(f"parquet file not found: {tokens_path}")
    samples_path = _samples_parquet_path(tokens_path)
    if not samples_path.is_file():
        raise SchemaError(
            f"samples table not found next to {tokens_path.name}: "
            f"expected {samples_path.name} (written by to_parquet_lossless.sh)"
        )

    # Sentence metadata, keyed by sample_id.
    meta: dict[str, dict] = {}
    for f in _iter_tsv(samples_path, SAMPLE_COLUMNS):
        record = _raw_sample(f)
        meta[record["sample_id"]] = record

    # Tokens stream in file order; group by consecutive sample_id.
    samples: list[Sample] = []
    current_id: str | None = None
    tokens: list[dict] = []

    def flush() -> None:
        nonlocal current_id, tokens
        if current_id is None:
            return
        record = dict(meta.get(current_id, {"sample_id": current_id}))
        record["tokens"] = tokens
        samples.append(
            parse_sample(
                record, cfg,
                require_labels=require_labels,
                location=str(tokens_path),
            )
        )
        current_id, tokens = None, []

    for f in _iter_tsv(tokens_path, TOKEN_COLUMNS):
        sid = f[_COL["sample_id"]]
        if sid != current_id:
            flush()
            if limit is not None and len(samples) >= limit:
                break
            current_id = sid
            tokens = []
        tokens.append(_raw_token(f))
    flush()

    if not samples:
        raise SchemaError(f"{tokens_path}: no samples found")
    return samples
