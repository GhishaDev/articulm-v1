#!/usr/bin/env bash
# Losslessly flatten an ArticuLM JSONL split into two Parquet tables.
#
# Unlike scripts/to_parquet.sh (model fields only), this keeps EVERY field of
# the JSONL - teacher_metadata, timing_metadata, generation, normalization -
# so the original JSONL can be reconstructed byte-for-byte semantically.
#
# Output per split (in OUT_DIR):
#   <stem>.tokens.parquet   one row per phoneme token (34 columns, Nullable
#                           columns preserve JSON nulls)
#   <stem>.samples.parquet  one row per sentence (text + generation + ...,
#                           JSON-valued fields stored as raw JSON strings)
#
# Usage: scripts/to_parquet_lossless.sh data/v2/test.jsonl [out_dir]
set -euo pipefail

SRC="${1:?usage: to_parquet_lossless.sh <split.jsonl> [out_dir]}"
OUT_DIR="${2:-data/parquet_lossless}"
STEM="$(basename "$SRC" .jsonl)"
mkdir -p "$OUT_DIR"

# ---- token table: every per-token field, nulls preserved ----
clickhouse-local --query "
SELECT
  sample_id,
  token_index,
  JSONExtractString(tok, 'phoneme') AS phoneme,
  JSONExtractString(tok, 'language') AS language,
  JSONExtractInt(tok, 'surface_tone') AS surface_tone,
  JSONExtractInt(tok, 'stress') AS stress,
  if(JSONType(tok, 'syllable_role') = 'Null', NULL, JSONExtractString(tok, 'syllable_role')) AS syllable_role,
  if(JSONExtractString(tok,'articulatory','type')='','[NA]',JSONExtractString(tok,'articulatory','type')) AS articulatory_type,
  if(JSONExtractString(tok,'articulatory','height')='','[NA]',JSONExtractString(tok,'articulatory','height')) AS articulatory_height,
  if(JSONExtractString(tok,'articulatory','backness')='','[NA]',JSONExtractString(tok,'articulatory','backness')) AS articulatory_backness,
  if(JSONExtractString(tok,'articulatory','rounded')='','[NA]',JSONExtractString(tok,'articulatory','rounded')) AS articulatory_rounded,
  if(JSONExtractString(tok,'articulatory','place')='','[NA]',JSONExtractString(tok,'articulatory','place')) AS articulatory_place,
  if(JSONExtractString(tok,'articulatory','manner')='','[NA]',JSONExtractString(tok,'articulatory','manner')) AS articulatory_manner,
  if(JSONExtractString(tok,'articulatory','voiced')='','[NA]',JSONExtractString(tok,'articulatory','voiced')) AS articulatory_voiced,
  if(JSONExtractString(tok,'articulatory','aspirated')='','[NA]',JSONExtractString(tok,'articulatory','aspirated')) AS articulatory_aspirated,
  JSONExtractString(tok,'boundary','word_start') AS boundary_word_start,
  JSONExtractString(tok,'boundary','word_end') AS boundary_word_end,
  JSONExtractString(tok,'boundary','phrase_start') AS boundary_phrase_start,
  JSONExtractString(tok,'boundary','phrase_end') AS boundary_phrase_end,
  JSONExtractString(tok,'boundary','boundary_type') AS boundary_boundary_type,
  JSONExtractInt(tok,'labels','viseme_id') AS viseme_id,
  JSONExtractFloat(tok,'labels','strength') AS strength,
  JSONExtractString(tok,'labels','viseme_source') AS viseme_source,
  JSONExtractString(tok,'labels','strength_source') AS strength_source,
  JSONExtractString(tok,'teacher_metadata','shapeV2') AS teacher_shapeV2,
  JSONExtractString(tok,'teacher_metadata','shape') AS teacher_shape,
  if(JSONType(tok,'teacher_metadata','raw_value')='Null', NULL, JSONExtractInt(tok,'teacher_metadata','raw_value')) AS teacher_raw_value,
  JSONExtractString(tok,'teacher_metadata','raw_phoneme') AS teacher_raw_phoneme,
  JSONExtractInt(tok,'teacher_metadata','word_index') AS teacher_word_index,
  JSONExtractInt(tok,'teacher_metadata','char_index') AS teacher_char_index,
  JSONExtractFloat(tok,'timing_metadata','start_percent') AS timing_start_percent,
  JSONExtractFloat(tok,'timing_metadata','end_percent') AS timing_end_percent,
  if(JSONType(tok,'timing_metadata','duration_raw')='Null', NULL, JSONExtractFloat(tok,'timing_metadata','duration_raw')) AS timing_duration_raw,
  JSONExtractFloat(tok,'timing_metadata','duration_ms') AS timing_duration_ms
FROM (
  SELECT
    JSONExtractString(line, 'sample_id') AS sample_id,
    arrayEnumerate(JSONExtractArrayRaw(line, 'tokens')) AS _idx,
    JSONExtractArrayRaw(line, 'tokens') AS tok
  FROM file('$SRC', 'LineAsString')
)
ARRAY JOIN _idx AS token_index, tok
SETTINGS max_threads=1
" --output-format Parquet > "$OUT_DIR/$STEM.tokens.parquet" 2>> "$OUT_DIR/errors.log"

# ---- sample table: sentence-level fields ----
clickhouse-local --query "
SELECT
  JSONExtractString(line, 'sample_id') AS sample_id,
  JSONExtractString(line, 'batch_id') AS batch_id,
  JSONExtractString(line, 'schema_version') AS schema_version,
  JSONExtractString(line, 'text') AS text,
  JSONExtractString(line, 'original_text') AS original_text,
  JSONExtractString(line, 'text_normalized') AS text_normalized,
  JSONExtractInt(line, 'num_tokens') AS num_tokens,
  JSONExtractRaw(line, 'normalization') AS normalization,
  JSONExtractRaw(line, 'generation') AS generation,
  JSONExtractRaw(line, 'warnings') AS warnings
FROM file('$SRC', 'LineAsString')
SETTINGS max_threads=1
" --output-format Parquet > "$OUT_DIR/$STEM.samples.parquet" 2>> "$OUT_DIR/errors.log"

echo "$(basename "$SRC"): $(du -h "$OUT_DIR/$STEM.tokens.parquet" | cut -f1) + $(du -h "$OUT_DIR/$STEM.samples.parquet" | cut -f1) -> $OUT_DIR"
