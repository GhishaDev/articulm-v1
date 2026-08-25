#!/usr/bin/env bash
# Flatten an ArticuLM JSONL split into a columnar Parquet table via clickhouse-local.
# Usage: scripts/to_parquet.sh data/v2/test.jsonl [out_dir]
set -euo pipefail

SRC="${1:?usage: to_parquet.sh <split.jsonl> [out_dir]}"
OUT_DIR="${2:-data/parquet}"
STEM="$(basename "$SRC" .jsonl)"
mkdir -p "$OUT_DIR"
DEST="$OUT_DIR/$STEM.tokens.parquet"

clickhouse-local --query "
SELECT
  JSONExtractString(line, 'sample_id') AS sample_id,
  JSONExtractString(tok, 'phoneme') AS phoneme,
  JSONExtractString(tok, 'language') AS language,
  JSONExtractInt(tok, 'surface_tone') AS surface_tone,
  JSONExtractInt(tok, 'stress') AS stress,
  if(JSONExtractString(tok,'syllable_role')='','other',JSONExtractString(tok,'syllable_role')) AS syllable_role,
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
  JSONExtractString(tok,'labels','strength_source') AS strength_source
FROM file('$SRC', 'LineAsString')
ARRAY JOIN JSONExtractArrayRaw(line, 'tokens') AS tok
SETTINGS max_threads=1
" --output-format Parquet > "$DEST" 2>> "$OUT_DIR/errors.log"

echo "$(basename "$SRC"): $(du -h "$DEST" | cut -f1) -> $DEST"
