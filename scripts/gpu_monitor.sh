#!/usr/bin/env bash
# GPU usage sampler for the baseline_gpu training run.
# Appends one CSV row every interval to runs/baseline_gpu/logs/gpu_usage.csv
# so utilization/memory/power/temperature can be folded into the training report.
#
# Queries each GPU individually (-i N) so one failed card (e.g. GPU 0's device
# handle error, which makes whole-fleet queries exit 255) does not kill the
# sampling of healthy cards. Failed queries append an empty-field row for that
# GPU, keeping the timestamp cadence visible in the CSV.
set -u

RUN_DIR="${1:-runs/baseline_gpu}"
INTERVAL="${2:-30}"
OUT="$RUN_DIR/logs/gpu_usage.csv"

mkdir -p "$RUN_DIR/logs"
if [[ ! -s "$OUT" ]]; then
  echo "timestamp,gpu_index,utilization_pct,memory_used_mb,memory_total_mb,power_w,temp_c" > "$OUT"
fi

while true; do
  now="$(date '+%Y-%m-%dT%H:%M:%S')"
  for i in 0 1 2 3; do
    line="$(nvidia-smi -i "$i" --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | grep -E '^[0-9]' \
      | awk -F', *' -v OFS=',' '{print $1, $2, $3, $4, $5; exit}')"
    if [[ -n "${line}" ]]; then
      echo "${now},${i},${line}" >> "$OUT"
    else
      echo "${now},${i},,,,,," >> "$OUT"
    fi
  done
  sleep "$INTERVAL"
done
