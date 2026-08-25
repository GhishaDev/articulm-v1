#!/usr/bin/env bash
# Boot-time GPU recovery + training restart for ArticuLM v2 run.
# Installed as a user-crontab @reboot job (no root needed).
#
# Background (2026-08-22): GPU 02:00.0 fell off the bus (Xid 79), which corrupts
# CUDA device selection - both numeric and PCI-address CUDA_VISIBLE_DEVICES pins
# landed on the wrong physical card (verified via kernel Xid serials). Serial
# 0328205007200 (PCI 82:00.0) is the only card with a clean history (3h v1 run).
#
# Strategy: after boot, find that card BY SERIAL, pin by UUID, run a sustained
# load and VERIFY the load physically lands on the right serial via nvidia-smi
# before launching training. All decisions logged.
set -u

REPO=/home/work/ArticuLM_V1_Model
PY=$REPO/.venv/bin/python
LOGDIR=$REPO/logs/gpu_recovery
FLAG_DONE=$LOGDIR/recovery_done
LOG=$LOGDIR/recovery_$(date '+%Y%m%d_%H%M%S').log
TARGET_SERIAL=0328205007200

mkdir -p "$LOGDIR"
exec >>"$LOG" 2>&1
echo "=== gpu recovery started $(date) ==="

if [[ -f $FLAG_DONE ]]; then
  echo "recovery already done, exiting"; exit 0
fi

# ---- 1. wait for driver to settle (up to 10 min) ----
for i in $(seq 1 60); do
  if nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader >/dev/null 2>&1; then break; fi
  sleep 10
done
nvidia-smi -i 2 --query-gpu=uuid --format=csv,noheader || { echo "driver never became ready"; exit 1; }

# ---- 2. find target card by serial -> PCI bus -> UUID ----
# Fleet-wide nvidia-smi fails when a card is dead (handle error kills the whole
# query), so enumerate card-by-card and tolerate individual failures.
CARD_LINE=""
for i in 0 1 2 3 4 5 6 7; do
  line=$(nvidia-smi -i "$i" --query-gpu=uuid,pci.bus_id,serial --format=csv,noheader 2>/dev/null) || continue
  [[ -z $line ]] && continue
  echo "gpu[$i]: $line"
  if echo "$line" | grep -qi "$TARGET_SERIAL"; then CARD_LINE="$line"; fi
done
echo "target card line: $CARD_LINE"
if [[ -z $CARD_LINE ]]; then
  echo "FATAL: serial $TARGET_SERIAL not found in enumeration - card not visible"
  exit 1
fi
UUID=$(echo "$CARD_LINE" | cut -d, -f1 | tr -d ' ')
PCI=$(echo "$CARD_LINE" | cut -d, -f2 | tr -d ' ')
echo "target: serial=$TARGET_SERIAL pci=$PCI uuid=$UUID"

# ---- 3. sustained load pinned by UUID + physical placement verification ----
echo "--- sustained load test (UUID pin) ---"
CUDA_VISIBLE_DEVICES="$UUID" $PY - <<'EOF' &
import torch, time
x = torch.randn(8192, 8192, device='cuda')          # 256 MB
w = torch.randn(8192, 8192, device='cuda')
for i in range(600):                                 # ~45 s of matmuls
    x = (x @ w) * 1e-6 + x
    if i % 50 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
time.sleep(30)
print("LOAD_TEST_OK")
EOF
LOAD_PID=$!

PLACED=""
for i in $(seq 1 12); do
  sleep 5
  HIT=""
  for j in 0 1 2 3 4 5 6 7; do
    row=$(nvidia-smi -i "$j" --query-gpu=serial,memory.used --format=csv,noheader 2>/dev/null) || continue
    [[ -z $row ]] && continue
    if echo "$row" | grep -qi "$TARGET_SERIAL"; then
      mem=$(echo "$row" | cut -d, -f2 | tr -d ' ')
      if [[ ${mem:-0} -gt 200 ]]; then PLACED="$row"; break 2; fi
    fi
  done
done
echo "placement check: ${PLACED:-NOT_FOUND_ON_TARGET}"
if ! wait $LOAD_PID; then
  echo "FATAL: load test on target card failed (ECC?)"; exit 1
fi
if [[ -z $PLACED ]]; then
  echo "FATAL: UUID pin did not physically land on $TARGET_SERIAL - driver mapping still broken; NOT starting training"
  exit 1
fi
echo "target card passed sustained load with verified placement"

# ---- 4. launch training pinned by UUID ----
echo "--- launching v2 training ---"
cd $REPO
CUDA_VISIBLE_DEVICES="$UUID" nohup setsid $PY -m articulm.train \
    --config config/train_v1_50m_strength_v2.yaml \
    --run-dir runs/baseline_v2_strength --device cuda \
    >> runs/baseline_v2_strength_train.log 2>&1 &
echo "training launched (pid $!) pinned to $UUID"
sleep 5
touch $FLAG_DONE
echo "=== recovery complete $(date) ==="
