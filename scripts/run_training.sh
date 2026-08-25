#!/usr/bin/env bash
#
# ArticuLM-V1 training launcher.
#
# Chains the documented workflow (docs/11_training_operations.md) in the order
# that the project's non-negotiable constraints require:
#
#   0. environment / GPU / precision preflight
#   1. dataset validation (fail fast on schema or label violations)
#   2. tiny-overfit gate  ---- MANDATORY, aborts the run if it fails
#   3. pre-flight report for the real config (no training)
#   4. the real training run
#
# The gate is not advisory. Stage 2 verifies the gate's *metrics* via
# `python -m articulm.gate`, not merely that the command exited 0. Skipping it
# requires an explicit --skip-gate and prints a loud warning.
#
# Usage:
#   scripts/run_training.sh                                   # full default flow
#   scripts/run_training.sh --config config/train_v1_50m.yaml
#   scripts/run_training.sh --resume runs/.../checkpoints/last.pt
#   scripts/run_training.sh --dry-run                         # stop before stage 4
#   scripts/run_training.sh --background                      # detach stage 4
#   scripts/run_training.sh --stage train                     # run one stage only
#
# Exit codes: 0 ok | 1 a stage failed | 2 bad usage | 3 gate failed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

TRAIN_CONFIG="config/train_v1_50m.yaml"
GATE_CONFIG="config/train_tiny_overfit.yaml"
RESUME=""
DEVICE=""
STAGE="all"
SKIP_GATE=0
DRY_RUN=0
BACKGROUND=0
GATE_MIN_ACCURACY="0.95"
GATE_MAX_MAE="5.0"
EXTRA_TRAIN_ARGS=()

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/launcher/${TIMESTAMP}"

usage() {
    # Print the leading comment block only: stop at the first non-comment line.
    awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Options:
  --config PATH          training config (default config/train_v1_50m.yaml)
  --gate-config PATH     tiny-overfit config (default config/train_tiny_overfit.yaml)
  --resume PATH          checkpoint to resume the real run from
  --device DEV           force cuda / cpu / mps
  --stage NAME           all | preflight | validate | gate | dryrun | train
  --skip-gate            skip the mandatory tiny-overfit gate (discouraged)
  --gate-min-accuracy F  gate viseme-accuracy threshold (default 0.95)
  --gate-max-mae F       gate strength-MAE threshold, 0..100 units (default 5.0)
  --dry-run              run stages 0-3 and stop before real training
  --background           detach the real training run with nohup
  --log-dir PATH         launcher log directory (default runs/launcher/<ts>)
  -- ARGS...             pass everything after -- to `python -m articulm.train`
  -h, --help             this help
EOF
}

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '\n%s\n== %s\n%s\n' "$(printf '=%.0s' {1..72})" "$*" "$(printf '=%.0s' {1..72})"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             TRAIN_CONFIG="${2:?}"; shift 2 ;;
        --gate-config)        GATE_CONFIG="${2:?}"; shift 2 ;;
        --resume)             RESUME="${2:?}"; shift 2 ;;
        --device)             DEVICE="${2:?}"; shift 2 ;;
        --stage)              STAGE="${2:?}"; shift 2 ;;
        --gate-min-accuracy)  GATE_MIN_ACCURACY="${2:?}"; shift 2 ;;
        --gate-max-mae)       GATE_MAX_MAE="${2:?}"; shift 2 ;;
        --log-dir)            LOG_DIR="${2:?}"; shift 2 ;;
        --skip-gate)          SKIP_GATE=1; shift ;;
        --dry-run)            DRY_RUN=1; shift ;;
        --background)         BACKGROUND=1; shift ;;
        -h|--help)            usage; exit 0 ;;
        --)                   shift; EXTRA_TRAIN_ARGS=("$@"); break ;;
        *)                    usage; die "unknown option: $1" 2 ;;
    esac
done

case "$STAGE" in
    all|preflight|validate|gate|dryrun|train) ;;
    *) die "unknown --stage: $STAGE (expected all|preflight|validate|gate|dryrun|train)" 2 ;;
esac

[[ -f "$TRAIN_CONFIG" ]] || die "training config not found: $TRAIN_CONFIG" 2
mkdir -p "$LOG_DIR"

run_stage() {
    # Should this stage run, given --stage?
    [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

# --------------------------------------------------------------------------
# Stage 0 — environment preflight
# --------------------------------------------------------------------------
if run_stage preflight; then
    step "Stage 0/4 — environment preflight"
    log "repo:           $REPO_ROOT"
    log "python:         $($PYTHON -V 2>&1)"
    log "train config:   $TRAIN_CONFIG"
    log "launcher logs:  $LOG_DIR"

    $PYTHON - "$TRAIN_CONFIG" <<'PY' | tee "$LOG_DIR/preflight.txt"
import sys
import torch
from articulm.config import load_train_config
from articulm.runtime import describe_hardware, resolve_device, resolve_precision

cfg = load_train_config(sys.argv[1])
hw = describe_hardware(resolve_device())
plan = resolve_precision(cfg.training.precision, hw)

print(f"torch:          {torch.__version__}  cuda={torch.version.cuda}")
print(f"device:         {hw.device} ({hw.device_name})")
print(f"capability:     {hw.as_dict()['compute_capability']}")
print(f"device memory:  {hw.as_dict()['total_memory_gb']} GB")
print(f"bf16 supported: {hw.bf16_supported}")
print(f"precision:      {plan.name}  ({plan.reason})")
print(f"grad scaler:    {plan.use_grad_scaler}")
print(f"experiment:     {cfg.experiment.name}  seed={cfg.experiment.seed}")
print(f"data config:    {cfg.data_config_path}")
print(f"model config:   {cfg.model_config_path}")
print(f"train split:    {cfg.data.train_path}")
if hw.device_kind != "cuda":
    print("\nNOTE: no CUDA device visible — this will train on "
          f"{hw.device_kind} in fp32. Fine for smoke tests, not for a real run.")
PY

    DATA_CONFIG="$($PYTHON -c "
from articulm.config import load_train_config
print(load_train_config('$TRAIN_CONFIG').data_config_path)
")"
    log "resolved data config: $DATA_CONFIG"
else
    DATA_CONFIG="$($PYTHON -c "
from articulm.config import load_train_config
print(load_train_config('$TRAIN_CONFIG').data_config_path)
")"
fi

# --------------------------------------------------------------------------
# Stage 1 — dataset validation
# --------------------------------------------------------------------------
if run_stage validate; then
    step "Stage 1/4 — dataset validation"
    log "validating splits declared in $DATA_CONFIG"
    if ! $PYTHON -m articulm.data.validate \
            --config "$DATA_CONFIG" \
            --json-out "$LOG_DIR/data_report.json" \
            2>&1 | tee "$LOG_DIR/validate.log"; then
        die "dataset validation failed — see $LOG_DIR/validate.log"
    fi
    log "data report: $LOG_DIR/data_report.json"
fi

# --------------------------------------------------------------------------
# Stage 2 — tiny-overfit gate (mandatory)
# --------------------------------------------------------------------------
GATE_RUN_DIR="$LOG_DIR/tiny_overfit_gate"

if run_stage gate; then
    step "Stage 2/4 — tiny-overfit gate"
    if [[ "$SKIP_GATE" == "1" ]]; then
        cat <<'EOF'
!! --skip-gate was passed.

The project's non-negotiable constraints require the tiny-overfit gate to pass
before any long training run. Skipping it means a masking bug, a broken
gradient path or a NaN can burn GPU hours undetected. Proceeding anyway.
EOF
        sleep 3
    else
        [[ -f "$GATE_CONFIG" ]] || die "gate config not found: $GATE_CONFIG" 2
        log "running gate: $GATE_CONFIG -> $GATE_RUN_DIR"

        gate_args=(--config "$GATE_CONFIG" --run-dir "$GATE_RUN_DIR")
        [[ -n "$DEVICE" ]] && gate_args+=(--device "$DEVICE")

        if ! $PYTHON -m articulm.train "${gate_args[@]}" \
                2>&1 | tee "$LOG_DIR/gate_train.log"; then
            die "the tiny-overfit run itself failed — see $LOG_DIR/gate_train.log" 3
        fi

        log "verifying gate criteria (docs/09 M2)"
        if ! $PYTHON -m articulm.gate \
                --run-dir "$GATE_RUN_DIR" \
                --min-accuracy "$GATE_MIN_ACCURACY" \
                --max-strength-mae "$GATE_MAX_MAE" \
                --json-out "$LOG_DIR/gate_result.json" \
                2>&1 | tee "$LOG_DIR/gate_check.log"; then
            die "tiny-overfit gate FAILED — see $LOG_DIR/gate_check.log.
Do not start a long training run. Debug the implementation first." 3
        fi
        log "gate passed: $LOG_DIR/gate_result.json"
    fi
fi

# --------------------------------------------------------------------------
# Stage 3 — pre-flight report for the real config
# --------------------------------------------------------------------------
RUN_DIR="$LOG_DIR/train"

if run_stage dryrun; then
    step "Stage 3/4 — pre-flight report (no training)"
    dry_args=(--config "$TRAIN_CONFIG" --dry-run --run-dir "$LOG_DIR/dryrun")
    [[ -n "$DEVICE" ]] && dry_args+=(--device "$DEVICE")
    if ! $PYTHON -m articulm.train "${dry_args[@]}" \
            2>&1 | tee "$LOG_DIR/dryrun.log"; then
        die "pre-flight failed — see $LOG_DIR/dryrun.log"
    fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
    step "stopping before stage 4 (--dry-run)"
    log "artifacts in $LOG_DIR"
    log "re-run without --dry-run to start real training"
    exit 0
fi

# --------------------------------------------------------------------------
# Stage 4 — real training
# --------------------------------------------------------------------------
if run_stage train; then
    step "Stage 4/4 — training"
    train_args=(--config "$TRAIN_CONFIG" --run-dir "$RUN_DIR")
    [[ -n "$DEVICE" ]] && train_args+=(--device "$DEVICE")
    [[ -n "$RESUME" ]] && train_args+=(--resume "$RESUME")
    [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]] && train_args+=("${EXTRA_TRAIN_ARGS[@]}")

    log "run dir:     $RUN_DIR"
    log "command:     $PYTHON -m articulm.train ${train_args[*]}"

    if [[ "$BACKGROUND" == "1" ]]; then
        nohup $PYTHON -m articulm.train "${train_args[@]}" \
            > "$LOG_DIR/train.log" 2>&1 &
        pid=$!
        echo "$pid" > "$LOG_DIR/train.pid"
        log "detached, pid $pid"
        log "follow:  tail -f $LOG_DIR/train.log"
        log "events:  tail -f $RUN_DIR/logs/events.jsonl"
        log "stop:    kill $pid"
        exit 0
    fi

    if ! $PYTHON -m articulm.train "${train_args[@]}" \
            2>&1 | tee "$LOG_DIR/train.log"; then
        die "training failed — see $LOG_DIR/train.log"
    fi

    step "done"
    log "run dir:  $RUN_DIR"
    log "summary:  $RUN_DIR/metrics/training_summary.json"
    log "events:   $RUN_DIR/logs/events.jsonl"
    log "best:     $RUN_DIR/checkpoints/best.pt"
    cat <<EOF

Next steps:

  # evaluate — synthetic and Human Gold are separate reports, never merged
  $PYTHON -m articulm.evaluate \\
      --checkpoint $RUN_DIR/checkpoints/best.pt \\
      --data <test.jsonl> --label-set synthetic \\
      --out-dir $RUN_DIR/reports/synthetic_test

  # resume if the run was interrupted
  scripts/run_training.sh --config $TRAIN_CONFIG --skip-gate \\
      --resume $RUN_DIR/checkpoints/last.pt
EOF
fi
