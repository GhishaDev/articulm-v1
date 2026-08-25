# Training Launcher Guide — `scripts/run_training.sh`

Operations guide for the ArticuLM-V1 training launcher. Companion to
`docs/11_training_operations.md`, which describes the underlying commands; this
document describes the script that sequences them and enforces the gate.

---

## 1. What it is, and when to use it

`scripts/run_training.sh` chains the documented workflow in the order the
project's non-negotiable constraints require, and **refuses to start a long
training run until the tiny-overfit gate has passed on measured metrics**.

```text
Stage 0   environment / GPU / precision preflight
Stage 1   dataset validation            fail fast on schema or label violations
Stage 2   tiny-overfit gate             MANDATORY — aborts the run if it fails
Stage 3   pre-flight report             real config, no training
Stage 4   real training
```

| Use the launcher when | Use `python -m articulm.train` directly when |
|---|---|
| Starting a real training run | Debugging a single stage interactively |
| Running on a shared GPU box | Driving training from your own Python code |
| You want the gate enforced | You are inside a notebook or another harness |
| You want all stage logs in one place | You already ran the gate manually |

The launcher does not implement any training logic. It calls
`articulm.data.validate`, `articulm.train` and `articulm.gate`, and adds
sequencing, gating and log collection.

---

## 2. Quick start

```bash
# from the repository root
scripts/run_training.sh
```

That runs all five stages with the defaults below and blocks until training
finishes. For a first run on a new machine, prefer:

```bash
# see everything the launcher would do, stop before real training
scripts/run_training.sh --dry-run
```

---

## 3. Defaults

| Setting | Default | Override |
|---|---|---|
| Training config | `config/train_v1_50m.yaml` | `--config` |
| Gate config | `config/train_tiny_overfit.yaml` | `--gate-config` |
| Data config | read from the training config | — |
| Stage selection | `all` | `--stage` |
| Device | auto (CUDA > MPS > CPU) | `--device` |
| Gate accuracy threshold | `0.95` | `--gate-min-accuracy` |
| Gate strength-MAE threshold | `5.0` (0..100 units) | `--gate-max-mae` |
| Log directory | `runs/launcher/<YYYYmmdd_HHMMSS>` | `--log-dir` |
| Python interpreter | `python3` | `PYTHON` env var |

The data config is never passed explicitly — the launcher resolves it from the
training config's `data_config:` field, so the two can never drift apart.

---

## 4. Options

```text
--config PATH            training config (default config/train_v1_50m.yaml)
--gate-config PATH       tiny-overfit config
--resume PATH            checkpoint to resume the real run from
--device DEV             force cuda / cpu / mps
--stage NAME             all | preflight | validate | gate | dryrun | train
--skip-gate              skip the mandatory tiny-overfit gate (discouraged)
--gate-min-accuracy F    gate viseme-accuracy threshold
--gate-max-mae F         gate strength-MAE threshold, 0..100 units
--dry-run                run stages 0-3 and stop before real training
--background             detach the real training run with nohup
--log-dir PATH           launcher log directory
-- ARGS...               pass everything after -- to `python -m articulm.train`
-h, --help               help
```

`--` passthrough example:

```bash
# cap the corpus and force a run directory, straight through to articulm.train
scripts/run_training.sh -- --limit 5000 --vocab runs/prev/vocab/feature_vocab.json
```

---

## 5. Exit codes

| Code | Meaning | What to do |
|---|---|---|
| `0` | All requested stages succeeded | Proceed to evaluation |
| `1` | A stage failed (validation, pre-flight, or training) | Read the named log |
| `2` | Bad usage — unknown option, missing config file | Fix the command |
| `3` | **Tiny-overfit gate failed** | Do not train. Debug the implementation |

Exit `3` is deliberately distinct from `1` so CI can treat "the gate says the
model is broken" differently from "the disk filled up".

---

## 6. Stage reference

### Stage 0 — preflight

Prints and records the resolved environment: torch/CUDA versions, device name,
compute capability, device memory, bf16 support, and the **precision the
trainer will actually select** with its reason.

Read this line before every real run:

```text
precision:      bf16  (auto: NVIDIA A100-SXM4-40GB supports bf16)
```

If it says `fp32 (cpu device runs fp32)` on a machine you believe has a GPU,
stop — CUDA is not visible to this interpreter and the run would be useless.

Artifact: `preflight.txt`

### Stage 1 — dataset validation

Runs `articulm.data.validate` over every split declared in the data config.
Reports sentence/token counts, sequence-length stats, per-viseme distribution,
strength histogram, language/tone/stress/boundary distributions, unknown
phoneme rate and label sources.

Fails on any schema violation, out-of-range label, NaN/Inf, or token/label
count mismatch. Nothing is auto-repaired.

Artifacts: `validate.log`, `data_report.json`

### Stage 2 — tiny-overfit gate (mandatory)

Trains the gate config to completion, then verifies the **metrics** via
`python -m articulm.gate`. Ten checks, all of which must pass:

```text
training actually ran                              global_step > 0
no NaN/Inf during training                         saw_non_finite_loss is False
final train loss is finite
viseme train accuracy approaches a perfect fit     >= --gate-min-accuracy
strength error converged                           <= --gate-max-mae
validation metrics present
validation viseme accuracy approaches a perfect fit
validation strength error converged
last.pt loads
best.pt loads
```

**Why metric-based and not exit-code-based:** a 2-step run also exits `0`. The
command succeeding says nothing about whether the model can fit. Checking the
numbers is the entire point of the gate.

Artifacts: `gate_train.log`, `gate_check.log`, `gate_result.json`,
`tiny_overfit_gate/` (a full run directory with its own checkpoints)

### Stage 3 — pre-flight report

Runs `articulm.train --dry-run` against the real config: parameter breakdown,
layers × hidden, position encoding, Local Conv on/off, device and precision,
batching strategy, estimated activation and optimizer memory, loss weights.

Nothing is trained, but the run inputs (`model_config.yaml`, `data_config.yaml`,
`train_config.yaml`, `vocab/`) are still written for provenance.

Artifacts: `dryrun.log`, `dryrun/`

### Stage 4 — real training

Runs `articulm.train` with the real config into `<log-dir>/train/`. On
completion it prints the follow-up evaluation and resume commands.

Artifacts: `train.log`, `train/` (run directory), `train.pid` when detached

---

## 7. Common recipes

### First run on a new GPU machine

```bash
# 1. confirm precision selection and data health without training
scripts/run_training.sh --dry-run

# 2. if the report looks right, run for real in the background
scripts/run_training.sh --background
tail -f runs/launcher/<ts>/train.log
```

### Long run on a shared box

```bash
scripts/run_training.sh --background --log-dir runs/launcher/baseline_v1
# ...
cat runs/launcher/baseline_v1/train.pid          # the pid
tail -f runs/launcher/baseline_v1/train/logs/events.jsonl
kill "$(cat runs/launcher/baseline_v1/train.pid)"  # stop it
```

`events.jsonl` is the structured log: one JSON object per event with
`run_id`, `stage`, `elapsed_s`, plus loss / accuracy / lr / grad_norm /
tokens_per_s for `train_step` events. Use it for curves, not `train.log`.

### Resume after an interruption

```bash
scripts/run_training.sh \
  --resume runs/launcher/<ts>/train/checkpoints/last.pt \
  --skip-gate
```

`--skip-gate` is appropriate here and only here: the gate already passed for
this code, and re-running it would waste time. The checkpoint restores model,
optimizer, scheduler, scaler, step, epoch and RNG state.

### Just check the gate, change nothing else

```bash
scripts/run_training.sh --stage gate
```

### Re-verify a gate run that already finished

```bash
python -m articulm.gate --run-dir runs/launcher/<ts>/tiny_overfit_gate
python -m articulm.gate --run-dir <dir> --min-accuracy 0.99   # stricter
```

### Local Conv ablation (E3)

```bash
scripts/run_training.sh --config config/train_e3_localconv.yaml
```

### Force CPU for a quick smoke test

```bash
scripts/run_training.sh --device cpu --dry-run
```

---

## 8. Log directory layout

```text
runs/launcher/<timestamp>/
├── preflight.txt              resolved env / device / precision
├── validate.log               stage 1 console output
├── data_report.json           machine-readable dataset report
├── gate_train.log             tiny-overfit training output
├── gate_check.log             the ten gate checks
├── gate_result.json           machine-readable gate verdict
├── tiny_overfit_gate/         full gate run dir (checkpoints, events, summary)
├── dryrun.log                 stage 3 pre-flight report
├── dryrun/                    persisted run inputs (configs + vocab, no checkpoints)
├── train.log                  stage 4 console output
├── train.pid                  present only with --background
└── train/                     the real run directory
    ├── model_config.yaml
    ├── data_config.yaml
    ├── train_config.yaml
    ├── vocab/feature_vocab.json
    ├── logs/events.jsonl
    ├── metrics/training_summary.json
    └── checkpoints/{best.pt,last.pt,step_*.pt}
```

> **Disk warning.** Each 50M checkpoint is ~571 MB measured (weights, gradients
> and AdamW moments). A run with `keep_last_n: 3` holds `best` + `last` + 3 step
> files ≈ 2.8 GB, and the gate stage writes its own set. Budget accordingly.
> The launcher never deletes checkpoints; `keep_last_n` rotation only ever
> removes `step_*.pt` — `best.pt` and `last.pt` are always preserved.

---

## 9. Troubleshooting

### `GATE FAILED` — what now

Read `gate_check.log` and look at *which* checks failed:

| Failing check | Likely cause |
|---|---|
| `no NaN/Inf during training` | LR too high, or a masking bug letting PAD into the loss |
| `viseme train accuracy` low | Gate ran too few steps, or the gradient path is broken |
| `strength error converged` high | Strength head not learning — check the soft-viseme path |
| `validation metrics present` | The gate config has no `validation_path` |
| `best.pt loads` | Disk full or interrupted write |

If the gate genuinely ran too few steps for the model to fit, raise
`max_steps` in the gate config — do **not** lower the thresholds to make it
pass. The thresholds are the acceptance criteria.

### `dataset validation failed`

The message names the file, line, sample id and field. Common causes:

- Chinese token with `surface_tone` outside `1..5`, or `stress != 0`
- English token with `surface_tone != 0`
- `viseme_id` outside `[0,17]` or `strength` outside `[0,100]`
- `viseme_id` / `strength` / `shapeV2` / `Talk` / `raw_value` present at token
  feature level — that is label leakage and is rejected outright
- phoneme count != label count

Fix the data generator, not the validator.

### `dataset file not found: data/train.jsonl`

The default `config/data_v1.yaml` points at the real corpus, which comes from
`articulm_data_pipeline`. Either produce it, or point at another data config:

```bash
scripts/run_training.sh --config config/train_e0_sanity.yaml   # uses data_dev.yaml
```

### Precision says `fp32` on a GPU machine

CUDA is not visible to this interpreter. Check:

```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CPU-only torch wheel is the usual cause. See README section 1 for the correct
install command per host.

### `bf16 requested but ... does not support it`

You asked for bf16 on a V100-class GPU. That is refused by design — V100 must
use fp16. Set `precision: auto` or `precision: fp16` in the training config.

### Out of memory during stage 4

Lower `training.batching.max_phoneme_tokens_per_batch` (dynamic batching) or
`batch_size` (fixed batching), and raise `gradient_accumulation_steps` to keep
the effective batch size. The stage 3 report's activation estimate is an
order-of-magnitude guide only — trust the OOM over the estimate.

---

## 10. Known limitations

- **Single process, single GPU.** There is no DDP or `DistributedSampler`.
  Multi-GPU training requires new work in the trainer.
- **The corpus is parsed eagerly.** Peak host RAM is roughly
  1 KB per phoneme token: about 4 GB at 100k sentences and **42 GB at 1M**.
  Above ~300k sentences a streaming dataset is needed first.
- **`--skip-gate` is a real footgun.** It exists for resume, and prints a
  warning plus a 3-second pause. Do not put it in a default CI command.
- **Bash only.** The script assumes a POSIX shell with bash; it is not tested
  on Windows outside WSL.
- **`--background` uses `nohup`, not a supervisor.** It will not restart a
  crashed run. For long unattended jobs, wrap it in your cluster's scheduler.
