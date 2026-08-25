# Training Plan

## Stage 0 — Smoke Test

Use 100–1000 samples.

Verify:

- forward
- backward
- optimizer step
- checkpoint save/load
- inference after reload

## Stage 1 — Tiny Overfit

Use 32–256 samples.

Expected:

- Viseme train accuracy approaches near-perfect fit
- Strength loss falls strongly
- no NaN/Inf
- no padding/mask bug

If tiny overfit fails, do not start full training.

## Stage 2 — Synthetic Pretraining

Recommended baseline:

```yaml
optimizer: AdamW
learning_rate: 3.0e-4
weight_decay: 0.01
warmup_ratio: 0.05
gradient_clip_norm: 1.0
```

Precision:

```text
V100 → fp16
A100/H100 → bf16 preferred if supported
```

Prefer dynamic batching by phoneme tokens.

Initial target:

```text
max_phoneme_tokens_per_batch ≈ 6000
gradient_accumulation_steps = 2
```

Tune according to actual memory.

Synthetic loss:

```text
L = 1.0 * L_viseme
  + 0.3 * L_pseudo_strength
```

## Stage 3 — Ablation

Required:

```text
Local Conv OFF
vs
Local Conv ON
```

Optional:

- Strength loss weight
- label smoothing
- position encoding

## Stage 4 — Human Gold Fine-tuning

Recommended initial weights:

```text
L = 1.0 * L_viseme
  + 1.0 * L_strength
```

Suggested LR:

```text
backbone: 1e-5 ~ 5e-5
heads:    1e-4 ~ 3e-4
```

Checkpoint must preserve:

- model
- optimizer
- scheduler
- scaler
- step/epoch
- configs
- vocab
- seed

Use validation plateau for early stopping rather than a fixed epoch assumption.
