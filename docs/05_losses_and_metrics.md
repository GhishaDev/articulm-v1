# Losses and Metrics

Viseme:

```text
CrossEntropyLoss
label_smoothing = 0.05
```

Strength:

```text
target_norm = strength / 100
prediction_norm = sigmoid(raw)
SmoothL1Loss / Huber
```

Synthetic:

```text
L_total = 1.0*L_viseme + 0.3*L_strength
```

Human Gold:

```text
L_total = 1.0*L_viseme + 1.0*L_strength
```

All losses must ignore padding.

Viseme metrics:

- accuracy
- macro F1
- weighted F1
- per-class precision/recall/F1
- confusion matrix

Strength metrics:

- MAE
- RMSE
- per-viseme MAE
- median absolute error

Slice metrics:

- language
- tone
- stress
- syllable role
- phrase/word position
- sequence length bucket

Synthetic and Human Gold metrics must be reported separately.

For the virtual-anchor product, also run perceptual A/B or MOS on:

- lip-shape correctness
- transition naturalness
- Strength naturalness
- overall talking-head naturalness
