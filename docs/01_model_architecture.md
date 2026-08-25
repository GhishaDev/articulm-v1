# ArticuLM-V1-50M Architecture Specification

## 1. Model

```text
x_i =
  phoneme
  language
  surface_tone / stress
  syllable_role
  articulatory
  boundary
       ↓
Feature Embeddings
       ↓
Concat 384D
       ↓
Projection 384→640
       ↓
10 × Transformer Encoder
       ↓
H=[h1,...,hT], h_i∈R^640
       ↓
Viseme Head → P(v)
       ↓
Soft Viseme Embedding 32D
       ↓
concat(h_i,e_v)
       ↓
Strength Head
```

## 2. Transformer baseline

```yaml
hidden_size: 640
num_layers: 10
num_heads: 10
head_dim: 64
ffn_size: 2560
dropout: 0.1
attention_dropout: 0.1
activation: gelu
norm: pre_layer_norm
max_seq_len: 256
```

Approximate Transformer parameters:

```text
per layer ≈ 4.915M
10 layers ≈ 49.15M
```

Embeddings, fusion and heads add a relatively small amount, so the full model remains around 50M. Exact count depends on vocab sizes.

## 3. Recommended embedding dimensions

```text
phoneme                  256
language                   8
surface_tone              16
stress                     8
syllable_role             16
articulatory composite    60
boundary                  20
-----------------------------
total                     384
```

Articulatory composite:

```text
type         8
height       8
backness     8
rounded      4
place       12
manner      12
voiced       4
aspirated    4
-------------
total       60
```

These dimensions are engineering defaults.

## 4. Feature Fusion

```text
[B,T,384]
→ Linear 384→640
→ LayerNorm
→ GELU
→ Dropout
→ [B,T,640]
```

## 5. Position Encoding

Recommended default: RoPE. Keep configurable.

## 6. Optional Local Coarticulation Module

```text
kernel_size = 5
every_n_layers = 2
```

Recommended implementation: depthwise Conv1D + activation + residual.

Baseline config keeps it OFF; run an explicit ON/OFF ablation.

## 7. Viseme Head

```text
[B,T,640]
→ Linear 640→256
→ GELU
→ Dropout
→ Linear 256→18
→ logits [B,T,18]
```

Inference:
```text
viseme_id = argmax(logits)
```

## 8. Soft Viseme Embedding

Trainable:
```text
E ∈ R^(18×32)
```

Compute:
```text
P = softmax(logits)
e_v = P @ E
```

Shape:
```text
[B,T,32]
```

## 9. Strength Head

```text
concat(H,e_v) → [B,T,672]
→ 672→256
→ GELU
→ Dropout
→ 256→64
→ GELU
→ 64→1
→ Sigmoid
```

Training target:
```text
strength_norm = strength / 100
```

Inference:
```text
strength = strength_norm * 100
```

## 10. Masking

Padding must be excluded from:

- attention
- Viseme loss
- Strength loss
- metrics
