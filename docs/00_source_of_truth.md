# Source of Truth

Primary design source: `ArticuLM-V1_模型介绍(2).pptx`.

## Slide 1 — Positioning

ArticuLM-V1 is a Context-aware Multilingual Phoneme-to-Viseme model for virtual-anchor articulation.

Core flow:

```text
Input Features
→ Feature Embedding / Fusion
→ Context Transformer
→ Viseme + Strength Heads
→ phoneme-aligned sequence output
```

## Slide 2 — Architecture

```text
Phoneme / Tone / Syllable Role / Articulation / Boundary / Language Embeddings
→ Concat ≈384D
→ Linear Projection 384→640
→ 10-layer Context Transformer
→ H ∈ R^(B×T×640)
```

Optional local module:

```text
Local Conv1D
kernel_size = 5
insert every 2 Transformer blocks
```

Viseme Head:

```text
640→256→18→Softmax
```

Soft Viseme Embedding:

```text
18-way probability weighted embedding
dim = 32
```

Strength Head:

```text
640 + 32
→ 672→256→64→1
→ Sigmoid×100
```

## Slide 3 — Input

Unified phoneme-level schema:

```text
phoneme
language
tone / stress
syllable role
articulatory
boundary
```

Convention:

```text
Chinese surface_tone = 1~5
English surface_tone = 0
English stress = 0/1/2
```

## Slide 4 — Output

Per phoneme:

```text
viseme_id ∈ {0,...,17}
strength ∈ [0,100]
```

## Engineering defaults not fixed by the PPT

The following remain configurable implementation choices:

- exact embedding dimensions
- optimizer / scheduler
- label smoothing
- Strength regression loss
- position encoding implementation
- Local Conv enabled/disabled
- synthetic vs Human Gold loss weights

Any change must be reflected in configs, tests and experiment notes.
