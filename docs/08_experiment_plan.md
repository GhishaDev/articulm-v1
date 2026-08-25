# Experiment Plan

E0 — Pipeline sanity:
```text
100–1000 samples
```

E1 — Tiny overfit:
```text
32–256 samples
```

E2 — 50M baseline:
```text
10L × 640
Local Conv OFF
pseudo_strength_weight=0.3
```

E3 — Local Conv ablation:
```text
Local Conv ON
kernel=5
every 2 layers
```

E4 — pseudo Strength weight:
```text
0.1 / 0.3 / 0.5
```

E5 — data scaling:
```text
100k / 300k / 1M sentences
```

E6 — Human Gold fine-tune:
```text
heads only
top layers + heads
full fine-tune
```

E7 — model scaling after baseline stabilizes:
```text
30M / 50M / 100M
```

Use Human Gold scaling curve to decide whether larger models are justified.

Experiment naming:

```text
articulm_v1_50m_e2_baseline_YYYYMMDD
```
