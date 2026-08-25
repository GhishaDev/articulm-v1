# Training Operations Guide

Recommended environment:

```text
Python 3.11
PyTorch 2.x
CUDA matching the host
```

Hardware:

```text
V100 16/32GB → FP16
A100/H100 → BF16 preferred
```

Typical workflow:

```bash
python -m articulm.data.validate   --config config/data_v1.yaml

python -m articulm.train   --config config/train_tiny_overfit.yaml

python -m articulm.train   --config config/train_v1_50m.yaml

python -m articulm.train   --config config/train_v1_50m.yaml   --resume checkpoints/last.pt

python -m articulm.evaluate   --config config/train_v1_50m.yaml   --checkpoint checkpoints/best.pt

python -m articulm.infer   --config config/model_v1_50m.yaml   --checkpoint checkpoints/best.pt   --input examples/sample.jsonl
```

Recommended run layout:

```text
runs/
  articulm_v1_50m_YYYYMMDD_HHMMSS/
    model_config.yaml
    train_config.yaml
    data_config.yaml
    vocab/
    logs/
    checkpoints/
    metrics/
    predictions/
```

Monitor:

- total loss
- Viseme loss
- Strength loss
- validation accuracy
- validation macro-F1
- validation Strength MAE
- LR
- grad norm
- tokens/s
- GPU memory
