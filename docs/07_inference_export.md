# Inference and Export

Inference inputs are the same non-label features used during training.

Per phoneme output:

```json
{
  "phoneme": "n",
  "viseme_id": 14,
  "strength": 62.0
}
```

Baseline:

```text
viseme_id = argmax(viseme_logits)
strength = sigmoid(raw_strength) * 100
```

Do not silently add smoothing or heuristics.

Export priority:

1. PyTorch checkpoint
2. torch.export / TorchScript if needed
3. ONNX only after parity tests

For ONNX:

- dynamic batch axis
- dynamic sequence axis
- numerical parity test

Benchmark:

- hardware
- batch size
- sequence length
- precision
- p50 / p95 latency
- throughput

Each export should be accompanied by:

- model version
- config hash
- phoneme vocab version
- feature vocab version
- training-data version
- checkpoint step
