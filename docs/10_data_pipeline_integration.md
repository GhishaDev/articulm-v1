# Integration with articulm_data_pipeline

Recommended upstream:

```text
articulm_data_pipeline
```

The model repository should consume a stable versioned JSONL export.

Do not couple training code directly to:

- GLM API
- browser automation
- Viseme Lab page structure

Recommended flow:

```text
articulm_data_pipeline
→ selected sentences
→ phoneme / linguistic features
→ teacher/pseudo labels
→ versioned JSONL
→ ArticuLM model training
```

Dataset manifest example:

```json
{
  "dataset_version": "v1.0",
  "num_sentences": 100000,
  "num_phoneme_tokens": 0,
  "languages": ["zh"],
  "source_batches": ["batch_001"],
  "viseme_label_source": "teacher_rule",
  "strength_label_source": "pseudo_strength_v1",
  "schema_version": "articulm_v1_sample_v1"
}
```

Training should fail fast on schema mismatch or invalid labels.
